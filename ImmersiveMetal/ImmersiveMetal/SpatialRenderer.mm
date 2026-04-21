#include "SpatialRenderer.h"
#include "Mesh.h"
#include "ShaderTypes.h"

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <Spatial/Spatial.h>
#import <Vision/Vision.h>

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cmath>

static void SetDebugStatus(Video3DConfiguration *configuration, NSString *status) {
    if (configuration == nil || status == nil) {
        return;
    }
    if ([configuration.rendererDebugStatus isEqualToString:status]) {
        return;
    }
    configuration.rendererDebugStatus = status;
    NSLog(@"DepthPlayerRenderer: %@", status);
}

static MTLPixelFormat metalPixelFormatForVideoPixelBuffer(CVPixelBufferRef pixelBuffer) {
    switch (CVPixelBufferGetPixelFormatType(pixelBuffer)) {
        case kCVPixelFormatType_64RGBAHalf:
            return MTLPixelFormatRGBA16Float;
        case kCVPixelFormatType_32BGRA:
            return MTLPixelFormatBGRA8Unorm;
        default:
            return MTLPixelFormatInvalid;
    }
}

static simd_float4x4 matrix_float4x4_from_double4x4(simd_double4x4 m) {
    return simd_matrix(simd_make_float4(m.columns[0][0], m.columns[0][1], m.columns[0][2], m.columns[0][3]),
                       simd_make_float4(m.columns[1][0], m.columns[1][1], m.columns[1][2], m.columns[1][3]),
                       simd_make_float4(m.columns[2][0], m.columns[2][1], m.columns[2][2], m.columns[2][3]),
                       simd_make_float4(m.columns[3][0], m.columns[3][1], m.columns[3][2], m.columns[3][3]));
}

static simd_float4x4 matrix_scale(float sx, float sy, float sz) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0][0] = sx;
    m.columns[1][1] = sy;
    m.columns[2][2] = sz;
    return m;
}

static MLModel *LoadBundledDepthModel() {
    // Try known model names so the app can run with either renamed or legacy bundles.
    NSArray<NSString *> *candidateNames = @[@"DepthAnythingV2SmallF16", @"DepthAnythingSmallF16", @"DepthAnythingV2Small"];
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = MLComputeUnitsAll;

    for (NSString *name in candidateNames) {
        NSURL *compiledURL = [[NSBundle mainBundle] URLForResource:name withExtension:@"mlmodelc"];
        if (compiledURL == nil) {
            continue;
        }

        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:compiledURL configuration:configuration error:&error];
        if (model != nil) {
            NSLog(@"Loaded depth model: %@", name);
            return model;
        }

        NSLog(@"Failed loading depth model %@: %@", name, error.localizedDescription);
    }

    NSLog(@"No bundled DepthAnything model found. Add a compiled .mlmodelc to the app bundle.");
    return nil;
}

static bool CopyDepthObservationToFloatBuffer(VNRequest *request,
                                              std::vector<float> &outDepth,
                                              uint32_t &outWidth,
                                              uint32_t &outHeight)
{
    if (request.results.count == 0) {
        return false;
    }

    id observation = request.results.firstObject;

    // Some CoreML depth models expose MLMultiArray while others expose image outputs.
    if ([observation isKindOfClass:[VNCoreMLFeatureValueObservation class]]) {
        VNCoreMLFeatureValueObservation *featureObservation = (VNCoreMLFeatureValueObservation *)observation;
        if (featureObservation.featureValue.type != MLFeatureTypeMultiArray) {
            return false;
        }

        MLMultiArray *multiArray = featureObservation.featureValue.multiArrayValue;
        if (multiArray == nil || multiArray.shape.count < 2) {
            return false;
        }

        outHeight = (uint32_t)multiArray.shape[multiArray.shape.count - 2].unsignedIntegerValue;
        outWidth = (uint32_t)multiArray.shape[multiArray.shape.count - 1].unsignedIntegerValue;
        if (outWidth == 0 || outHeight == 0) {
            return false;
        }

        outDepth.resize((size_t)outWidth * outHeight);

        const NSInteger widthIndex = multiArray.shape.count - 1;
        const NSInteger heightIndex = multiArray.shape.count - 2;
        const NSInteger widthStride = multiArray.strides[widthIndex].integerValue;
        const NSInteger heightStride = multiArray.strides[heightIndex].integerValue;
        const MLMultiArrayDataType dataType = multiArray.dataType;
        const char *base = (const char *)multiArray.dataPointer;

        auto readValueAtOffset = [&](NSInteger elementOffset) -> float {
            switch (dataType) {
                case MLMultiArrayDataTypeDouble:
                    return (float)((const double *)base)[elementOffset];
                case MLMultiArrayDataTypeFloat32:
                    return ((const float *)base)[elementOffset];
                case MLMultiArrayDataTypeFloat16:
                    return (float)((const __fp16 *)base)[elementOffset];
                case MLMultiArrayDataTypeInt32:
                    return (float)((const int32_t *)base)[elementOffset];
                default:
                    return 0.5f;
            }
        };

        float minDepth = FLT_MAX;
        float maxDepth = -FLT_MAX;
        for (uint32_t y = 0; y < outHeight; ++y) {
            for (uint32_t x = 0; x < outWidth; ++x) {
                const NSInteger offset = (NSInteger)y * heightStride + (NSInteger)x * widthStride;
                float v = readValueAtOffset(offset);
                minDepth = std::min(minDepth, v);
                maxDepth = std::max(maxDepth, v);
                outDepth[(size_t)y * outWidth + x] = v;
            }
        }

        const float denom = std::max(maxDepth - minDepth, 1e-6f);
        for (float &v : outDepth) {
            v = std::clamp((v - minDepth) / denom, 0.0f, 1.0f);
        }

        return true;
    }

    if ([observation isKindOfClass:[VNPixelBufferObservation class]]) {
        VNPixelBufferObservation *pixelObservation = (VNPixelBufferObservation *)observation;
        CVPixelBufferRef pixelBuffer = pixelObservation.pixelBuffer;
        if (pixelBuffer == nil) {
            return false;
        }

        outWidth = (uint32_t)CVPixelBufferGetWidth(pixelBuffer);
        outHeight = (uint32_t)CVPixelBufferGetHeight(pixelBuffer);
        if (outWidth == 0 || outHeight == 0) {
            return false;
        }

        OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
        if (!(pixelFormat == kCVPixelFormatType_OneComponent8 ||
              pixelFormat == kCVPixelFormatType_OneComponent16Half ||
              pixelFormat == kCVPixelFormatType_OneComponent16 ||
              pixelFormat == kCVPixelFormatType_OneComponent32Float)) {
            return false;
        }

        outDepth.resize((size_t)outWidth * outHeight);

        CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
        const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

        float minDepth = FLT_MAX;
        float maxDepth = -FLT_MAX;
        for (uint32_t y = 0; y < outHeight; ++y) {
            const uint8_t *row = base + (size_t)y * bytesPerRow;
            for (uint32_t x = 0; x < outWidth; ++x) {
                float v = 0.5f;
                switch (pixelFormat) {
                    case kCVPixelFormatType_OneComponent8:
                        v = ((const uint8_t *)row)[x] / 255.0f;
                        break;
                    case kCVPixelFormatType_OneComponent16Half:
                        v = (float)((const __fp16 *)row)[x];
                        break;
                    case kCVPixelFormatType_OneComponent16:
                        v = ((const uint16_t *)row)[x] / 65535.0f;
                        break;
                    case kCVPixelFormatType_OneComponent32Float:
                        v = ((const float *)row)[x];
                        break;
                    default:
                        break;
                }
                minDepth = std::min(minDepth, v);
                maxDepth = std::max(maxDepth, v);
                outDepth[(size_t)y * outWidth + x] = v;
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

        const float denom = std::max(maxDepth - minDepth, 1e-6f);
        for (float &v : outDepth) {
            v = std::clamp((v - minDepth) / denom, 0.0f, 1.0f);
        }

        return true;
    }

    return false;
}

SpatialRenderer::SpatialRenderer(cp_layer_renderer_t layerRenderer, Video3DConfiguration *configuration) :
    _layerRenderer { layerRenderer },
    _configuration { configuration },
    _lastRenderTime(CACurrentMediaTime()),
    _lastDepthTimestamp(0.0),
    _latestDepthTimestamp(0.0),
    _lastInferenceTime(0.0),
    _disparityStrength(110.0f),
    _temporalSmoothing(0.24f),
    _depthWidth(0),
    _depthHeight(0),
    _frameSkipCounter(0),
    _hasDepthModel(false),
    _depthInferenceInFlight(false)
{
    _device = cp_layer_renderer_get_device(layerRenderer);
    _commandQueue = [_device newCommandQueue];
    CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, _device, nil, &_videoTextureCache);

    MLModel *depthModel = LoadBundledDepthModel();
    if (depthModel != nil) {
        NSError *error = nil;
        VNCoreMLModel *visionModel = [VNCoreMLModel modelForMLModel:depthModel error:&error];
        if (visionModel != nil) {
            _depthRequest = [[VNCoreMLRequest alloc] initWithModel:visionModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable err) {
                if (err != nil) {
                    _depthInferenceInFlight = false;
                    return;
                }

                std::vector<float> inferredDepth;
                uint32_t w = 0;
                uint32_t h = 0;
                if (!CopyDepthObservationToFloatBuffer(request, inferredDepth, w, h)) {
                    _depthInferenceInFlight = false;
                    return;
                }

                std::lock_guard<std::mutex> lock(_depthMutex);
                if (_smoothedDepth.size() != inferredDepth.size()) {
                    // First valid frame initializes the temporal history buffer.
                    _smoothedDepth = inferredDepth;
                    _depthWidth = w;
                    _depthHeight = h;
                } else {
                    // Enhanced temporal smoothing with edge-preserving bilateral filtering.
                    // Uses motion detection to adapt smoothing strength.
                    const float minAlpha = 0.04f;  // Stronger temporal persistence
                    const float maxAlpha = std::clamp(_temporalSmoothing, 0.12f, 0.50f);
                    float totalMotion = 0.0f;  // Detect scene motion
                    uint32_t motionPixels = 0;

                    // First pass: detect motion and edge boundaries
                    std::vector<float> edgeStrength(inferredDepth.size(), 0.0f);
                    for (uint32_t y = 1; y < h - 1; ++y) {
                        for (uint32_t x = 1; x < w - 1; ++x) {
                            const size_t idx = (size_t)y * w + x;
                            const float d = inferredDepth[idx];
                            
                            // Sobel-like edge detection on depth
                            const float dx = (inferredDepth[idx + 1] - inferredDepth[idx - 1]) * 0.5f;
                            const float dy = (inferredDepth[idx + w] - inferredDepth[idx - w]) * 0.5f;
                            edgeStrength[idx] = std::sqrt(dx*dx + dy*dy);
                            
                            // Accumulate motion
                            const float delta = std::fabs(d - _smoothedDepth[idx]);
                            totalMotion += delta;
                            if (delta > 0.02f) motionPixels++;
                        }
                    }
                    float motionRatio = motionPixels / static_cast<float>(w * h);
                    // In high-motion scenes, reduce smoothing to preserve detail
                    float motionFactor = 1.0f - std::clamp(motionRatio * 2.0f, 0.0f, 0.6f);

                    // Second pass: edge-aware temporal blending
                    for (uint32_t y = 0; y < h; ++y) {
                        const uint32_t yPrev = (y == 0) ? 0 : (y - 1);
                        const uint32_t yNext = (y + 1 >= h) ? (h - 1) : (y + 1);

                        for (uint32_t x = 0; x < w; ++x) {
                            const uint32_t xPrev = (x == 0) ? 0 : (x - 1);
                            const uint32_t xNext = (x + 1 >= w) ? (w - 1) : (x + 1);
                            const size_t idx = (size_t)y * w + x;

                            float currentDepth = inferredDepth[idx];
                            if (!std::isfinite(currentDepth)) {
                                currentDepth = _smoothedDepth[idx];
                            }
                            currentDepth = std::clamp(currentDepth, 0.0f, 1.0f);

                            const float previousDepth = _smoothedDepth[idx];
                            const float temporalDelta = std::fabs(currentDepth - previousDepth);
                            // Exponential decay for temporal confidence
                            const float temporalConfidence = std::exp(-temporalDelta * 12.0f);

                            // Enhanced spatial gradient with multi-neighbor analysis
                            const float left = inferredDepth[(size_t)y * w + xPrev];
                            const float right = inferredDepth[(size_t)y * w + xNext];
                            const float up = inferredDepth[(size_t)yPrev * w + x];
                            const float down = inferredDepth[(size_t)yNext * w + x];
                            const float localGradient = std::max(std::fabs(currentDepth - left),
                                                         std::max(std::fabs(currentDepth - right),
                                                         std::max(std::fabs(currentDepth - up), std::fabs(currentDepth - down))));
                            
                            // Preserve edges: high gradient = low smoothing
                            const float isEdge = std::clamp(localGradient * 3.5f, 0.0f, 1.0f);
                            const float edgeFactor = 1.0f - isEdge * 0.8f;  // Edges get less smoothing
                            const float spatialConfidence = 1.0f - std::clamp(localGradient * 2.5f, 0.0f, 1.0f);

                            // Combined confidence with motion and edge awareness
                            float confidence = std::clamp(0.65f * temporalConfidence + 0.35f * spatialConfidence, 0.0f, 1.0f);
                            confidence *= edgeFactor * motionFactor;
                            
                            const float alpha = minAlpha + (maxAlpha - minAlpha) * confidence;
                            _smoothedDepth[idx] = alpha * currentDepth + (1.0f - alpha) * previousDepth;
                        }
                    }
                }

                _depthInferenceInFlight = false;
            }];
            _depthRequest.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
            _hasDepthModel = true;
        } else {
            NSLog(@"Failed creating VNCoreMLModel: %@", error.localizedDescription);
        }
    }

    makeResources();

    makeRenderPipelines();
}

void SpatialRenderer::makeResources() {
    // Quad dimensions define the virtual screen where converted video is rendered.
    _videoQuadMesh = std::make_unique<StereoQuadMesh>(1.6f, 0.9f, @"bluemarble.png", _device);
}

void SpatialRenderer::ensureDepthMapTexture(size_t videoWidth, size_t videoHeight) {
    if (videoWidth == 0 || videoHeight == 0) {
        return;
    }

    static const size_t kMaxDepthDimension = 1024;
    float scale = 1.0f;
    const size_t longest = std::max(videoWidth, videoHeight);
    if (longest > kMaxDepthDimension) {
        scale = (float)kMaxDepthDimension / (float)longest;
    }

    const size_t targetWidth = std::max((size_t)1, (size_t)std::round((float)videoWidth * scale));
    const size_t targetHeight = std::max((size_t)1, (size_t)std::round((float)videoHeight * scale));

    if (_depthMapTexture != nil &&
        _depthMapTexture.width == targetWidth &&
        _depthMapTexture.height == targetHeight) {
        return;
    }

    MTLTextureDescriptor *depthDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                           width:targetWidth
                                                                                          height:targetHeight
                                                                                       mipmapped:NO];
    depthDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    depthDesc.storageMode = MTLStorageModeShared;
    _depthMapTexture = [_device newTextureWithDescriptor:depthDesc];
    _videoQuadMesh->setDepthTexture(_depthMapTexture);
}

void SpatialRenderer::makeRenderPipelines() {
    NSError *error = nil;
    cp_layer_renderer_configuration_t layerConfiguration = cp_layer_renderer_get_configuration(_layerRenderer);
    cp_layer_renderer_layout layout = cp_layer_renderer_configuration_get_layout(layerConfiguration);

    MTLRenderPipelineDescriptor *pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.colorAttachments[0].pixelFormat = cp_layer_renderer_configuration_get_color_format(layerConfiguration);
    pipelineDescriptor.depthAttachmentPixelFormat = cp_layer_renderer_configuration_get_depth_format(layerConfiguration);

    id<MTLLibrary> library = [_device newDefaultLibrary];
    id<MTLFunction> vertexFunction, fragmentFunction;

    BOOL layoutIsDedicated = (layout == cp_layer_renderer_layout_dedicated);
    BOOL layoutIsLayered = (layout == cp_layer_renderer_layout_layered);

    MTLFunctionConstantValues *functionConstants = [MTLFunctionConstantValues new];
    [functionConstants setConstantValue:&layoutIsLayered type:MTLDataTypeBool withName:@"useLayeredRendering"];

    {
        vertexFunction = [library newFunctionWithName: layoutIsDedicated ? @"vertex_dedicated_panel_main" : @"vertex_panel_main"
                                       constantValues:functionConstants
                                                error:&error];
        fragmentFunction = [library newFunctionWithName:@"fragment_stereo_conversion"];
        pipelineDescriptor.vertexFunction = vertexFunction;
        pipelineDescriptor.fragmentFunction = fragmentFunction;
        pipelineDescriptor.vertexDescriptor = _videoQuadMesh->vertexDescriptor();
        if (!layoutIsDedicated) {
            pipelineDescriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            pipelineDescriptor.maxVertexAmplificationCount = 2;
        }

        _contentRenderPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
        if (_contentRenderPipelineState == nil) {
            NSLog(@"Error occurred when creating render pipeline state: %@", error);
        }
    }
    {
        id<MTLFunction> computeFunction = [library newFunctionWithName:@"estimate_depth_from_luma"];
        if (computeFunction != nil) {
            _fallbackDepthComputePipelineState = [_device newComputePipelineStateWithFunction:computeFunction error:&error];
            if (_fallbackDepthComputePipelineState == nil) {
                NSLog(@"Error creating fallback depth compute pipeline: %@", error);
            }
        } else {
            NSLog(@"Failed to load estimate_depth_from_luma compute function.");
        }
    }

    MTLDepthStencilDescriptor *depthDescriptor = [MTLDepthStencilDescriptor new];
    depthDescriptor.depthWriteEnabled = YES;
    depthDescriptor.depthCompareFunction = MTLCompareFunctionGreater;
    _contentDepthStencilState = [_device newDepthStencilStateWithDescriptor:depthDescriptor];
}

void SpatialRenderer::drawAndPresent(cp_frame_t frame, cp_drawable_t drawable) {
    CFTimeInterval renderTime = CACurrentMediaTime();

    cp_layer_renderer_configuration_t layerConfiguration = cp_layer_renderer_get_configuration(_layerRenderer);
    cp_layer_renderer_layout layout = cp_layer_renderer_configuration_get_layout(layerConfiguration);

    ar_device_anchor_t deviceAnchor = cp_drawable_get_device_anchor(drawable);
    simd_float4x4 headPose = ar_anchor_get_origin_from_anchor_transform(deviceAnchor);

    simd_float4x4 panelOffset = matrix_identity_float4x4;
    panelOffset.columns[3] = simd_make_float4((float)_configuration.panelOffsetX,
                                              (float)_configuration.panelOffsetY,
                                              (float)_configuration.panelOffsetZ,
                                              1.0f);
    _disparityStrength = std::clamp((float)_configuration.disparityStrength, 0.0f, 50.0f);
    float panelScale = std::max(0.2f, (float)_configuration.panelScale);
    
    simd_float4x4 panelScaleMatrix = matrix_scale(panelScale, panelScale, 1.0f);
    // Keep panel upright, but anchor it to the current head translation so it stays aligned with UI windows.
    simd_float4x4 headTranslation = matrix_identity_float4x4;
    headTranslation.columns[3] = headPose.columns[3];
    _videoQuadMesh->setModelMatrix(simd_mul(headTranslation, simd_mul(panelOffset, panelScaleMatrix)));

    updateVideoFrame();

    // If AI depth is missing or lagging behind the current video frame, generate a fallback depth map from luma.
    double depthLatency = std::max(0.0, _latestDepthTimestamp - _lastDepthTimestamp);
    bool depthStale = (_hasDepthModel && depthLatency > 0.12);
    // Damp disparity when depth is slightly behind video to reduce pan-time shimmer artifacts.
    float disparityStabilityScale = 1.0f - std::clamp((float)((depthLatency - 0.016) / 0.12), 0.0f, 0.45f);
    if (!_hasDepthModel || depthStale) {
        id<MTLCommandBuffer> fallbackCommandBuffer = [_commandQueue commandBuffer];
        updateFallbackDepthFromLuma(fallbackCommandBuffer);
        [fallbackCommandBuffer commit];
    }

    uploadSmoothedDepthToTexture();

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];

    size_t viewCount = cp_drawable_get_view_count(drawable);

    std::array<MTLViewport, 2> viewports {};
    std::array<PoseConstants, 2> poseConstants {};
    for (int i = 0; i < viewCount; ++i) {
        viewports[i] = viewportForViewIndex(drawable, i);

        poseConstants[i] = poseConstantsForViewIndex(drawable, i);
    }

    if (layout == cp_layer_renderer_layout_dedicated) {
        // When rendering with a "dedicated" layout, we draw each eye's view to a separate texture.
        // Since we can't switch render targets within a pass, we render one pass per view.
        for (int i = 0; i < viewCount; ++i) {
            MTLRenderPassDescriptor *renderPassDescriptor = createRenderPassDescriptor(drawable, i);
            id<MTLRenderCommandEncoder> renderCommandEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

            [renderCommandEncoder setCullMode:MTLCullModeBack];

            [renderCommandEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderCommandEncoder setDepthStencilState:_contentDepthStencilState];
            [renderCommandEncoder setRenderPipelineState:_contentRenderPipelineState];
            // Separation converts a user-facing strength value into texture-space eye offset.
            float separation = _disparityStrength / std::max(1.0f, (float)_videoQuadMesh->texture().width);
            separation *= disparityStabilityScale;
            uint32_t eyeOverride = static_cast<uint32_t>(i);
            float colorBoost = std::clamp((float)_configuration.colorBoost, 0.80f, 1.40f);
            float stabilityAmount = std::clamp((float)_configuration.stabilityAmount, 0.0f, 1.0f);
            [renderCommandEncoder setFragmentBytes:&separation length:sizeof(separation) atIndex:0];
            [renderCommandEncoder setFragmentBytes:&eyeOverride length:sizeof(eyeOverride) atIndex:1];
            [renderCommandEncoder setFragmentBytes:&colorBoost length:sizeof(colorBoost) atIndex:2];
            [renderCommandEncoder setFragmentBytes:&stabilityAmount length:sizeof(stabilityAmount) atIndex:3];
            _videoQuadMesh->draw(renderCommandEncoder, &poseConstants[i], 1);

            [renderCommandEncoder endEncoding];
        }
    } else {
        // When rendering in a "shared" or "layered" layout, we use vertex amplification to efficiently
        // run the vertex pipeline for each view. The "shared" layout uses the viewport array to write
        // each view to a distinct region of a single render target, while the "layered" layout writes
        // each view to a separate slice of the render target array texture.
        MTLRenderPassDescriptor *renderPassDescriptor = createRenderPassDescriptor(drawable, 0);
        id<MTLRenderCommandEncoder> renderCommandEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

        [renderCommandEncoder setViewports:viewports.data() count:viewCount];
        [renderCommandEncoder setVertexAmplificationCount:viewCount viewMappings:nil];

        [renderCommandEncoder setCullMode:MTLCullModeBack];

        [renderCommandEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
        [renderCommandEncoder setDepthStencilState:_contentDepthStencilState];
        [renderCommandEncoder setRenderPipelineState:_contentRenderPipelineState];
        // In amplified mode each viewport gets per-eye sampling from one draw call.
        float separation = _disparityStrength / std::max(1.0f, (float)_videoQuadMesh->texture().width);
        separation *= disparityStabilityScale;
        uint32_t eyeOverride = UINT32_MAX;
        float colorBoost = std::clamp((float)_configuration.colorBoost, 0.80f, 1.40f);
        float stabilityAmount = std::clamp((float)_configuration.stabilityAmount, 0.0f, 1.0f);
        [renderCommandEncoder setFragmentBytes:&separation length:sizeof(separation) atIndex:0];
        [renderCommandEncoder setFragmentBytes:&eyeOverride length:sizeof(eyeOverride) atIndex:1];
        [renderCommandEncoder setFragmentBytes:&colorBoost length:sizeof(colorBoost) atIndex:2];
        [renderCommandEncoder setFragmentBytes:&stabilityAmount length:sizeof(stabilityAmount) atIndex:3];
        _videoQuadMesh->draw(renderCommandEncoder, poseConstants.data(), viewCount);

        [renderCommandEncoder endEncoding];
    }

    cp_drawable_encode_present(drawable, commandBuffer);

    [commandBuffer commit];
    _configuration.renderedFrameCount += 1;
    if (_configuration.renderedFrameCount == 1) {
        SetDebugStatus(_configuration, @"Rendering active");
    }
    _lastRenderTime = renderTime;
}

void SpatialRenderer::updateVideoFrame() {
    AVPlayerItemVideoOutput *videoOutput = _configuration.videoOutput;
    if (videoOutput == nil || _videoTextureCache == nullptr) {
        SetDebugStatus(_configuration, @"Waiting for AVPlayer video output");
        return;
    }

    CFTimeInterval hostTime = CACurrentMediaTime();
    CMTime itemTime = [videoOutput itemTimeForHostTime:hostTime];
    _latestDepthTimestamp = CMTimeGetSeconds(itemTime);

    CVPixelBufferRef pixelBuffer = [videoOutput copyPixelBufferForItemTime:itemTime itemTimeForDisplay:nil];
    if (pixelBuffer == nil && CMTIME_IS_VALID(itemTime)) {
        // Retry against a slightly older host time to recover from timing jitter.
        CMTime retryItemTime = [videoOutput itemTimeForHostTime:hostTime - (1.0 / 120.0)];
        pixelBuffer = [videoOutput copyPixelBufferForItemTime:retryItemTime itemTimeForDisplay:nil];
        if (pixelBuffer != nil) {
            itemTime = retryItemTime;
            _latestDepthTimestamp = CMTimeGetSeconds(itemTime);
        }
    }

    if (pixelBuffer == nil) {
        SetDebugStatus(_configuration, @"Renderer active: no video frame available");
        return;
    }

    const size_t width = CVPixelBufferGetWidth(pixelBuffer);
    const size_t height = CVPixelBufferGetHeight(pixelBuffer);
    MTLPixelFormat videoPixelFormat = metalPixelFormatForVideoPixelBuffer(pixelBuffer);

    if (videoPixelFormat == MTLPixelFormatInvalid) {
        SetDebugStatus(_configuration, @"Unsupported pixel format from AVPlayer output");
        CVPixelBufferRelease(pixelBuffer);
        return;
    }

    CVMetalTextureRef metalTextureRef = nullptr;
    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                              _videoTextureCache,
                                              pixelBuffer,
                                              nil,
                                              videoPixelFormat,
                                              width,
                                              height,
                                              0,
                                              &metalTextureRef);
    if (metalTextureRef != nullptr) {
        _videoTexture = CVMetalTextureGetTexture(metalTextureRef);
        if (_videoTexture != nil) {
            ensureDepthMapTexture(width, height);
            _videoQuadMesh->setColorTexture(_videoTexture);
            _configuration.receivedVideoFrameCount += 1;
        }
        CFRelease(metalTextureRef);
    }

    requestDepthInference(pixelBuffer, CMTimeGetSeconds(itemTime));
    CVPixelBufferRelease(pixelBuffer);
}

void SpatialRenderer::updateFallbackDepthFromLuma(id<MTLCommandBuffer> commandBuffer) {
    if (commandBuffer == nil ||
        _fallbackDepthComputePipelineState == nil ||
        _videoTexture == nil ||
        _depthMapTexture == nil) {
        return;
    }

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) {
        return;
    }

    [computeEncoder setComputePipelineState:_fallbackDepthComputePipelineState];
    [computeEncoder setTexture:_videoTexture atIndex:0];
    [computeEncoder setTexture:_depthMapTexture atIndex:1];

    MTLSize gridSize = MTLSizeMake(_depthMapTexture.width, _depthMapTexture.height, 1);
    NSUInteger tw = _fallbackDepthComputePipelineState.threadExecutionWidth;
    NSUInteger th = _fallbackDepthComputePipelineState.maxTotalThreadsPerThreadgroup / tw;
    if (th == 0) {
        th = 1;
    }
    MTLSize threadsPerThreadgroup = MTLSizeMake(tw, th, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
}

void SpatialRenderer::requestDepthInference(CVPixelBufferRef pixelBuffer, CFTimeInterval frameTime) {
    if (_depthRequest == nil || _depthInferenceInFlight) {
        return;
    }

    // Keep depth inference cadence tighter to reduce pan-induced stereo shimmer.
    const int skipInterval = 2;
    _frameSkipCounter++;
    if (_frameSkipCounter % skipInterval != 0) {
        return;  // Reuse previous depth, continue with temporal smoothing
    }

    // Cap inference cadence to reduce pressure that can destabilize immersive rendering.
    CFTimeInterval timeSinceLastInference = frameTime - _lastInferenceTime;
    if (timeSinceLastInference < 0.066) {
        _frameSkipCounter--;  // Don't consume skip counter if we're debouncing
        return;
    }

    _depthInferenceInFlight = true;
    _lastInferenceTime = frameTime;
    CVPixelBufferRetain(pixelBuffer);
    // Run Vision inference off the render thread to avoid stalling frame submission.
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSError *error = nil;
            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
            [handler performRequests:@[_depthRequest] error:&error];
            if (error != nil) {
                _depthInferenceInFlight = false;
                SetDebugStatus(_configuration, [NSString stringWithFormat:@"Depth inference error: %@", error.localizedDescription]);
            } else {
                _lastDepthTimestamp = frameTime;
                _configuration.depthFrameCount += 1;
            }
            CVPixelBufferRelease(pixelBuffer);
        }
    });
}

void SpatialRenderer::uploadSmoothedDepthToTexture() {
    if (_depthMapTexture == nil) {
        return;
    }

    uint32_t srcW = 0;
    uint32_t srcH = 0;
    {
        std::lock_guard<std::mutex> lock(_depthMutex);
        if (_smoothedDepth.empty() || _depthWidth == 0 || _depthHeight == 0) {
            return;
        }
        _depthCopyBuffer = _smoothedDepth;
        srcW = _depthWidth;
        srcH = _depthHeight;
    }

    const uint32_t dstW = static_cast<uint32_t>(_depthMapTexture.width);
    const uint32_t dstH = static_cast<uint32_t>(_depthMapTexture.height);
    // Resize model output to video resolution so shader sampling stays aligned with color pixels.
    const size_t dstSize = (size_t)dstW * dstH;
    if (_depthResizeBuffer.size() != dstSize) {
        _depthResizeBuffer.resize(dstSize);
    }
    std::fill(_depthResizeBuffer.begin(), _depthResizeBuffer.end(), 0.5f);

    for (uint32_t y = 0; y < dstH; ++y) {
        float fy = (srcH <= 1 || dstH <= 1) ? 0.0f : ((float)y * (float)(srcH - 1) / (float)(dstH - 1));
        uint32_t y0 = (uint32_t)std::floor(fy);
        uint32_t y1 = std::min(y0 + 1, srcH - 1);
        float ty = fy - (float)y0;

        for (uint32_t x = 0; x < dstW; ++x) {
            float fx = (srcW <= 1 || dstW <= 1) ? 0.0f : ((float)x * (float)(srcW - 1) / (float)(dstW - 1));
            uint32_t x0 = (uint32_t)std::floor(fx);
            uint32_t x1 = std::min(x0 + 1, srcW - 1);
            float tx = fx - (float)x0;

            float d00 = _depthCopyBuffer[(size_t)y0 * srcW + x0];
            float d10 = _depthCopyBuffer[(size_t)y0 * srcW + x1];
            float d01 = _depthCopyBuffer[(size_t)y1 * srcW + x0];
            float d11 = _depthCopyBuffer[(size_t)y1 * srcW + x1];

            float top = d00 + (d10 - d00) * tx;
            float bottom = d01 + (d11 - d01) * tx;
            _depthResizeBuffer[(size_t)y * dstW + x] = top + (bottom - top) * ty;
        }
    }

    MTLRegion region = MTLRegionMake2D(0, 0, dstW, dstH);
    [_depthMapTexture replaceRegion:region mipmapLevel:0 withBytes:_depthResizeBuffer.data() bytesPerRow:dstW * sizeof(float)];
}

MTLRenderPassDescriptor* SpatialRenderer::createRenderPassDescriptor(cp_drawable_t drawable, size_t index) {
    cp_layer_renderer_configuration_t layerConfiguration = cp_layer_renderer_get_configuration(_layerRenderer);
    cp_layer_renderer_layout layout = cp_layer_renderer_configuration_get_layout(layerConfiguration);

    MTLRenderPassDescriptor *passDescriptor = [[MTLRenderPassDescriptor alloc] init];

    passDescriptor.colorAttachments[0].texture = cp_drawable_get_color_texture(drawable, index);
    passDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
    passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;

    passDescriptor.depthAttachment.texture = cp_drawable_get_depth_texture(drawable, index);
    passDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
    passDescriptor.depthAttachment.clearDepth = 0.0;
    passDescriptor.depthAttachment.storeAction = MTLStoreActionStore;

    switch (layout) {
        case cp_layer_renderer_layout_layered:
            passDescriptor.renderTargetArrayLength = cp_drawable_get_view_count(drawable);
            break;
        case cp_layer_renderer_layout_shared:
            // Even though we don't use an array texture as the render target in "shared" layout, we're
            // obligated to set the render target array length because the index is set by the vertex shader.
            passDescriptor.renderTargetArrayLength = 1;
            break;
        case cp_layer_renderer_layout_dedicated:
            break;
    }

    if (cp_drawable_get_rasterization_rate_map_count(drawable) > 0) {
        passDescriptor.rasterizationRateMap = cp_drawable_get_rasterization_rate_map(drawable, index);
    }

    return passDescriptor;
}

MTLViewport SpatialRenderer::viewportForViewIndex(cp_drawable_t drawable, size_t index) {
    cp_view_t view = cp_drawable_get_view(drawable, index);
    cp_view_texture_map_t texture_map = cp_view_get_view_texture_map(view);
    return cp_view_texture_map_get_viewport(texture_map);
}

PoseConstants SpatialRenderer::poseConstantsForViewIndex(cp_drawable_t drawable, size_t index) {
    PoseConstants outPose;

    ar_device_anchor_t anchor = cp_drawable_get_device_anchor(drawable);

    simd_float4x4 poseTransform = ar_anchor_get_origin_from_anchor_transform(anchor);

    cp_view_t view = cp_drawable_get_view(drawable, index);

    if (@available(visionOS 2.0, *)) {
        outPose.projectionMatrix = cp_drawable_compute_projection(drawable,
                                                                  cp_axis_direction_convention_right_up_back,
                                                                  index);
    } else {
        simd_float4 tangents = cp_view_get_tangents(view);
        simd_float2 depth_range = cp_drawable_get_depth_range(drawable);
        SPProjectiveTransform3D projectiveTransform = SPProjectiveTransform3DMakeFromTangents(tangents[0],
                                                                                              tangents[1],
                                                                                              tangents[2],
                                                                                              tangents[3],
                                                                                              depth_range[1],
                                                                                              depth_range[0],
                                                                                              true);
        outPose.projectionMatrix = matrix_float4x4_from_double4x4(projectiveTransform.matrix);
    }

    simd_float4x4 cameraMatrix = simd_mul(poseTransform, cp_view_get_transform(view));
    outPose.viewMatrix = simd_inverse(cameraMatrix);
    return outPose;
}
