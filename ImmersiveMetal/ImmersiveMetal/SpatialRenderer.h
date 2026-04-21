#pragma once

#include "Mesh.h"
#include "ShaderTypes.h"
#include "VideoRendererBridge.h"

#include <memory>
#include <mutex>
#include <vector>

#import <CompositorServices/CompositorServices.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>

@class AVPlayerItemVideoOutput;
@class VNCoreMLRequest;

class SpatialRenderer {
public:
    SpatialRenderer(cp_layer_renderer_t layerRenderer, Video3DConfiguration *configuration);

    void drawAndPresent(cp_frame_t frame, cp_drawable_t drawable);

private:
    void makeResources();
    void makeRenderPipelines();
    void ensureDepthMapTexture(size_t videoWidth, size_t videoHeight);
    void updateVideoFrame();
    void requestDepthInference(CVPixelBufferRef pixelBuffer, CFTimeInterval frameTime);
    void updateFallbackDepthFromLuma(id<MTLCommandBuffer> commandBuffer);
    void uploadSmoothedDepthToTexture();
    MTLRenderPassDescriptor* createRenderPassDescriptor(cp_drawable_t drawable, size_t index);
    MTLViewport viewportForViewIndex(cp_drawable_t drawable, size_t index);
    PoseConstants poseConstantsForViewIndex(cp_drawable_t drawable, size_t index);

    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _contentRenderPipelineState;
    id<MTLComputePipelineState> _fallbackDepthComputePipelineState;
    id<MTLDepthStencilState> _contentDepthStencilState;
    id<MTLTexture> _depthMapTexture;
    id<MTLTexture> _videoTexture;
    CVMetalTextureCacheRef _videoTextureCache;
    cp_layer_renderer_t _layerRenderer;
    std::unique_ptr<StereoQuadMesh> _videoQuadMesh;
    VNCoreMLRequest *_depthRequest;
    Video3DConfiguration *_configuration;
    CFTimeInterval _lastRenderTime;
    CFTimeInterval _lastDepthTimestamp;
    CFTimeInterval _latestDepthTimestamp;
    CFTimeInterval _lastInferenceTime;  // Track last ML inference for debouncing
    float _disparityStrength;
    float _temporalSmoothing;
    uint32_t _depthWidth;
    uint32_t _depthHeight;
    uint32_t _frameSkipCounter;  // Skip inference on static scenes
    bool _hasDepthModel;
    bool _depthInferenceInFlight;
    std::vector<float> _smoothedDepth;
    std::vector<float> _depthCopyBuffer;
    std::vector<float> _depthResizeBuffer;
    std::mutex _depthMutex;
};
