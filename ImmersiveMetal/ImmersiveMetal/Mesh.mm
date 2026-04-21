
#include "Mesh.h"

#include <array>

static id<MTLTexture> _Nullable CreateTextureFromImage(NSString *imageName, id<MTLDevice> device, NSError **error) {
    MTKTextureLoader *textureLoader = [[MTKTextureLoader alloc] initWithDevice:device];
    NSURL *imageURL = [[NSBundle mainBundle] URLForResource:imageName withExtension:nil];
    CGImageSourceRef imageSource = CGImageSourceCreateWithURL((__bridge CFURLRef)imageURL, NULL);
    if (imageSource) {
        CGImageRef image = CGImageSourceCreateImageAtIndex(imageSource, 0, NULL);
        if (image) {
            id<MTLTexture> texture = [textureLoader newTextureWithCGImage:image options:nil error:error];
            CGImageRelease(image);
            return texture;
        }
        CFRelease(imageSource);
    }
    return nil;
}

MTLVertexDescriptor *Mesh::vertexDescriptor() const {
    MTLVertexDescriptor *vertexDescriptor = [MTLVertexDescriptor new];
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[1].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[1].bufferIndex = 0;
    vertexDescriptor.attributes[1].offset = sizeof(float) * 3;
    vertexDescriptor.attributes[2].format = MTLVertexFormatFloat2;
    vertexDescriptor.attributes[2].bufferIndex = 0;
    vertexDescriptor.attributes[2].offset = sizeof(float) * 6;
    vertexDescriptor.layouts[0].stride = sizeof(float) * 8;
    return vertexDescriptor;
}

struct StereoQuadVertex {
    simd_float4 position;
    simd_float2 texCoord;
};

StereoQuadMesh::StereoQuadMesh(float width, float height, NSString *imageName, id<MTLDevice> device) {
    NSError *error = nil;
    _colorTexture = CreateTextureFromImage(imageName, device, &error);

    const float halfWidth = width * 0.5f;
    const float halfHeight = height * 0.5f;
    std::array<StereoQuadVertex, 4> vertices = {{
        { simd_make_float4(-halfWidth,  halfHeight, 0.0f, 1.0f), simd_make_float2(0.0f, 0.0f) },
        { simd_make_float4(-halfWidth, -halfHeight, 0.0f, 1.0f), simd_make_float2(0.0f, 1.0f) },
        { simd_make_float4( halfWidth, -halfHeight, 0.0f, 1.0f), simd_make_float2(1.0f, 1.0f) },
        { simd_make_float4( halfWidth,  halfHeight, 0.0f, 1.0f), simd_make_float2(1.0f, 0.0f) }
    }};
    std::array<uint16_t, 6> indices = {{0, 1, 2, 0, 2, 3}};

    _vertexBuffer = [device newBufferWithBytes:vertices.data()
                                        length:vertices.size() * sizeof(StereoQuadVertex)
                                       options:MTLResourceStorageModeShared];
    _indexBuffer = [device newBufferWithBytes:indices.data()
                                       length:indices.size() * sizeof(uint16_t)
                                      options:MTLResourceStorageModeShared];
    _indexCount = indices.size();
}

MTLVertexDescriptor *StereoQuadMesh::vertexDescriptor() const {
    MTLVertexDescriptor *vertexDescriptor = [MTLVertexDescriptor new];
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat4;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[1].format = MTLVertexFormatFloat2;
    vertexDescriptor.attributes[1].bufferIndex = 0;
    vertexDescriptor.attributes[1].offset = sizeof(float) * 4;
    vertexDescriptor.layouts[0].stride = sizeof(StereoQuadVertex);
    return vertexDescriptor;
}

void StereoQuadMesh::draw(id<MTLRenderCommandEncoder> renderCommandEncoder, PoseConstants *poseConstants, size_t poseCount) {
    InstanceConstants instanceConstants;
    instanceConstants.modelMatrix = modelMatrix();

    [renderCommandEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
    [renderCommandEncoder setVertexBytes:poseConstants length:sizeof(PoseConstants) * poseCount atIndex:1];
    [renderCommandEncoder setVertexBytes:&instanceConstants length:sizeof(instanceConstants) atIndex:2];
    [renderCommandEncoder setFragmentTexture:_colorTexture atIndex:0];
    [renderCommandEncoder setFragmentTexture:(_depthTexture != nil ? _depthTexture : _colorTexture) atIndex:1];
    [renderCommandEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                     indexCount:_indexCount
                                      indexType:MTLIndexTypeUInt16
                                    indexBuffer:_indexBuffer
                              indexBufferOffset:0];
}
