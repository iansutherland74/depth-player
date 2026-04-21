#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "ShaderTypes.h"

class Mesh {
public:
    virtual MTLVertexDescriptor *vertexDescriptor() const;
    virtual void draw(id<MTLRenderCommandEncoder> renderCommandEncoder, PoseConstants *poseConstants, size_t poseCount) = 0;

    simd_float4x4 modelMatrix() const { return _modelMatrix; }

    void setModelMatrix(simd_float4x4 m) { _modelMatrix = m; };

private:
    simd_float4x4 _modelMatrix = matrix_identity_float4x4;
};

class StereoQuadMesh: public Mesh {
public:
    StereoQuadMesh(float width, float height, NSString *imageName, id<MTLDevice> device);

    MTLVertexDescriptor *vertexDescriptor() const override;
    void draw(id<MTLRenderCommandEncoder> renderCommandEncoder, PoseConstants *poseConstants, size_t poseCount) override;

    id<MTLTexture> texture() const { return _colorTexture; }
    void setColorTexture(id<MTLTexture> texture) { if (texture != nil) { _colorTexture = texture; } }
    void setDepthTexture(id<MTLTexture> texture) { _depthTexture = texture; }

private:
    id<MTLBuffer> _vertexBuffer;
    id<MTLBuffer> _indexBuffer;
    NSUInteger _indexCount;
    id<MTLTexture> _colorTexture;
    id<MTLTexture> _depthTexture;
};
