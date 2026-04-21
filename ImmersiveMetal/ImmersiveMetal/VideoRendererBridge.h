#pragma once

#import <Foundation/Foundation.h>
#import <CompositorServices/CompositorServices.h>

@class AVPlayerItemVideoOutput;

NS_ASSUME_NONNULL_BEGIN

// A type for communicating immersion style changes from SwiftUI views to the low-level rendering layer
@interface Video3DConfiguration : NSObject
@property (assign) CGFloat panelOffsetX;
@property (assign) CGFloat panelOffsetY;
@property (assign) CGFloat panelOffsetZ;
@property (assign) CGFloat panelScale;
@property (assign) CGFloat disparityStrength;
@property (assign) CGFloat stabilityAmount;
@property (assign) CGFloat colorBoost;
@property (strong, nullable) AVPlayerItemVideoOutput *videoOutput;
@property (copy) NSString *rendererDebugStatus;
@property (assign) NSUInteger renderedFrameCount;
@property (assign) NSUInteger receivedVideoFrameCount;
@property (assign) NSUInteger depthFrameCount;
@end

#if __cplusplus
extern "C" {
#endif

void StartVideo3DRenderer(cp_layer_renderer_t layerRenderer, Video3DConfiguration *configuration);

#if __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
