import AVFoundation
import CoreGraphics
import Foundation

final class Video3DConfiguration: NSObject {
    @objc dynamic var panelOffsetX: CGFloat = 0
    @objc dynamic var panelOffsetY: CGFloat = 0
    @objc dynamic var panelOffsetZ: CGFloat = -1
    @objc dynamic var panelScale: CGFloat = 1
    @objc dynamic var disparityStrength: CGFloat = 6.5
    @objc dynamic var stabilityAmount: CGFloat = 0.85
    @objc dynamic var colorBoost: CGFloat = 1
    @objc dynamic var playerVideoOutput: AVPlayerVideoOutput?
    @objc dynamic var rendererDebugStatus: String = "Idle"
    @objc dynamic var renderedFrameCount: UInt = 0
    @objc dynamic var receivedVideoFrameCount: UInt = 0
    @objc dynamic var depthFrameCount: UInt = 0
}