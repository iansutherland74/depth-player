import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import QuartzCore
import CoreMedia
import OSLog
import Accelerate
import Darwin
#if os(visionOS)
import RealityKit
#endif

struct StereoProcessedFrame {
    let stereoPixelBuffer: CVPixelBuffer
}

@MainActor
final class StereoPresentationCoordinator: ObservableObject {
    @Published private(set) var renderer: AVSampleBufferVideoRenderer?
    @Published private(set) var immersiveStatus: String = "Preparing immersive stereo playback..."
    @Published private(set) var stopRequestID: Int = 0

    func attach(renderer: AVSampleBufferVideoRenderer) {
        self.renderer = renderer
        immersiveStatus = "Stereo renderer attached"
    }

    func flushRenderer() {
        renderer?.flush()
    }

    func detach() {
        renderer = nil
        immersiveStatus = "Stereo renderer detached"
    }

    func setStatus(_ status: String) {
        immersiveStatus = status
    }

    func requestStopPlayback() {
        stopRequestID += 1
    }
}

final class StereoSampleBufferBridge: @unchecked Sendable {
    let renderer = AVSampleBufferVideoRenderer()

    private var cachedFormatDescription: CMFormatDescription?
    private var cachedDimensions: CMVideoDimensions?
    // Monotonic presentation clock for VideoPlayerComponent's host-time render synchronizer.
    private var nextScheduledTime: CMTime = .invalid
    private var lastEnqueueHostTime: CMTime = .invalid
    private static let lookahead = CMTime(seconds: 0.028, preferredTimescale: 1_000_000)

    func flushForReattach() {
        renderer.flush()
        cachedFormatDescription = nil
        cachedDimensions = nil
        nextScheduledTime = .invalid
        lastEnqueueHostTime = .invalid
    }

    func enqueue(pixelBuffer: CVPixelBuffer, at presentationTime: CMTime, duration: CMTime = .invalid) throws {
        // VideoPlayerComponent's render synchronizer runs on the host (mach absolute time) clock.
        // Media timestamps would be treated as far-future and never shown.
        //
        // We measure the actual wall-clock interval between successive enqueue calls and use that
        // as the frame duration. When depth inference runs at 12 fps the measured interval is ~83 ms
        // and frames tile perfectly with no gaps. A fixed 1/30 s duration would leave 50 ms holes
        // between every frame causing visible stutter.
        let hostNow = CMClockGetTime(CMClockGetHostTimeClock())
        let earliest = CMTimeAdd(hostNow, Self.lookahead)

        // Use a fixed frame duration that matches the active processing cadence.
        // The caller supplies duration when load shedding adjusts cadence.
        lastEnqueueHostTime = hostNow
        let frameDuration = (duration.isValid && duration != .zero) ? duration : CMTime(value: 1, timescale: 12)

        // Cap drift: if nextScheduledTime is more than 2 frame durations ahead of now+lookahead,
        // reset it. This prevents a burst of late-delivered frames from queuing up far in the future
        // and playing back too fast when the renderer catches up (visible as stutter/judder).
        let threeFrames = CMTimeAdd(CMTimeAdd(frameDuration, frameDuration), frameDuration)
        let maxLead = CMTimeAdd(earliest, threeFrames)
        let scheduleTime: CMTime
        if nextScheduledTime.isValid && CMTimeCompare(nextScheduledTime, earliest) > 0 && CMTimeCompare(nextScheduledTime, maxLead) <= 0 {
            scheduleTime = nextScheduledTime
        } else {
            scheduleTime = earliest
        }
        nextScheduledTime = CMTimeAdd(scheduleTime, frameDuration)

        let formatDescription = try getFormatDescription(for: pixelBuffer)
        let timing = CMSampleTimingInfo(duration: frameDuration, presentationTimeStamp: scheduleTime, decodeTimeStamp: .invalid)
        let sampleBuffer = try CMSampleBuffer(
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: timing
        )
        renderer.enqueue(sampleBuffer)
    }

    func flush() {
        renderer.flush()
        nextScheduledTime = .invalid
        lastEnqueueHostTime = .invalid
    }

    private func getFormatDescription(for pixelBuffer: CVPixelBuffer) throws -> CMFormatDescription {
        let dimensions = CMVideoDimensions(width: Int32(CVPixelBufferGetWidth(pixelBuffer)), height: Int32(CVPixelBufferGetHeight(pixelBuffer)))

        if let cachedFormatDescription,
           let cachedDimensions,
           cachedDimensions.width == dimensions.width,
           cachedDimensions.height == dimensions.height {
            return cachedFormatDescription
        }

        let baseFormat = try CMVideoFormatDescription(imageBuffer: pixelBuffer)
        var extensions = baseFormat.extensions

        if #available(visionOS 26.0, *) {
            extensions[.viewPackingKind] = .viewPackingKind(.sideBySide)
            extensions[.projectionKind] = .projectionKind(.rectilinear)
            extensions[.horizontalFieldOfView] = .number(UInt32(65_000))
        }

        let formatDescription = try CMVideoFormatDescription(
            videoCodecType: baseFormat.mediaSubType,
            width: Int(baseFormat.dimensions.width),
            height: Int(baseFormat.dimensions.height),
            extensions: extensions
        )

        cachedFormatDescription = formatDescription
        cachedDimensions = dimensions
        return formatDescription
    }
}

final class StereoFrameProcessor: @unchecked Sendable {
    private let estimator: DepthAnythingEstimator
    // Explicit sRGB output prevents Vision Pro's wide-gamut display from washing out colours.
    private let ciContext = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any,
    ])
    private let disparityControlQueue = DispatchQueue(label: "com.vision.depthplayer.disparity-control")
    private var runtimeMaxDisparity: CGFloat
    private let temporalSmoothing: Float
    private var depthEMA: MLMultiArray?
    private var stereoPixelBufferPool: CVPixelBufferPool?
    private var stereoPoolWidth: Int = 0
    private var stereoPoolHeight: Int = 0
    private var depthPixelBufferPool: CVPixelBufferPool?
    private var depthPoolWidth: Int = 0
    private var depthPoolHeight: Int = 0
    private var cachedDepthForReuse: MLMultiArray?
    private var lastDepthInferenceHostTime: CFTimeInterval = 0
    private let depthInferenceInterval: CFTimeInterval = 1.0 / 6.0
    private var consecutiveDepthFailures = 0
    private var depthBypassUntilHostTime: CFTimeInterval = 0
    private let maxConsecutiveDepthFailures = 3
    private let depthBypassDuration: CFTimeInterval = 2.0

    init(model: MLModel, maxDisparity: CGFloat, temporalSmoothing: Float) throws {
        self.estimator = try DepthAnythingEstimator(model: model)
        self.runtimeMaxDisparity = max(0, maxDisparity)
        self.temporalSmoothing = min(max(temporalSmoothing, 0), 0.99)
    }

    func updateMaxDisparity(_ value: CGFloat) {
        disparityControlQueue.sync {
            runtimeMaxDisparity = max(0, value)
        }
    }

    func process(pixelBuffer: CVPixelBuffer) throws -> StereoProcessedFrame {
        let activeDisparity = disparityControlQueue.sync { runtimeMaxDisparity }
        if activeDisparity <= 0.001 {
            return try makeStereoSBSWithoutDepth(pixelBuffer: pixelBuffer)
        }
        // Stable path: uniform stereo parallax preserves cadence and avoids audio-only regressions.
        return try makeStereoSBSParallax(pixelBuffer: pixelBuffer, disparity: activeDisparity)
    }

    private func makeStereoSBSWithoutDepth(pixelBuffer: CVPixelBuffer) throws -> StereoProcessedFrame {
        let source = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: Self.bt709])
        let width = source.extent.width
        let height = source.extent.height

        let canvasRect = CGRect(x: 0, y: 0, width: width * 2, height: height)
        let background = CIImage(color: .black).cropped(to: canvasRect)

        let rightPlaced = source.transformed(by: CGAffineTransform(translationX: width, y: 0))
        let composed = rightPlaced
            .composited(over: source)
            .composited(over: background)

        let stereoWidth = Int(width * 2)
        let stereoHeight = Int(height)
        let stereoPixelBuffer = try makeReusableStereoPixelBuffer(width: stereoWidth, height: stereoHeight)

        let colorAttachments: [CFString: Any] = [
            kCVImageBufferColorPrimariesKey: kCVImageBufferColorPrimaries_ITU_R_709_2,
            kCVImageBufferTransferFunctionKey: kCVImageBufferTransferFunction_ITU_R_709_2,
            kCVImageBufferYCbCrMatrixKey: kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        ]
        CVBufferSetAttachments(stereoPixelBuffer, colorAttachments as CFDictionary, .shouldPropagate)

        ciContext.render(
            composed,
            to: stereoPixelBuffer,
            bounds: CGRect(x: 0, y: 0, width: CGFloat(stereoWidth), height: CGFloat(stereoHeight)),
            colorSpace: Self.bt709
        )
        return StereoProcessedFrame(stereoPixelBuffer: stereoPixelBuffer)
    }

    private func makeStereoSBSParallax(pixelBuffer: CVPixelBuffer, disparity: CGFloat) throws -> StereoProcessedFrame {
        let source = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: Self.bt709])
        let width = source.extent.width
        let height = source.extent.height

        // Stable pseudo-depth: blend near/far shifted layers using a luminance-derived mask.
        // This adds depth cues without enabling ML inference in the playback loop.
        let shiftPixels = min(max(disparity * 150.0, 0), 150.0)
        let nearShift = shiftPixels * 0.5
        let farShift = shiftPixels * 0.2

        let pseudoDepthMask = source
            .applyingFilter("CIColorControls", parameters: [
                kCIInputSaturationKey: 0.0,
                kCIInputContrastKey: 1.25,
                kCIInputBrightnessKey: 0.0,
            ])
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 3.0])
            .cropped(to: source.extent)

        let leftNear = source.transformed(by: CGAffineTransform(translationX: nearShift, y: 0))
        let leftFar = source.transformed(by: CGAffineTransform(translationX: -farShift, y: 0))
        let rightNear = source.transformed(by: CGAffineTransform(translationX: -nearShift, y: 0))
        let rightFar = source.transformed(by: CGAffineTransform(translationX: farShift, y: 0))

        let left = leftNear
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: leftFar,
                kCIInputMaskImageKey: pseudoDepthMask,
            ])
            .cropped(to: source.extent)

        let right = rightNear
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: rightFar,
                kCIInputMaskImageKey: pseudoDepthMask,
            ])
            .cropped(to: source.extent)

        let canvasRect = CGRect(x: 0, y: 0, width: width * 2, height: height)
        let background = CIImage(color: .black).cropped(to: canvasRect)
        let rightPlaced = right.transformed(by: CGAffineTransform(translationX: width, y: 0))
        let composed = rightPlaced
            .composited(over: left)
            .composited(over: background)

        let stereoWidth = Int(width * 2)
        let stereoHeight = Int(height)
        let stereoPixelBuffer = try makeReusableStereoPixelBuffer(width: stereoWidth, height: stereoHeight)

        let colorAttachments: [CFString: Any] = [
            kCVImageBufferColorPrimariesKey: kCVImageBufferColorPrimaries_ITU_R_709_2,
            kCVImageBufferTransferFunctionKey: kCVImageBufferTransferFunction_ITU_R_709_2,
            kCVImageBufferYCbCrMatrixKey: kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        ]
        CVBufferSetAttachments(stereoPixelBuffer, colorAttachments as CFDictionary, .shouldPropagate)

        ciContext.render(
            composed,
            to: stereoPixelBuffer,
            bounds: CGRect(x: 0, y: 0, width: CGFloat(stereoWidth), height: CGFloat(stereoHeight)),
            colorSpace: Self.bt709
        )
        return StereoProcessedFrame(stereoPixelBuffer: stereoPixelBuffer)
    }

    private func smoothDepth(_ depth: MLMultiArray) throws -> MLMultiArray {
        if temporalSmoothing <= 0.001 {
            depthEMA = depth
            return depth
        }

        guard depth.dataType == .float32 else {
            depthEMA = depth
            return depth
        }

        let ema: MLMultiArray
        if let prev = depthEMA, prev.count == depth.count, prev.dataType == .float32 {
            ema = prev
        } else {
            ema = try MLMultiArray(shape: depth.shape, dataType: .float32)
            let src = depth.dataPointer.bindMemory(to: Float.self, capacity: depth.count)
            let dst = ema.dataPointer.bindMemory(to: Float.self, capacity: depth.count)
            dst.assign(from: src, count: depth.count)
            depthEMA = ema
            return ema
        }

        let alpha = temporalSmoothing
        var alphaScalar = alpha
        var oneMinusAlpha = 1.0 - alpha
        let depthPtr = depth.dataPointer.bindMemory(to: Float.self, capacity: depth.count)
        let emaPtr = ema.dataPointer.bindMemory(to: Float.self, capacity: ema.count)
        let n = vDSP_Length(depth.count)

        // EMA in-place: ema = alpha * ema + (1 - alpha) * depth.
        vDSP_vsmul(emaPtr, 1, &alphaScalar, emaPtr, 1, n)
        vDSP_vsma(depthPtr, 1, &oneMinusAlpha, emaPtr, 1, emaPtr, 1, n)

        depthEMA = ema
        return ema
    }

    private static let bt709 = CGColorSpace(name: CGColorSpace.itur_709)!

    private func makeStereoSBS(pixelBuffer: CVPixelBuffer, depth: MLMultiArray, disparity: CGFloat) throws -> StereoProcessedFrame {
        // Tag the source image with BT.709 so Core Image's colour pipeline stays in the
        // correct space rather than treating gamma-encoded video as linear light.
        let source = CIImage(cvPixelBuffer: pixelBuffer, options: [.colorSpace: Self.bt709])
        let width = source.extent.width
        let height = source.extent.height

        let depthImage = try makeNormalizedDepthImage(from: depth)
            .transformed(by: CGAffineTransform(scaleX: width / depthWidth(depth), y: height / depthHeight(depth)))
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 2.8])
            .cropped(to: source.extent)

        let maxShiftPixels = min(max(disparity * 150.0, 0), 150.0)
        let nearShift = maxShiftPixels * 0.5
        let farShift = maxShiftPixels * 0.2

        let leftNear = source.transformed(by: CGAffineTransform(translationX: nearShift, y: 0))
        let leftFar = source.transformed(by: CGAffineTransform(translationX: -farShift, y: 0))
        let rightNear = source.transformed(by: CGAffineTransform(translationX: -nearShift, y: 0))
        let rightFar = source.transformed(by: CGAffineTransform(translationX: farShift, y: 0))

        let left = leftNear
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: leftFar,
                kCIInputMaskImageKey: depthImage,
            ])
            .cropped(to: source.extent)

        let right = rightNear
            .applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: rightFar,
                kCIInputMaskImageKey: depthImage,
            ])
            .cropped(to: source.extent)

        let canvasRect = CGRect(x: 0, y: 0, width: width * 2, height: height)
        let background = CIImage(color: .black).cropped(to: canvasRect)

        let rightPlaced = right.transformed(by: CGAffineTransform(translationX: width, y: 0))
        let composed = rightPlaced
            .composited(over: left)
            .composited(over: background)

        let stereoWidth = Int(width * 2)
        let stereoHeight = Int(height)
        let stereoPixelBuffer = try makeReusableStereoPixelBuffer(width: stereoWidth, height: stereoHeight)

        // Explicitly attach BT.709 colour metadata to each buffer. Pool creation attributes
        // do not automatically propagate to CVBuffer attachments that VideoPlayerComponent reads.
        let colorAttachments: [CFString: Any] = [
            kCVImageBufferColorPrimariesKey: kCVImageBufferColorPrimaries_ITU_R_709_2,
            kCVImageBufferTransferFunctionKey: kCVImageBufferTransferFunction_ITU_R_709_2,
            kCVImageBufferYCbCrMatrixKey: kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        ]
        CVBufferSetAttachments(stereoPixelBuffer, colorAttachments as CFDictionary, .shouldPropagate)

        ciContext.render(composed, to: stereoPixelBuffer, bounds: CGRect(x: 0, y: 0, width: CGFloat(stereoWidth), height: CGFloat(stereoHeight)), colorSpace: Self.bt709)
        return StereoProcessedFrame(stereoPixelBuffer: stereoPixelBuffer)
    }

    private func makeNormalizedDepthImage(from depth: MLMultiArray) throws -> CIImage {
        guard depth.dataType == .float32 || depth.dataType == .float16 || depth.dataType == .double else {
            throw DepthEstimatorError.unexpectedOutputType
        }

        let shape = depth.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            throw DepthEstimatorError.unexpectedOutputShape
        }

        let depthH = shape[shape.count - 2]
        let depthW = shape[shape.count - 1]
        let count = depthH * depthW
        guard count > 0 else {
            throw DepthEstimatorError.unexpectedOutputShape
        }

        var minValue = Float.greatestFiniteMagnitude
        var maxValue = -Float.greatestFiniteMagnitude
        for i in 0..<count {
            let v = depth[i].floatValue
            if v < minValue { minValue = v }
            if v > maxValue { maxValue = v }
        }

        let denom = max(maxValue - minValue, 1e-6)
        let pixelBuffer = try makeReusableDepthPixelBuffer(width: depthW, height: depthH)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DepthEstimatorError.imageCreationFailed
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dst = baseAddress.assumingMemoryBound(to: UInt8.self)

        for y in 0..<depthH {
            let row = dst.advanced(by: y * bytesPerRow)
            let rowOffset = y * depthW
            for x in 0..<depthW {
                let i = rowOffset + x
                let normalized = (depth[i].floatValue - minValue) / denom
                let scaled = Int(normalized * 255.0)
                row[x] = UInt8(max(0, min(255, scaled)))
            }
        }

        return CIImage(cvPixelBuffer: pixelBuffer)
    }

    private func makeReusableStereoPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if stereoPixelBufferPool == nil || stereoPoolWidth != width || stereoPoolHeight != height {
            let poolAttributes: [CFString: Any] = [
                kCVPixelBufferPoolMinimumBufferCountKey: 3,
            ]
            let pixelBufferAttributes: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferMetalCompatibilityKey: true,
                // Colour metadata so VideoPlayerComponent applies the right transform.
                kCVImageBufferColorPrimariesKey: kCVImageBufferColorPrimaries_ITU_R_709_2,
                kCVImageBufferTransferFunctionKey: kCVImageBufferTransferFunction_ITU_R_709_2,
            ]

            var pool: CVPixelBufferPool?
            let poolStatus = CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                poolAttributes as CFDictionary,
                pixelBufferAttributes as CFDictionary,
                &pool
            )

            guard poolStatus == kCVReturnSuccess, let pool else {
                throw NSError(domain: "StereoFrameProcessor", code: -2)
            }

            stereoPixelBufferPool = pool
            stereoPoolWidth = width
            stereoPoolHeight = height
        }

        guard let pool = stereoPixelBufferPool else {
            throw NSError(domain: "StereoFrameProcessor", code: -3)
        }

        var stereoPixelBuffer: CVPixelBuffer?
        let auxAttributes: [CFString: Any] = [
            kCVPixelBufferPoolAllocationThresholdKey: 6,
        ]
        let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
            kCFAllocatorDefault,
            pool,
            auxAttributes as CFDictionary,
            &stereoPixelBuffer
        )
        if status == kCVReturnWouldExceedAllocationThreshold {
            throw NSError(domain: "StereoFrameProcessor", code: -5)
        }
        guard status == kCVReturnSuccess, let stereoPixelBuffer else {
            throw NSError(domain: "StereoFrameProcessor", code: -4)
        }
        return stereoPixelBuffer
    }

    private func makeReusableDepthPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if depthPixelBufferPool == nil || depthPoolWidth != width || depthPoolHeight != height {
            let poolAttributes: [CFString: Any] = [
                kCVPixelBufferPoolMinimumBufferCountKey: 2,
            ]
            let pixelBufferAttributes: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_OneComponent8),
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferMetalCompatibilityKey: true,
            ]

            var pool: CVPixelBufferPool?
            let poolStatus = CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                poolAttributes as CFDictionary,
                pixelBufferAttributes as CFDictionary,
                &pool
            )

            guard poolStatus == kCVReturnSuccess, let pool else {
                throw NSError(domain: "StereoFrameProcessor", code: -6)
            }

            depthPixelBufferPool = pool
            depthPoolWidth = width
            depthPoolHeight = height
        }

        guard let pool = depthPixelBufferPool else {
            throw NSError(domain: "StereoFrameProcessor", code: -7)
        }

        var pixelBuffer: CVPixelBuffer?
        let auxAttributes: [CFString: Any] = [
            kCVPixelBufferPoolAllocationThresholdKey: 4,
        ]
        let status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
            kCFAllocatorDefault,
            pool,
            auxAttributes as CFDictionary,
            &pixelBuffer
        )
        if status == kCVReturnWouldExceedAllocationThreshold {
            throw NSError(domain: "StereoFrameProcessor", code: -8)
        }
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw NSError(domain: "StereoFrameProcessor", code: -9)
        }
        return pixelBuffer
    }

    private func depthWidth(_ depth: MLMultiArray) -> CGFloat {
        let shape = depth.shape.map { $0.intValue }
        if shape.count >= 2 {
            return CGFloat(shape[shape.count - 1])
        }
        return 1
    }

    private func depthHeight(_ depth: MLMultiArray) -> CGFloat {
        let shape = depth.shape.map { $0.intValue }
        if shape.count >= 2 {
            return CGFloat(shape[shape.count - 2])
        }
        return 1
    }
}

@MainActor
final class StereoVideoPlayerController: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var lastError: String?
    @Published var loadingMessage: String?
    @Published var playbackDebugState: String = "idle"
    @Published var playbackSeconds: Double = 0
    
    let player: AVPlayer
    
    private let processor: StereoFrameProcessor
    private let stereoBridge = StereoSampleBufferBridge()
    private let playerVideoOutput: AVPlayerVideoOutput
    private let processingQueue = DispatchQueue(label: "com.vision.depthplayer.processing", qos: .userInitiated)
    
    private var displayLink: CADisplayLink?
    private var fallbackTimer: Timer?
    private var outputTimer: Timer?
    private var isProcessingFrame = false
    private var pendingFrame: PendingFrame?
    private weak var rendererConfiguration: Video3DConfiguration?
    private var observedItem: AVPlayerItem?
    private var itemStatusObservation: NSKeyValueObservation?
    private var bufferEmptyObservation: NSKeyValueObservation?
    private var likelyToKeepUpObservation: NSKeyValueObservation?
    private var timeControlObservation: NSKeyValueObservation?
    private var currentItemObservation: NSKeyValueObservation?
    private var itemEndObserver: NSObjectProtocol?
    private var itemStallObserver: NSObjectProtocol?
    private var itemFailedToEndObserver: NSObjectProtocol?
    private var itemErrorLogObserver: NSObjectProtocol?
    private var itemAccessLogObserver: NSObjectProtocol?
    private var lastDeliveredFrameHostTime = CACurrentMediaTime()
    private var lastAutoResumeAttemptHostTime = 0.0
    private var lastObservedItemTimeSeconds: Double = -1
    private var unchangedItemTimeTicks = 0
    private var lastReportedDebugState = "idle"
    private var lastHighFrequencyStateLoggedSecond = -1
    private var lastNoisyStateLoggedSecond = -1
    private var lastProcessedPresentationTime: CMTime = .invalid
    private var lastLoopRestartHostTime = 0.0
    private var lastDetailedDiagnosticsSecond = -1
    private var hasProducedFirstFrame = false
    private var startupHostTime = 0.0
    private var lastStartupTimelineNudgeHostTime = 0.0
    private var noFrameIntervalState: OSSignpostIntervalState?
    private var startupIntervalState: OSSignpostIntervalState?
    private var lastFrameProcessHostTime = 0.0
    private var lastOutputEnqueueHostTime = 0.0
    private var lastActualOutputEnqueueHostTime = 0.0
    private var latestOutputItemTime: CMTime = .invalid
    private var lastRenderedStereoPixelBuffer: CVPixelBuffer?
    private var processDurationSumMs = 0.0
    private var processDurationMinMs = Double.greatestFiniteMagnitude
    private var processDurationMaxMs = 0.0
    private var processDurationCount = 0
    private var enqueueIntervalSumMs = 0.0
    private var enqueueIntervalMinMs = Double.greatestFiniteMagnitude
    private var enqueueIntervalMaxMs = 0.0
    private var enqueueIntervalCount = 0
    private let signposter = OSSignposter(subsystem: "com.vision.depth-player", category: "playback-pipeline")
    // Ultra-smooth profile: prioritize temporal stability over depth intensity.
    private let outputFrameInterval: Double = 1.0 / 90.0
    private let baseFrameProcessInterval: Double
    private var currentFrameProcessInterval: Double
    private var lastSheddingTier: Int = 0
    
    private let maxDisparity: CGFloat
    private let temporalSmoothing: Float

    private struct PendingFrame {
        let pixelBuffer: CVPixelBuffer
        let itemTime: CMTime
    }
    
    init(hlsURL: URL, model: MLModel, maxDisparity: CGFloat = 12.0, temporalSmoothing: Float = 0.85) throws {
        self.player = AVPlayer(url: hlsURL)
        self.maxDisparity = maxDisparity
        self.temporalSmoothing = min(max(temporalSmoothing, 0), 0.99)
        self.processor = try StereoFrameProcessor(
            model: model,
            maxDisparity: maxDisparity,
            temporalSmoothing: temporalSmoothing
        )
        // In depth-bypass mode, process at 30 fps and duplicate to output cadence.
        // In depth-enabled mode, keep video cadence high and reuse depth between inferences.
        if maxDisparity <= 0.001 {
            self.baseFrameProcessInterval = 1.0 / 30.0
        } else {
            self.baseFrameProcessInterval = 1.0 / 30.0
        }
        self.currentFrameProcessInterval = self.baseFrameProcessInterval
        
        // Build AVVideoOutputSpecification for monoscopic (2D) content.
        // kCMStereoView_None = CMStereoViewComponents() — no stereo eyes, i.e. regular 2D.
        let monoTag = CMTag.stereoView(CMStereoViewComponents())
        let spec = AVVideoOutputSpecification(tagCollections: [[monoTag]])
        spec.defaultPixelBufferAttributes = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        self.playerVideoOutput = AVPlayerVideoOutput(specification: spec)

        super.init()
        // Attach to player (not per-item) — no rebinding needed across item transitions.
        player.videoOutput = self.playerVideoOutput

        player.actionAtItemEnd = .none
        player.automaticallyWaitsToMinimizeStalling = false
        configureObservers()
        
    }
    
    isolated deinit {
        displayLink?.invalidate()
        fallbackTimer?.invalidate()
        outputTimer?.invalidate()
        itemStatusObservation?.invalidate()
        bufferEmptyObservation?.invalidate()
        likelyToKeepUpObservation?.invalidate()
        timeControlObservation?.invalidate()
        currentItemObservation?.invalidate()
        if let itemEndObserver {
            NotificationCenter.default.removeObserver(itemEndObserver)
        }
        if let itemStallObserver {
            NotificationCenter.default.removeObserver(itemStallObserver)
        }
        if let itemFailedToEndObserver {
            NotificationCenter.default.removeObserver(itemFailedToEndObserver)
        }
        if let itemErrorLogObserver {
            NotificationCenter.default.removeObserver(itemErrorLogObserver)
        }
        if let itemAccessLogObserver {
            NotificationCenter.default.removeObserver(itemAccessLogObserver)
        }
        player.videoOutput = nil
        NotificationCenter.default.removeObserver(self)
    }
    
    func start() {
        guard !isRunning else { return }
        isRunning = true
        loadingMessage = "Buffering stream..."
        lastError = nil
        playbackSeconds = 0
        hasProducedFirstFrame = false
        startupHostTime = CACurrentMediaTime()
        lastStartupTimelineNudgeHostTime = 0
        lastDeliveredFrameHostTime = CACurrentMediaTime()
        lastAutoResumeAttemptHostTime = 0
        lastObservedItemTimeSeconds = -1
        unchangedItemTimeTicks = 0
        lastProcessedPresentationTime = .invalid
        lastOutputEnqueueHostTime = 0
        lastActualOutputEnqueueHostTime = 0
        latestOutputItemTime = .invalid
        lastRenderedStereoPixelBuffer = nil
        resetJitterTelemetryWindow()
        setPlaybackDebugState("started")
        PlaybackFaultLogger.shared.log("controller-start")
        startupIntervalState = signposter.beginInterval("StartupToFirstFrame", id: signposter.makeSignpostID())
        signposter.emitEvent("ControllerStart", id: signposter.makeSignpostID())
        
        player.playImmediately(atRate: 1.0)
        
#if os(visionOS)
        let link = CADisplayLink(target: self, selector: #selector(step))
    // Lock to 90 Hz so output pacing does not wander between 60/90/120 domains.
    link.preferredFrameRateRange = CAFrameRateRange(minimum: 90, maximum: 90, preferred: 90)
        link.add(to: .main, forMode: .common)
        displayLink = link
#else
    fallbackTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.step()
            }
        }
#endif

#if !os(visionOS)
        outputTimer = Timer.scheduledTimer(withTimeInterval: outputFrameInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.outputTick()
            }
        }
        if let outputTimer {
            RunLoop.main.add(outputTimer, forMode: .common)
        }
#endif
    }
    
    func stop() {
        isRunning = false
        displayLink?.invalidate()
        displayLink = nil
        fallbackTimer?.invalidate()
        fallbackTimer = nil
        outputTimer?.invalidate()
        outputTimer = nil
        pendingFrame = nil
        isProcessingFrame = false
        lastOutputEnqueueHostTime = 0
        lastActualOutputEnqueueHostTime = 0
        latestOutputItemTime = .invalid
        lastRenderedStereoPixelBuffer = nil
        resetJitterTelemetryWindow()
        loadingMessage = nil
        playbackSeconds = 0
        setPlaybackDebugState("stopped")
        PlaybackFaultLogger.shared.log("controller-stop")
        signposter.emitEvent("ControllerStop", id: signposter.makeSignpostID())
        if let startupIntervalState {
            signposter.endInterval("StartupToFirstFrame", startupIntervalState)
            self.startupIntervalState = nil
        }
        if let noFrameIntervalState {
            signposter.endInterval("NoFrameGap", noFrameIntervalState)
            self.noFrameIntervalState = nil
        }
        player.pause()
        stereoBridge.flush()
    }

    func attachStereoPresentation(_ presentation: StereoPresentationCoordinator) {
        presentation.attach(renderer: stereoBridge.renderer)
    }

#if os(visionOS)
    func attachRendererConfiguration(_ configuration: Video3DConfiguration) {
        self.rendererConfiguration = configuration
        configuration.playerVideoOutput = playerVideoOutput
        configuration.rendererDebugStatus = "Player attached AVPlayerVideoOutput"
    }

    func detachRendererConfiguration(_ configuration: Video3DConfiguration) {
        if configuration.playerVideoOutput === playerVideoOutput {
            configuration.playerVideoOutput = nil
        }
        configuration.rendererDebugStatus = "Player detached AVPlayerVideoOutput"
    }

    func updateDepthStrength(_ uiValue: CGFloat) {
        // User-facing depth is 0...1.04; keep mapping in the focus-safe band.
        let clampedUI = min(max(uiValue, 0), 1.04)
        let mappedDisparity = min(max(clampedUI / 30.0, 0), 1.0)
        processor.updateMaxDisparity(mappedDisparity)
        currentFrameProcessInterval = 1.0 / 30.0
        setPlaybackDebugState(String(format: "depth-safe-ui-%.2f-map-%.2f", clampedUI, mappedDisparity))
    }
#endif
    
    @objc private func loopIfNeeded() {
        maybeRestartEndedItem(reason: "did-play-to-end")
    }

    @objc private func handlePlaybackStalled() {
        setPlaybackDebugState("notification-stalled")
        recoverPlayback(reason: "stalled")
    }

    @objc private func handleFailedToPlayToEnd() {
        setPlaybackDebugState("notification-failed-to-end")
        recoverPlayback(reason: "failed-to-end")
    }
    
    @objc private func step() {
    #if os(visionOS)
        let hostTime = displayLink?.targetTimestamp ?? CACurrentMediaTime()
    #else
        let hostTime = CACurrentMediaTime()
    #endif
        let hostCMTime = CMTime(seconds: hostTime, preferredTimescale: 1_000_000)

        // Pull frame from AVPlayerVideoOutput (player-level, no per-item rebinding needed).
        // taggedBuffers(forHostTime:) returns the frame appropriate for the given host time.
        let frameResult = playerVideoOutput.taggedBuffers(forHostTime: hostCMTime)

        let playerTime = player.currentTime()
        let rawPresentationTime = frameResult?.presentationTime
        let resolvedFrameTime: CMTime
        let mediaClockSeconds: Double?
        if let rawPresentationTime, rawPresentationTime.isValid, rawPresentationTime != .zero {
            // Prefer whichever clock is advancing. Some outputs keep a repeated presentation time.
            if playerTime.isValid, playerTime != .invalid, playerTime != .zero {
                let rawSeconds = CMTimeGetSeconds(rawPresentationTime)
                let playerSeconds = CMTimeGetSeconds(playerTime)
                if rawSeconds.isFinite, playerSeconds.isFinite, playerSeconds > rawSeconds + 0.001 {
                    resolvedFrameTime = playerTime
                    mediaClockSeconds = playerSeconds
                } else {
                    resolvedFrameTime = rawPresentationTime
                    mediaClockSeconds = rawSeconds.isFinite ? rawSeconds : nil
                }
            } else {
                resolvedFrameTime = rawPresentationTime
                let rawSeconds = CMTimeGetSeconds(rawPresentationTime)
                mediaClockSeconds = rawSeconds.isFinite ? rawSeconds : nil
            }
        } else if playerTime.isValid,
                  playerTime != .invalid,
                  playerTime != .zero {
            resolvedFrameTime = playerTime
            let playerSeconds = CMTimeGetSeconds(playerTime)
            mediaClockSeconds = playerSeconds.isFinite ? playerSeconds : nil
        } else {
            // Do not use host uptime as media time; it can trigger false end-of-item handling.
            resolvedFrameTime = .invalid
            mediaClockSeconds = nil
        }

        // Update playback time only from media clocks (player/item output), never host uptime.
        if let mediaClockSeconds, mediaClockSeconds >= 0 {
            playbackSeconds = mediaClockSeconds
            if abs(mediaClockSeconds - lastObservedItemTimeSeconds) < 0.0001 {
                unchangedItemTimeTicks += 1
            } else {
                unchangedItemTimeTicks = 0
                lastObservedItemTimeSeconds = mediaClockSeconds
            }
        }

        // On visionOS, drive output enqueue from display-link time to avoid Timer jitter.
        let displayItemTime: CMTime
        if latestOutputItemTime.isValid, latestOutputItemTime != .zero {
            displayItemTime = latestOutputItemTime
        } else if resolvedFrameTime.isValid, resolvedFrameTime != .zero {
            displayItemTime = resolvedFrameTime
        } else {
            displayItemTime = playerTime
        }
        if displayItemTime.isValid, displayItemTime != .zero {
            enqueueDuplicateFrameIfNeeded(itemTime: displayItemTime, hostTime: hostTime)
        }

        emitDiagnosticsIfNeeded(hostTime: hostTime, hasFrame: frameResult != nil)

        // Startup watchdog: if timeline remains pinned at 0, force a gentle timeline nudge.
        if isRunning,
           hostTime - startupHostTime > 2.0,
           playbackSeconds < 0.01,
           let item = player.currentItem,
           item.status == .readyToPlay,
           hasProducedFirstFrame,
           hostTime - lastStartupTimelineNudgeHostTime > 2.0 {
            lastStartupTimelineNudgeHostTime = hostTime
            PlaybackFaultLogger.shared.log("startup-zero-time-nudge", fields: [
                "seconds": String(Int(playbackSeconds)),
                "rate": String(format: "%.2f", player.rate),
            ])
            player.seek(to: CMTime(value: 1, timescale: 600), toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.player.playImmediately(atRate: 1.0)
                    self?.setPlaybackDebugState("startup-timeline-nudged")
                }
            }
        }

        let itemSeconds = CMTimeGetSeconds(playerTime)

        // Duration-boundary loop detection using player timeline only.
        if let item = player.currentItem {
            let durationSeconds = CMTimeGetSeconds(item.duration)
            if durationSeconds.isFinite, durationSeconds > 0,
               itemSeconds.isFinite, itemSeconds >= max(0, durationSeconds - 0.20) {
                maybeRestartEndedItem(reason: "duration-boundary")
                return
            }
        }

        guard let frameResult else {
            if isRunning {
                let now = CACurrentMediaTime()
                if noFrameIntervalState == nil {
                    noFrameIntervalState = signposter.beginInterval("NoFrameGap", id: signposter.makeSignpostID())
                }
                if unchangedItemTimeTicks > 90 {
                    setPlaybackDebugState("item-time-not-advancing")
                    if unchangedItemTimeTicks % 180 == 0 {
                        logPlaybackSnapshot(context: "item-time-not-advancing")
                    }
                    if unchangedItemTimeTicks > 150 && isNearItemEnd() {
                        maybeRestartEndedItem(reason: "stalled-near-end")
                        return
                    }
                }
                if now - lastDeliveredFrameHostTime > 6.0 {
                    recoverPlayback(reason: "no-frames")
                }
            }
            return
        }

        lastDeliveredFrameHostTime = CACurrentMediaTime()
        if let noFrameIntervalState {
            signposter.endInterval("NoFrameGap", noFrameIntervalState)
            self.noFrameIntervalState = nil
        }
        if player.timeControlStatus == .playing {
            setPlaybackDebugState("streaming-frames")
        }

        let frameTimeForProcessing = (resolvedFrameTime.isValid && resolvedFrameTime != .zero) ? resolvedFrameTime : hostCMTime
        latestOutputItemTime = frameTimeForProcessing
        lastProcessedPresentationTime = resolvedFrameTime

        if hostTime - lastFrameProcessHostTime < currentFrameProcessInterval {
            return
        }
        lastFrameProcessHostTime = hostTime

        // Extract CVPixelBuffer from the tagged buffer group.
        // AVPlayerVideoOutput may surface either direct pixel buffers or sample buffers.
        var pixelBuffer: CVPixelBuffer?
        for taggedBuffer in frameResult.taggedBufferGroup {
            if case .pixelBuffer(let pb) = taggedBuffer.buffer {
                pixelBuffer = pb
                break
            }
            if case .sampleBuffer(let sb) = taggedBuffer.buffer,
               let pb = CMSampleBufferGetImageBuffer(sb) {
                pixelBuffer = pb
                break
            }
        }
        guard let pixelBuffer else {
            setPlaybackDebugState("tagged-buffer-no-cvpixelbuffer")
            return
        }

        if isProcessingFrame {
            pendingFrame = PendingFrame(pixelBuffer: pixelBuffer, itemTime: frameTimeForProcessing)
            return
        }

        processFrame(pixelBuffer, itemTime: frameTimeForProcessing)
    }

    @objc private func outputTick() {
        guard isRunning else { return }
#if os(visionOS)
        return
#else
        let hostTime = CACurrentMediaTime()
        let itemTime: CMTime
        if latestOutputItemTime.isValid, latestOutputItemTime != .zero {
            itemTime = latestOutputItemTime
        } else {
            itemTime = player.currentTime()
        }
        enqueueDuplicateFrameIfNeeded(itemTime: itemTime, hostTime: hostTime)
#endif
    }

    private func configureObservers() {
        currentItemObservation?.invalidate()
        currentItemObservation = player.observe(\.currentItem, options: [.initial, .new]) { [weak self] player, _ in
            Task { @MainActor in
                self?.bindCurrentItem(player.currentItem)
            }
        }
    }

    private func bindCurrentItem(_ item: AVPlayerItem?) {
        guard item !== observedItem else { return }

        itemStatusObservation?.invalidate()
        bufferEmptyObservation?.invalidate()
        likelyToKeepUpObservation?.invalidate()
        timeControlObservation?.invalidate()
        if let itemEndObserver {
            NotificationCenter.default.removeObserver(itemEndObserver)
            self.itemEndObserver = nil
        }
        if let itemStallObserver {
            NotificationCenter.default.removeObserver(itemStallObserver)
            self.itemStallObserver = nil
        }
        if let itemFailedToEndObserver {
            NotificationCenter.default.removeObserver(itemFailedToEndObserver)
            self.itemFailedToEndObserver = nil
        }

        observedItem = item
        guard let item else {
            loadingMessage = "Waiting for player item..."
            setPlaybackDebugState("missing-current-item")
            return
        }

        setPlaybackDebugState("bound-new-item")

        // Keep AVFoundation buffering bounded so it doesn't consume the same jetsam budget
        // as the compositor and depth pipeline.
        // Keep a 10-second forward buffer so the depth pipeline's processing overhead
        // doesn't starve AVPlayer — a 1-second buffer was running dry at second 239.
        item.preferredForwardBufferDuration = 10.0
        item.canUseNetworkResourcesForLiveStreamingWhilePaused = false
        // Prefer 4K variants when the stream ladder provides them.
        item.preferredMaximumResolution = CGSize(width: 3840, height: 2160)
        item.preferredPeakBitRate = 120_000_000

        itemEndObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.loopIfNeeded()
            }
        }

        itemStallObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemPlaybackStalled,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handlePlaybackStalled()
            }
        }

        itemFailedToEndObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemFailedToPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleFailedToPlayToEnd()
            }
        }

        itemErrorLogObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemNewErrorLogEntry,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let message = item.errorLog()?.events.last?.errorComment ?? "unknown"
                self.setPlaybackDebugState("error-log: \(message)")
            }
        }

        itemAccessLogObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemNewAccessLogEntry,
            object: item,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.setPlaybackDebugState("access-log-update")
            }
        }

        itemStatusObservation = item.observe(\.status, options: [.initial, .new]) { [weak self] item, _ in
            Task { @MainActor in
                guard let self else { return }
                switch item.status {
                case .unknown:
                    self.loadingMessage = "Preparing stream..."
                    self.setPlaybackDebugState("item-status-unknown")
                case .readyToPlay:
                    if !self.hasProducedFirstFrame {
                        self.loadingMessage = "Waiting for first frame..."
                    }
                    self.setPlaybackDebugState("item-ready")
                case .failed:
                    self.lastError = item.error?.localizedDescription ?? "Stream failed to load"
                    self.loadingMessage = nil
                    self.setPlaybackDebugState("item-failed")
                    self.recoverPlayback(reason: "item-failed")
                @unknown default:
                    self.loadingMessage = "Preparing stream..."
                    self.setPlaybackDebugState("item-status-unknown-default")
                }
            }
        }

        bufferEmptyObservation = item.observe(\.isPlaybackBufferEmpty, options: [.initial, .new]) { [weak self] item, _ in
            Task { @MainActor in
                guard let self else { return }
                if item.isPlaybackBufferEmpty {
                    self.loadingMessage = "Buffering stream..."
                    self.setPlaybackDebugState("buffer-empty")
                }
            }
        }

        likelyToKeepUpObservation = item.observe(\.isPlaybackLikelyToKeepUp, options: [.initial, .new]) { [weak self] item, _ in
            Task { @MainActor in
                guard let self else { return }
                if item.isPlaybackLikelyToKeepUp, !self.hasProducedFirstFrame {
                    self.loadingMessage = "Running depth estimation..."
                }
                if item.isPlaybackLikelyToKeepUp {
                    self.setPlaybackDebugState("likely-to-keep-up")
                }
            }
        }

        timeControlObservation = player.observe(\.timeControlStatus, options: [.initial, .new]) { [weak self] player, _ in
            Task { @MainActor in
                guard let self else { return }
                switch player.timeControlStatus {
                case .paused:
                    if self.isRunning {
                        if !self.hasProducedFirstFrame {
                            self.loadingMessage = "Paused before first frame"
                        }
                        self.setPlaybackDebugState("time-control-paused")
                        self.recoverPlayback(reason: "paused")
                    }
                case .waitingToPlayAtSpecifiedRate:
                    self.loadingMessage = "Buffering stream..."
                    self.setPlaybackDebugState("time-control-waiting")
                case .playing:
                    if !self.hasProducedFirstFrame {
                        self.loadingMessage = "Running depth estimation..."
                    }
                    self.setPlaybackDebugState("time-control-playing")
                @unknown default:
                    self.setPlaybackDebugState("time-control-unknown")
                    break
                }
            }
        }
    }

    private func recoverPlayback(reason: String) {
        guard isRunning else { return }
        guard player.currentItem != nil else {
            loadingMessage = "Playback item missing"
            return
        }

        if isNearItemEnd() {
            setPlaybackDebugState("recover-near-end-\(reason)")
            maybeRestartEndedItem(reason: "recover-\(reason)-near-end")
            return
        }

        let now = CACurrentMediaTime()
        guard now - lastAutoResumeAttemptHostTime >= 1.5 else { return }
        lastAutoResumeAttemptHostTime = now

        if player.rate == 0 || player.timeControlStatus != .playing {
            loadingMessage = "Recovering playback..."
            setPlaybackDebugState("recovering-\(reason)")
            PlaybackFaultLogger.shared.log("recover-playback", fields: ["reason": reason])
            logPlaybackSnapshot(context: "recover-\(reason)")
            signposter.emitEvent("RecoverPlayback", id: signposter.makeSignpostID())
            player.playImmediately(atRate: 1.0)
            #if DEBUG
            print("DepthPlayer: auto-resume triggered (\(reason))")
            #endif
        }
    }

    private func setPlaybackDebugState(_ state: String) {
        guard state != lastReportedDebugState else { return }
        lastReportedDebugState = state
        playbackDebugState = state
        NSLog("DepthPlayerPlaybackState: %@", state)

        // Avoid disk-write resource pressure by throttling very high-frequency states.
        let currentSecond = Int(playbackSeconds)
        if state == "frame-processed" || state == "streaming-frames" {
            if currentSecond == lastHighFrequencyStateLoggedSecond {
                return
            }
            lastHighFrequencyStateLoggedSecond = currentSecond
        }

        if state == "request-media-data-change" ||
            state == "request-media-data-change-throttled" ||
            state == "media-data-will-change" {
            if currentSecond == lastNoisyStateLoggedSecond {
                return
            }
            lastNoisyStateLoggedSecond = currentSecond
        }

        PlaybackFaultLogger.shared.log("playback-state", fields: [
            "state": state,
            "seconds": String(Int(playbackSeconds)),
        ])
    }

    private func processFrame(_ pixelBuffer: CVPixelBuffer, itemTime: CMTime) {
        isProcessingFrame = true
        let processor = self.processor
        let processingStartHostTime = CACurrentMediaTime()

        processingQueue.async { [weak self] in
            let result = autoreleasepool(invoking: {
                Result { try processor.process(pixelBuffer: pixelBuffer) }
            })
            let processingDurationMs = (CACurrentMediaTime() - processingStartHostTime) * 1000.0

            Task { @MainActor [weak self] in
                guard let self else { return }
                self.recordProcessingDurationMs(processingDurationMs)

                self.isProcessingFrame = false

                switch result {
                case let .success(frame):
                    self.lastError = nil
                    self.loadingMessage = nil
                    self.hasProducedFirstFrame = true
                    self.rendererConfiguration?.depthFrameCount += 1
                    self.rendererConfiguration?.receivedVideoFrameCount += 1
                    self.setPlaybackDebugState("frame-processed")
                    if let startupIntervalState = self.startupIntervalState {
                        self.signposter.endInterval("StartupToFirstFrame", startupIntervalState)
                        self.startupIntervalState = nil
                        PlaybackFaultLogger.shared.log("startup-to-first-frame-finished", fields: [
                            "seconds": String(Int(self.playbackSeconds)),
                        ])
                    }
                    // Update the latest processed stereo frame; fixed-rate output enqueues
                    // are performed on the display tick in enqueueDuplicateFrameIfNeeded().
                    self.lastRenderedStereoPixelBuffer = frame.stereoPixelBuffer
                case let .failure(error):
                    if let nsError = error as NSError?, nsError.domain == "StereoFrameProcessor", nsError.code == -5 {
                        // Pool allocation threshold reached: skip this frame to cap memory growth.
                        self.setPlaybackDebugState("frame-drop-pool-threshold")
                    } else {
                        self.lastError = error.localizedDescription
                        self.setPlaybackDebugState("processing-error: \(error.localizedDescription)")
                    }
                }

                if let pendingFrame = self.pendingFrame {
                    self.pendingFrame = nil
                    self.processFrame(pendingFrame.pixelBuffer, itemTime: pendingFrame.itemTime)
                }
            }
        }
    }

    private func enqueueDuplicateFrameIfNeeded(itemTime: CMTime, hostTime: CFTimeInterval) {
        guard let lastRenderedStereoPixelBuffer else { return }
#if !os(visionOS)
        guard hostTime - lastOutputEnqueueHostTime >= outputFrameInterval else { return }
#endif

        let enqueueDuration = CMTime(seconds: outputFrameInterval, preferredTimescale: 1_000_000)
        do {
            try stereoBridge.enqueue(pixelBuffer: lastRenderedStereoPixelBuffer, at: itemTime, duration: enqueueDuration)
            rendererConfiguration?.renderedFrameCount += 1
            recordOutputEnqueue(hostTime: hostTime)
#if os(visionOS)
            lastOutputEnqueueHostTime = hostTime
#else
            if lastOutputEnqueueHostTime == 0 {
                lastOutputEnqueueHostTime = hostTime
            } else {
                let advancedTime = lastOutputEnqueueHostTime + outputFrameInterval
                // If we are significantly behind, resync to avoid burst catch-up behavior.
                lastOutputEnqueueHostTime = (hostTime - advancedTime > outputFrameInterval) ? hostTime : advancedTime
            }
#endif
        } catch {
            // Recover from transient renderer/format state mismatches by flushing once.
            stereoBridge.flushForReattach()
            do {
                try stereoBridge.enqueue(pixelBuffer: lastRenderedStereoPixelBuffer, at: itemTime, duration: enqueueDuration)
                rendererConfiguration?.renderedFrameCount += 1
                recordOutputEnqueue(hostTime: hostTime)
#if os(visionOS)
                lastOutputEnqueueHostTime = hostTime
#else
                if lastOutputEnqueueHostTime == 0 {
                    lastOutputEnqueueHostTime = hostTime
                } else {
                    let advancedTime = lastOutputEnqueueHostTime + outputFrameInterval
                    lastOutputEnqueueHostTime = (hostTime - advancedTime > outputFrameInterval) ? hostTime : advancedTime
                }
#endif
                setPlaybackDebugState("frame-enqueue-recovered")
            } catch {
                lastError = "enqueue-failed: \(error.localizedDescription)"
                setPlaybackDebugState("enqueue-failed")
            }
        }
    }

    private func logPlaybackSnapshot(context: String) {
        guard let item = player.currentItem else {
            PlaybackFaultLogger.shared.log("playback-snapshot", fields: [
                "context": context,
                "item": "missing",
            ])
            return
        }

        let currentTime = item.currentTime().seconds
        let duration = item.duration.seconds

        let waitingReason: String
        if let reason = player.reasonForWaitingToPlay {
            waitingReason = reason.rawValue
        } else {
            waitingReason = "none"
        }

        let accessEvent = item.accessLog()?.events.last
        let accessSummary = [
            "obs=" + String(format: "%.2f", accessEvent?.observedBitrate ?? 0),
            "ind=" + String(format: "%.2f", accessEvent?.indicatedBitrate ?? 0),
            "stall=" + String(accessEvent?.numberOfStalls ?? 0),
            "switch=" + String(accessEvent?.numberOfServerAddressChanges ?? 0),
        ].joined(separator: ",")

        let loadedRanges = item.loadedTimeRanges
            .compactMap { $0.timeRangeValue }
            .map { range in
                let start = CMTimeGetSeconds(range.start)
                let end = CMTimeGetSeconds(range.start + range.duration)
                return String(format: "%.2f-%.2f", start, end)
            }
            .joined(separator: "|")

        PlaybackFaultLogger.shared.log("playback-snapshot", fields: [
            "context": context,
            "item_status": String(item.status.rawValue),
            "time_control": String(player.timeControlStatus.rawValue),
            "waiting_reason": waitingReason,
            "rate": String(format: "%.2f", player.rate),
            "current_s": String(format: "%.2f", currentTime.isFinite ? currentTime : -1),
            "duration_s": String(format: "%.2f", duration.isFinite ? duration : -1),
            "presentation_s": String(format: "%.2f", lastProcessedPresentationTime.isValid ? CMTimeGetSeconds(lastProcessedPresentationTime) : -1),
            "buffer_empty": item.isPlaybackBufferEmpty ? "true" : "false",
            "likely_to_keep_up": item.isPlaybackLikelyToKeepUp ? "true" : "false",
            "loaded_ranges": loadedRanges.isEmpty ? "none" : loadedRanges,
            "access": accessSummary,
        ])
    }

    private func maybeRestartEndedItem(reason: String) {
        guard isRunning else { return }

        let now = CACurrentMediaTime()
        guard now - lastLoopRestartHostTime > 0.75 else { return }
        lastLoopRestartHostTime = now

        setPlaybackDebugState("looping-ended-item")
        PlaybackFaultLogger.shared.log("item-ended-loop-restart", fields: [
            "reason": reason,
            "seconds": String(Int(playbackSeconds)),
        ])

        player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.lastDeliveredFrameHostTime = CACurrentMediaTime()
                self.lastObservedItemTimeSeconds = -1
                self.unchangedItemTimeTicks = 0
                    self.lastProcessedPresentationTime = .invalid
                    self.player.playImmediately(atRate: 1.0)
                self.setPlaybackDebugState("loop-restarted")
            }
        }
    }

    private func isNearItemEnd(thresholdSeconds: Double = 1.0) -> Bool {
        guard let item = player.currentItem else { return false }
        let duration = CMTimeGetSeconds(item.duration)
        guard duration.isFinite, duration > 0 else { return false }

        let itemTime = CMTimeGetSeconds(item.currentTime())
        if itemTime.isFinite, itemTime >= max(0, duration - thresholdSeconds) {
            return true
        }

        return playbackSeconds >= max(0, duration - thresholdSeconds)
    }

    private func emitDiagnosticsIfNeeded(hostTime: CFTimeInterval, hasFrame: Bool) {
        let currentSecond = Int(playbackSeconds)
        if currentSecond < 0 || currentSecond == lastDetailedDiagnosticsSecond {
            return
        }

        let isDeepWindow = (28...40).contains(currentSecond)
        if !isDeepWindow && currentSecond % 15 != 0 {
            return
        }
        lastDetailedDiagnosticsSecond = currentSecond

        let presentationSeconds = lastProcessedPresentationTime.isValid ? CMTimeGetSeconds(lastProcessedPresentationTime) : -1.0

        let memory = currentProcessMemoryStatsMB()
        var fields: [String: String] = [
            "seconds": String(currentSecond),
            "window": isDeepWindow ? "deep" : "periodic",
            "time_control": String(player.timeControlStatus.rawValue),
            "rate": String(format: "%.2f", player.rate),
            "presentation_s": String(format: "%.3f", presentationSeconds.isFinite ? presentationSeconds : -1),
            "has_frame": hasFrame ? "true" : "false",
            "is_processing": isProcessingFrame ? "true" : "false",
            "pending_frame": pendingFrame == nil ? "false" : "true",
            "unchanged_ticks": String(unchangedItemTimeTicks),
        ]

        if let rssMB = memory.rssMB {
            fields["rss_mb"] = String(format: "%.1f", rssMB)
        }
        if let footprintMB = memory.footprintMB {
            fields["footprint_mb"] = String(format: "%.1f", footprintMB)
            adjustLoadSheddingForMemory(footprintMB: footprintMB, second: currentSecond)
            fields["process_interval_s"] = String(format: "%.3f", currentFrameProcessInterval)
            fields["shedding_tier"] = String(lastSheddingTier)
        }

        if processDurationCount > 0 {
            let processAvgMs = processDurationSumMs / Double(processDurationCount)
            fields["proc_ms_avg"] = String(format: "%.2f", processAvgMs)
            fields["proc_ms_min"] = String(format: "%.2f", processDurationMinMs)
            fields["proc_ms_max"] = String(format: "%.2f", processDurationMaxMs)
            fields["proc_count"] = String(processDurationCount)
        }

        if enqueueIntervalCount > 0 {
            let enqueueAvgMs = enqueueIntervalSumMs / Double(enqueueIntervalCount)
            fields["enqueue_ms_avg"] = String(format: "%.2f", enqueueAvgMs)
            fields["enqueue_ms_min"] = String(format: "%.2f", enqueueIntervalMinMs)
            fields["enqueue_ms_max"] = String(format: "%.2f", enqueueIntervalMaxMs)
            fields["enqueue_count"] = String(enqueueIntervalCount)
        }

        PlaybackFaultLogger.shared.log("pipeline-heartbeat", fields: fields)
        resetJitterTelemetryWindow()
    }

    private func recordProcessingDurationMs(_ durationMs: Double) {
        processDurationCount += 1
        processDurationSumMs += durationMs
        processDurationMinMs = min(processDurationMinMs, durationMs)
        processDurationMaxMs = max(processDurationMaxMs, durationMs)
    }

    private func recordOutputEnqueue(hostTime: CFTimeInterval) {
        if lastActualOutputEnqueueHostTime > 0 {
            let intervalMs = (hostTime - lastActualOutputEnqueueHostTime) * 1000.0
            enqueueIntervalCount += 1
            enqueueIntervalSumMs += intervalMs
            enqueueIntervalMinMs = min(enqueueIntervalMinMs, intervalMs)
            enqueueIntervalMaxMs = max(enqueueIntervalMaxMs, intervalMs)
        }
        lastActualOutputEnqueueHostTime = hostTime
    }

    private func resetJitterTelemetryWindow() {
        processDurationSumMs = 0
        processDurationMinMs = Double.greatestFiniteMagnitude
        processDurationMaxMs = 0
        processDurationCount = 0
        enqueueIntervalSumMs = 0
        enqueueIntervalMinMs = Double.greatestFiniteMagnitude
        enqueueIntervalMaxMs = 0
        enqueueIntervalCount = 0
    }

    private func currentProcessMemoryStatsMB() -> (rssMB: Double?, footprintMB: Double?) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundPointer, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return (nil, nil)
        }

        let bytesToMB = 1.0 / (1024.0 * 1024.0)
        let rssMB = Double(info.resident_size) * bytesToMB
        let footprintMB = Double(info.phys_footprint) * bytesToMB
        return (rssMB, footprintMB)
    }

    private func adjustLoadSheddingForMemory(footprintMB: Double, second: Int) {
        let newTier: Int
        let newInterval: Double

        if footprintMB >= 4700 {
            newTier = 3
            newInterval = 1.0 / 2.0
        } else if footprintMB >= 4400 {
            newTier = 2
            newInterval = 1.0 / 4.0
        } else if footprintMB >= 4000 {
            newTier = 1
            newInterval = 1.0 / 8.0
        } else {
            newTier = 0
            newInterval = baseFrameProcessInterval
        }

        guard newTier != lastSheddingTier else {
            currentFrameProcessInterval = newInterval
            return
        }

        lastSheddingTier = newTier
        currentFrameProcessInterval = newInterval
        if newTier >= 2 {
            // Drop queued work so memory pressure can recover before a per-process kill.
            pendingFrame = nil
        }

        PlaybackFaultLogger.shared.log("memory-pressure-shedding", fields: [
            "seconds": String(second),
            "footprint_mb": String(format: "%.1f", footprintMB),
            "tier": String(newTier),
            "process_interval_s": String(format: "%.3f", newInterval),
        ])
    }
}

#if os(visionOS)
struct StereoImmersivePlaybackView: View {
    @EnvironmentObject private var stereoPresentation: StereoPresentationCoordinator
    private static let entityName = "DepthPlayerStereoEntity"

    var body: some View {
        RealityView { content in
            let root = Entity()
            root.name = Self.entityName
            root.position = [0, 1.2, -2.0]
            content.add(root)
        } update: { content in
            guard let root = content.entities.first(where: { $0.name == Self.entityName }) else {
                return
            }

            if let renderer = stereoPresentation.renderer {
                var component = VideoPlayerComponent(videoRenderer: renderer)
                component.desiredViewingMode = .stereo
                component.desiredImmersiveViewingMode = .full
                if #available(visionOS 26.0, *) {
                    component.desiredSpatialVideoMode = .spatial
                }
                root.components[VideoPlayerComponent.self] = component
            } else {
                root.components.remove(VideoPlayerComponent.self)
            }
        }
        .ignoresSafeArea()
        .onAppear {
            PlaybackFaultLogger.shared.log("immersive-view-on-appear")
            stereoPresentation.setStatus("Immersive view appeared")
            // Flush stale frames so VideoPlayerComponent receives only fresh ones.
            stereoPresentation.flushRenderer()
        }
        .onDisappear {
            PlaybackFaultLogger.shared.log("immersive-view-on-disappear")
            stereoPresentation.setStatus("Immersive view disappeared")
        }
        .overlay(alignment: .topLeading) {
            Button(action: {
                stereoPresentation.requestStopPlayback()
            }) {
                Image(systemName: "chevron.backward.circle.fill")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.96))
                    .shadow(color: .black.opacity(0.5), radius: 10, x: 0, y: 2)
            }
            .buttonStyle(.plain)
            .padding(.top, 18)
            .padding(.leading, 18)
            .accessibilityLabel("Back")
        }
    }
}
#endif

private struct PanelFrameOriginPreferenceKey: PreferenceKey {
    static let defaultValue: CGPoint = .zero

    static func reduce(value: inout CGPoint, nextValue: () -> CGPoint) {
        value = nextValue()
    }
}

struct StereoVideoPlayerView: View {
    private static let immersiveSignposter = OSSignposter(subsystem: "com.vision.depth-player", category: "immersive")
    private static let controlPanelSize = CGSize(width: 460, height: 280)
    @StateObject private var controller: StereoVideoPlayerController
    @Binding var isPlaying: Bool
    private let onUserStop: () -> Void
#if os(visionOS)
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    private let safeDepthCeiling: Double = 1.04
    private let rendererConfiguration: Video3DConfiguration
    private let showRendererMetrics: Bool
    private let autoOpenImmersiveOnAppear: Bool
    @State private var depthStrength: Double = 1.04
    @State private var depthScaleSlider: Double = 6.0
    @State private var rendererDiagnostics: String = "Renderer diagnostics pending..."
    @State private var panelOffsetXDisplay: Double = 0.0
    @State private var panelOffsetYDisplay: Double = 0.0
    @State private var panelOffsetZDisplay: Double = -1.0
    @State private var lastLoggedPanelOrigin: CGPoint?
    @State private var diagnosticsTask: Task<Void, Never>?
    @State private var didRequestImmersive = false
    @State private var isImmersiveOpen = false
    @State private var userInitiatedStop = false
#endif
    @EnvironmentObject private var stereoPresentation: StereoPresentationCoordinator
    @Environment(\.scenePhase) private var scenePhase
    
    #if os(visionOS)
    init(
        hlsURL: URL,
        isPlaying: Binding<Bool>,
        rendererConfiguration: Video3DConfiguration,
        showRendererMetrics: Bool = true,
        autoOpenImmersiveOnAppear: Bool = false,
        onUserStop: @escaping () -> Void = {}
    ) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        do {
            let model = try DepthAnythingModelLoader.loadBundledModel(configuration: config)
            let playerController = try StereoVideoPlayerController(
                hlsURL: hlsURL,
                model: model,
                maxDisparity: 0.0,
                temporalSmoothing: 0.0
            )
            _controller = StateObject(wrappedValue: playerController)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
        
        self._isPlaying = isPlaying
        self.onUserStop = onUserStop
        self.rendererConfiguration = rendererConfiguration
        self.showRendererMetrics = showRendererMetrics
        self.autoOpenImmersiveOnAppear = autoOpenImmersiveOnAppear
    }

    private var appliedDepthStrength: Double {
        min(max(depthStrength, 0), safeDepthCeiling)
    }

    private func mapDepthUIToDisparity(_ uiValue: Double) -> Double {
        let clampedUI = min(max(uiValue, 0), 1.04)
        return min(max(clampedUI / 30.0, 0), 1.0)
    }

    private func mapSliderScaleToDepth(_ sliderValue: Double) -> Double {
        let clamped = min(max(sliderValue, 1.0), 6.0)
        return ((clamped - 1.0) / 5.0) * 1.04
    }

    private func mapDepthToSliderScale(_ depthValue: Double) -> Double {
        let clamped = min(max(depthValue, 0.0), 1.04)
        return 1.0 + ((clamped / 1.04) * 5.0)
    }

    private var appliedParallaxPixels: Double {
        min(max(mapDepthUIToDisparity(appliedDepthStrength) * 150.0, 0), 150.0)
    }

    #else
    init(hlsURL: URL, isPlaying: Binding<Bool>, onUserStop: @escaping () -> Void = {}) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        do {
            let model = try DepthAnythingModelLoader.loadBundledModel(configuration: config)
            let playerController = try StereoVideoPlayerController(
                hlsURL: hlsURL,
                model: model,
                maxDisparity: 0.0,
                temporalSmoothing: 0.0
            )
            _controller = StateObject(wrappedValue: playerController)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
        
        self._isPlaying = isPlaying
        self.onUserStop = onUserStop
    }
    #endif
    
    var body: some View {
        ZStack {
            Color.clear.ignoresSafeArea()

            VStack(spacing: 10) {
#if os(visionOS)
                VStack(spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: "line.3.horizontal")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundColor(.white.opacity(0.75))
                        Text("Window Handle")
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundColor(.white.opacity(0.75))
                        Capsule()
                            .fill(Color.white.opacity(0.5))
                            .frame(width: 54, height: 4)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.bottom, 4)

                    HStack {
                        Text("Panel Window Coordinates")
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white.opacity(0.9))
                        Spacer()
                        Text("Unavailable from visionOS API")
                            .font(.system(size: 11, weight: .medium, design: .rounded))
                            .foregroundColor(.white.opacity(0.75))
                    }

                    HStack {
                        Text("Immersive Panel Coordinates")
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white.opacity(0.9))
                        Spacer()
                        Text(String(format: "X %.2f  Y %.2f  Z %.2f", panelOffsetXDisplay, panelOffsetYDisplay, panelOffsetZDisplay))
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.75))
                    }

                    HStack {
                        Text("3D Depth Scale")
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white)
                        Spacer()
                        Text(String(format: "%.0f", depthScaleSlider))
                            .font(.system(size: 12, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.75))
                    }

                    HStack {
                        Text("Applied Parallax")
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white.opacity(0.9))
                        Spacer()
                        Text(String(format: "depth %.2f  shift %.1f px", appliedDepthStrength, appliedParallaxPixels))
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.75))
                    }

                    Slider(value: $depthScaleSlider, in: 1...6, step: 1) {
                        Text("Depth Scale")
                    } minimumValueLabel: {
                        Text("1")
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    } maximumValueLabel: {
                        Text("6")
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    }
                        .tint(.white.opacity(0.95))
                        .onChange(of: depthScaleSlider) { _, newValue in
                            depthStrength = mapSliderScaleToDepth(newValue)
                            rendererConfiguration.disparityStrength = CGFloat(mapDepthUIToDisparity(depthStrength))
                            controller.updateDepthStrength(CGFloat(depthStrength))
                            rendererConfiguration.rendererDebugStatus = "Depth scale \(String(format: "%.0f", newValue))"
                        }
                }
                .frame(minWidth: 320, maxWidth: 420)

                if showRendererMetrics {
                    Text(rendererDiagnostics)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(.white.opacity(0.85))
                        .frame(minWidth: 320, maxWidth: 420, alignment: .leading)
                        .lineLimit(1)
                }
#endif
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(Color.black.opacity(0.45))
            )

        }
        .frame(width: Self.controlPanelSize.width, height: Self.controlPanelSize.height, alignment: .topLeading)
        .background(
            GeometryReader { proxy in
                Color.clear
                    .preference(
                        key: PanelFrameOriginPreferenceKey.self,
                        value: proxy.frame(in: .global).origin
                    )
            }
        )
        .onPreferenceChange(PanelFrameOriginPreferenceKey.self) { origin in
#if os(visionOS)
            logPanelMovementProbe(origin)
#endif
        }
        .overlay(alignment: .topLeading) {
            Button(action: {
                triggerUserStop()
            }) {
                Image(systemName: "chevron.backward.circle.fill")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.96))
                    .shadow(color: .black.opacity(0.45), radius: 8, x: 0, y: 2)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Back")
            .padding(.top, 18)
            .padding(.leading, 18)
        }
        .onAppear {
            PlaybackFaultLogger.shared.log("view-on-appear", fields: [
                "autoOpenImmersiveOnAppear": autoOpenImmersiveOnAppear ? "true" : "false",
            ])
            controller.attachStereoPresentation(stereoPresentation)
            controller.attachRendererConfiguration(rendererConfiguration)
            depthStrength = 1.04
            depthScaleSlider = mapDepthToSliderScale(depthStrength)
            rendererConfiguration.disparityStrength = CGFloat(mapDepthUIToDisparity(depthStrength))
            controller.updateDepthStrength(CGFloat(depthStrength))
            panelOffsetXDisplay = Double(rendererConfiguration.panelOffsetX)
            panelOffsetYDisplay = Double(rendererConfiguration.panelOffsetY)
            panelOffsetZDisplay = Double(rendererConfiguration.panelOffsetZ)
            lastLoggedPanelOrigin = nil
            isPlaying = true
            controller.start()
            rendererDiagnostics = "sec=\(Int(controller.playbackSeconds)) vf=\(rendererConfiguration.receivedVideoFrameCount) rf=\(rendererConfiguration.renderedFrameCount) df=\(rendererConfiguration.depthFrameCount)"
#if os(visionOS)
            rendererConfiguration.rendererDebugStatus = "Ready to open immersive on user action"
            userInitiatedStop = false
            diagnosticsTask?.cancel()
            diagnosticsTask = Task {
                while !Task.isCancelled {
                    await MainActor.run {
                        rendererDiagnostics = "sec=\(Int(controller.playbackSeconds)) vf=\(rendererConfiguration.receivedVideoFrameCount) rf=\(rendererConfiguration.renderedFrameCount) df=\(rendererConfiguration.depthFrameCount)"
                        panelOffsetXDisplay = Double(rendererConfiguration.panelOffsetX)
                        panelOffsetYDisplay = Double(rendererConfiguration.panelOffsetY)
                        panelOffsetZDisplay = Double(rendererConfiguration.panelOffsetZ)
                    }
                    try? await Task.sleep(nanoseconds: 400_000_000)
                }
            }
            if autoOpenImmersiveOnAppear {
                requestImmersiveOpen()
            }
#endif
        }
        .onChange(of: isPlaying) { _, newIsPlaying in
            guard !newIsPlaying else { return }
#if os(visionOS)
            guard userInitiatedStop else {
                PlaybackFaultLogger.shared.log("external-stop-ignored")
                rendererConfiguration.rendererDebugStatus = "Ignored external stop; keeping playback alive"
                stereoPresentation.setStatus("External stop ignored; playback kept alive")
                isPlaying = true
                return
            }
#endif
            PlaybackFaultLogger.shared.log("user-stop-accepted")
            controller.stop()
            stereoPresentation.detach()
#if os(visionOS)
            diagnosticsTask?.cancel()
            diagnosticsTask = nil
            Task {
                if isImmersiveOpen {
                    await dismissImmersiveSpace()
                    isImmersiveOpen = false
                    stereoPresentation.setStatus("Immersive playback closed")
                    rendererConfiguration.rendererDebugStatus = "Immersive space dismissed"
                }
            }
            didRequestImmersive = false
                userInitiatedStop = false
            controller.detachRendererConfiguration(rendererConfiguration)
#endif
        }
        .onDisappear {
            PlaybackFaultLogger.shared.log("view-on-disappear")
#if os(visionOS)
            diagnosticsTask?.cancel()
            diagnosticsTask = nil
#endif
        }
        .onChange(of: scenePhase) { _, newPhase in
            PlaybackFaultLogger.shared.log("scene-phase", fields: ["phase": String(describing: newPhase)])
        }
        .onChange(of: stereoPresentation.stopRequestID) { _, _ in
            triggerUserStop()
        }
#if os(visionOS)
        .glassBackgroundEffect(displayMode: .never)
#endif
    }

    private func triggerUserStop() {
#if os(visionOS)
        guard isPlaying else { return }
        userInitiatedStop = true
        Task {
            if isImmersiveOpen {
                await dismissImmersiveSpace()
                await MainActor.run {
                    isImmersiveOpen = false
                    rendererConfiguration.rendererDebugStatus = "Immersive space dismissed"
                    stereoPresentation.setStatus("Immersive playback closed")
                }
            }
            await MainActor.run {
                onUserStop()
                isPlaying = false
            }
        }
#else
        onUserStop()
        isPlaying = false
#endif
    }

#if os(visionOS)
    private func logPanelMovementProbe(_ origin: CGPoint) {
        let roundedOrigin = CGPoint(x: round(origin.x * 10) / 10, y: round(origin.y * 10) / 10)

        guard let previousOrigin = lastLoggedPanelOrigin else {
            lastLoggedPanelOrigin = roundedOrigin
            PlaybackFaultLogger.shared.log("panel-content-origin-initial", fields: [
                "x": String(format: "%.1f", roundedOrigin.x),
                "y": String(format: "%.1f", roundedOrigin.y),
            ])
            rendererConfiguration.rendererDebugStatus = String(
                format: "Panel probe origin x=%.1f y=%.1f",
                roundedOrigin.x,
                roundedOrigin.y
            )
            return
        }

        let deltaX = roundedOrigin.x - previousOrigin.x
        let deltaY = roundedOrigin.y - previousOrigin.y
        guard abs(deltaX) >= 0.5 || abs(deltaY) >= 0.5 else { return }

        lastLoggedPanelOrigin = roundedOrigin
        PlaybackFaultLogger.shared.log("panel-content-origin-changed", fields: [
            "x": String(format: "%.1f", roundedOrigin.x),
            "y": String(format: "%.1f", roundedOrigin.y),
            "dx": String(format: "%.1f", deltaX),
            "dy": String(format: "%.1f", deltaY),
        ])
        rendererConfiguration.rendererDebugStatus = String(
            format: "Panel probe moved x=%.1f y=%.1f dx=%.1f dy=%.1f",
            roundedOrigin.x,
            roundedOrigin.y,
            deltaX,
            deltaY
        )
    }

    private func requestImmersiveOpen() {
        guard !didRequestImmersive else { return }

        didRequestImmersive = true
        PlaybackFaultLogger.shared.log("immersive-open-requested")
        let immersiveOpenInterval = Self.immersiveSignposter.beginInterval("OpenImmersiveSpace", id: Self.immersiveSignposter.makeSignpostID())
        Self.immersiveSignposter.emitEvent("ImmersiveOpenRequested", id: Self.immersiveSignposter.makeSignpostID())
        Task {
            rendererConfiguration.rendererDebugStatus = "Requesting immersive space open"
            stereoPresentation.setStatus("Opening immersive stereo playback...")
            let result = await openImmersiveSpace(id: "DepthPlayerStereoImmersive")
            Self.immersiveSignposter.endInterval("OpenImmersiveSpace", immersiveOpenInterval)
            switch result {
            case .opened:
                PlaybackFaultLogger.shared.log("immersive-opened")
                Self.immersiveSignposter.emitEvent("ImmersiveOpened", id: Self.immersiveSignposter.makeSignpostID())
                isImmersiveOpen = true
                stereoPresentation.setStatus("Immersive stereo playback opened")
                rendererConfiguration.rendererDebugStatus = "Immersive space opened; awaiting RealityKit video attach"
            case .userCancelled:
                PlaybackFaultLogger.shared.log("immersive-user-cancelled")
                Self.immersiveSignposter.emitEvent("ImmersiveCancelled", id: Self.immersiveSignposter.makeSignpostID())
                isImmersiveOpen = false
                didRequestImmersive = false
                stereoPresentation.setStatus("Immersive playback was cancelled")
                rendererConfiguration.rendererDebugStatus = "Immersive open cancelled by user"
            case .error:
                PlaybackFaultLogger.shared.log("immersive-open-error")
                Self.immersiveSignposter.emitEvent("ImmersiveOpenError", id: Self.immersiveSignposter.makeSignpostID())
                isImmersiveOpen = false
                didRequestImmersive = false
                stereoPresentation.setStatus("Failed to open immersive playback")
                rendererConfiguration.rendererDebugStatus = "Immersive open failed"
            @unknown default:
                PlaybackFaultLogger.shared.log("immersive-open-unknown")
                Self.immersiveSignposter.emitEvent("ImmersiveOpenUnknown", id: Self.immersiveSignposter.makeSignpostID())
                isImmersiveOpen = false
                didRequestImmersive = false
                stereoPresentation.setStatus("Unknown immersive playback state")
                rendererConfiguration.rendererDebugStatus = "Immersive open unknown result"
            }
        }
    }
#endif
}

#Preview {
#if os(visionOS)
    StereoVideoPlayerView(
        hlsURL: URL(string: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8")!,
        isPlaying: .constant(true),
        rendererConfiguration: Video3DConfiguration(),
        autoOpenImmersiveOnAppear: true
    )
#else
    StereoVideoPlayerView(
        hlsURL: URL(string: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8")!,
        isPlaying: .constant(true)
    )
#endif
}
