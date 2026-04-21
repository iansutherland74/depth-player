import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import QuartzCore
import CoreMedia
import OSLog
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

    func attach(renderer: AVSampleBufferVideoRenderer) {
        self.renderer = renderer
        immersiveStatus = "Stereo renderer attached"
    }

    func detach() {
        renderer = nil
        immersiveStatus = "Stereo renderer detached"
    }

    func setStatus(_ status: String) {
        immersiveStatus = status
    }
}

final class StereoSampleBufferBridge: @unchecked Sendable {
    let renderer = AVSampleBufferVideoRenderer()

    private var cachedFormatDescription: CMFormatDescription?
    private var cachedDimensions: CMVideoDimensions?

    func enqueue(pixelBuffer: CVPixelBuffer, at presentationTime: CMTime, duration: CMTime = .invalid) throws {
        guard renderer.isReadyForMoreMediaData else { return }
        let formatDescription = try getFormatDescription(for: pixelBuffer)
        let timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: presentationTime, decodeTimeStamp: .invalid)
        let sampleBuffer = try CMSampleBuffer(
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: timing
        )
        renderer.enqueue(sampleBuffer)
    }

    func flush() {
        renderer.flush()
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
    private let ciContext = CIContext()
    private let maxDisparity: CGFloat
    private let temporalSmoothing: Float
    private var depthEMA: MLMultiArray?
    private var stereoPixelBufferPool: CVPixelBufferPool?
    private var stereoPoolWidth: Int = 0
    private var stereoPoolHeight: Int = 0
    private var depthPixelBufferPool: CVPixelBufferPool?
    private var depthPoolWidth: Int = 0
    private var depthPoolHeight: Int = 0

    init(model: MLModel, maxDisparity: CGFloat, temporalSmoothing: Float) throws {
        self.estimator = try DepthAnythingEstimator(model: model)
        self.maxDisparity = maxDisparity
        self.temporalSmoothing = min(max(temporalSmoothing, 0), 0.99)
    }

    func process(pixelBuffer: CVPixelBuffer) throws -> StereoProcessedFrame {
        let depth = try estimator.predictDepth(pixelBuffer: pixelBuffer)
        let smoothedDepth = try smoothDepth(depth)
        return try makeStereoSBS(pixelBuffer: pixelBuffer, depth: smoothedDepth)
    }

    private func smoothDepth(_ depth: MLMultiArray) throws -> MLMultiArray {
        guard let prev = depthEMA else {
            depthEMA = depth
            return depth
        }

        guard prev.count == depth.count else {
            depthEMA = depth
            return depth
        }

        let out = try MLMultiArray(shape: depth.shape, dataType: .float32)
        let alpha = temporalSmoothing

        for i in 0..<depth.count {
            let d = depth[i].floatValue
            let p = prev[i].floatValue
            out[i] = NSNumber(value: alpha * p + (1.0 - alpha) * d)
        }

        depthEMA = out
        return out
    }

    private func makeStereoSBS(pixelBuffer: CVPixelBuffer, depth: MLMultiArray) throws -> StereoProcessedFrame {
        let source = CIImage(cvPixelBuffer: pixelBuffer)
        let width = source.extent.width
        let height = source.extent.height

        let depthImage = try makeNormalizedDepthImage(from: depth)
            .transformed(by: CGAffineTransform(scaleX: width / depthWidth(depth), y: height / depthHeight(depth)))
            .cropped(to: source.extent)

        let leftOffset = maxDisparity * 0.25
        let rightOffset = -maxDisparity * 0.25

        let left = source
            .applyingFilter("CIDisplacementDistortion", parameters: [
                "inputDisplacementImage": depthImage,
                kCIInputScaleKey: leftOffset,
            ])
            .cropped(to: source.extent)

        let right = source
            .applyingFilter("CIDisplacementDistortion", parameters: [
                "inputDisplacementImage": depthImage,
                kCIInputScaleKey: rightOffset,
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

        ciContext.render(composed, to: stereoPixelBuffer)
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
final class StereoVideoPlayerController: NSObject, ObservableObject, AVPlayerItemOutputPullDelegate {
    @Published var isRunning = false
    @Published var lastError: String?
    @Published var loadingMessage: String?
    @Published var playbackDebugState: String = "idle"
    @Published var playbackSeconds: Double = 0
    
    let player: AVPlayer
    
    private let processor: StereoFrameProcessor
    private let stereoBridge = StereoSampleBufferBridge()
    private let processingVideoOutput: AVPlayerItemVideoOutput
    private let rendererVideoOutput: AVPlayerItemVideoOutput
    private let outputDelegateQueue = DispatchQueue(label: "com.vision.depthplayer.video-output", qos: .userInitiated)
    private let processingQueue = DispatchQueue(label: "com.vision.depthplayer.processing", qos: .userInitiated)
    
    private var displayLink: CADisplayLink?
    private var fallbackTimer: Timer?
    private var isProcessingFrame = false
    private var pendingFrame: PendingFrame?
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
    private var mediaDataChangeRequested = false
    private var lastMediaDataChangeRequestHostTime = 0.0
    private var mediaDataChangeRequestCount: UInt64 = 0
    private var mediaDataWillChangeCount: UInt64 = 0
    private var outputRebindCount: UInt64 = 0
    private var lastOutputRebindHostTime = 0.0
    private var lastLoopRestartHostTime = 0.0
    private var lastDetailedDiagnosticsSecond = -1
    private var hasProducedFirstFrame = false
    private var noFrameIntervalState: OSSignpostIntervalState?
    private var startupIntervalState: OSSignpostIntervalState?
    private var lastFrameProcessHostTime = 0.0
    private let signposter = OSSignposter(subsystem: "com.vision.depth-player", category: "playback-pipeline")
    private let baseFrameProcessInterval: Double = 1.0 / 12.0
    private var currentFrameProcessInterval: Double = 1.0 / 12.0
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
        
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        self.processingVideoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: attrs)
        self.processingVideoOutput.suppressesPlayerRendering = false
        // Share a single output object between Swift processing and compositor renderer
        // to avoid duplicate output queues and decoder-side buffering growth.
        self.rendererVideoOutput = self.processingVideoOutput
        
        super.init()

        self.processingVideoOutput.setDelegate(self, queue: outputDelegateQueue)
        
        player.actionAtItemEnd = .none
        player.automaticallyWaitsToMinimizeStalling = false
        configureObservers()
        
    }
    
    isolated deinit {
        displayLink?.invalidate()
        fallbackTimer?.invalidate()
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
        if let observedItem {
            observedItem.remove(processingVideoOutput)
            if rendererVideoOutput !== processingVideoOutput {
                observedItem.remove(rendererVideoOutput)
            }
        }
        NotificationCenter.default.removeObserver(self)
    }
    
    func start() {
        guard !isRunning else { return }
        isRunning = true
        loadingMessage = "Buffering stream..."
        lastError = nil
        playbackSeconds = 0
        hasProducedFirstFrame = false
        lastDeliveredFrameHostTime = CACurrentMediaTime()
        lastAutoResumeAttemptHostTime = 0
        lastObservedItemTimeSeconds = -1
        unchangedItemTimeTicks = 0
        setPlaybackDebugState("started")
        PlaybackFaultLogger.shared.log("controller-start")
        startupIntervalState = signposter.beginInterval("StartupToFirstFrame", id: signposter.makeSignpostID())
        signposter.emitEvent("ControllerStart", id: signposter.makeSignpostID())
        
        player.play()
        
#if os(visionOS)
        let link = CADisplayLink(target: self, selector: #selector(step))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 24, maximum: 60, preferred: 30)
        link.add(to: .main, forMode: .common)
        displayLink = link
#else
        fallbackTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.step()
            }
        }
#endif
    }
    
    func stop() {
        isRunning = false
        displayLink?.invalidate()
        displayLink = nil
        fallbackTimer?.invalidate()
        fallbackTimer = nil
        pendingFrame = nil
        isProcessingFrame = false
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
        configuration.videoOutput = rendererVideoOutput
        configuration.rendererDebugStatus = "Player attached dedicated renderer AVPlayerItemVideoOutput"
    }

    func detachRendererConfiguration(_ configuration: Video3DConfiguration) {
        if configuration.videoOutput === rendererVideoOutput {
            configuration.videoOutput = nil
        }
        configuration.rendererDebugStatus = "Player detached AVPlayerItemVideoOutput"
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
        let hostTime = CACurrentMediaTime()
        let itemTime = processingVideoOutput.itemTime(forHostTime: hostTime)
        let elapsed = CMTimeGetSeconds(itemTime)
        if elapsed.isFinite && elapsed >= 0 {
            playbackSeconds = elapsed
        }
        emitDiagnosticsIfNeeded(hostTime: hostTime, itemTime: itemTime)
        let itemSeconds = CMTimeGetSeconds(itemTime)
        if itemSeconds.isFinite && itemSeconds >= 0 {
            if abs(itemSeconds - lastObservedItemTimeSeconds) < 0.0001 {
                unchangedItemTimeTicks += 1
            } else {
                unchangedItemTimeTicks = 0
                lastObservedItemTimeSeconds = itemSeconds
            }
        }

        if let item = player.currentItem {
            let durationSeconds = CMTimeGetSeconds(item.duration)
            if durationSeconds.isFinite, durationSeconds > 0,
               itemSeconds.isFinite, itemSeconds >= max(0, durationSeconds - 0.20) {
                maybeRestartEndedItem(reason: "duration-boundary")
                return
            }
        }
        
          guard processingVideoOutput.hasNewPixelBuffer(forItemTime: itemTime),
              let pixelBuffer = processingVideoOutput.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: nil) else {
            if isRunning {
                let now = CACurrentMediaTime()
                if !mediaDataChangeRequested {
                    if now - lastMediaDataChangeRequestHostTime > 0.5 {
                        mediaDataChangeRequested = true
                        lastMediaDataChangeRequestHostTime = now
                        mediaDataChangeRequestCount &+= 1
                        processingVideoOutput.requestNotificationOfMediaDataChange(withAdvanceInterval: 0.03)
                        setPlaybackDebugState("request-media-data-change")
                        if mediaDataChangeRequestCount % 25 == 0 {
                            PlaybackFaultLogger.shared.log("media-data-change-request-count", fields: [
                                "count": String(mediaDataChangeRequestCount),
                                "seconds": String(Int(playbackSeconds)),
                            ])
                            logPlaybackSnapshot(context: "media-change-request-count")
                        }
                    } else {
                        setPlaybackDebugState("request-media-data-change-throttled")
                    }
                }
                if noFrameIntervalState == nil {
                    noFrameIntervalState = signposter.beginInterval("NoFrameGap", id: signposter.makeSignpostID())
                }
                if unchangedItemTimeTicks > 90 {
                    setPlaybackDebugState("item-time-not-advancing")
                    maybeRebindVideoOutput(context: "item-time-not-advancing")
                    if unchangedItemTimeTicks % 180 == 0 {
                        logPlaybackSnapshot(context: "item-time-not-advancing")
                    }
                }
                if now - lastDeliveredFrameHostTime > 6.0 {
                    maybeRebindVideoOutput(context: "no-frames-timeout")
                    recoverPlayback(reason: "no-frames")
                }
            }
            return
        }

        lastDeliveredFrameHostTime = CACurrentMediaTime()
        mediaDataChangeRequested = false
        if let noFrameIntervalState {
            signposter.endInterval("NoFrameGap", noFrameIntervalState)
            self.noFrameIntervalState = nil
        }
        if player.timeControlStatus == .playing {
            setPlaybackDebugState("streaming-frames")
        }

        if hostTime - lastFrameProcessHostTime < currentFrameProcessInterval {
            return
        }
        lastFrameProcessHostTime = hostTime

        if isProcessingFrame {
            pendingFrame = PendingFrame(pixelBuffer: pixelBuffer, itemTime: itemTime)
            return
        }

        processFrame(pixelBuffer, itemTime: itemTime)
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

        if let observedItem {
            observedItem.remove(processingVideoOutput)
            if rendererVideoOutput !== processingVideoOutput {
                observedItem.remove(rendererVideoOutput)
            }
        }

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
        item.preferredForwardBufferDuration = 1.0
        item.canUseNetworkResourcesForLiveStreamingWhilePaused = false

        item.add(processingVideoOutput)
        if rendererVideoOutput !== processingVideoOutput {
            item.add(rendererVideoOutput)
        }
        mediaDataChangeRequested = false
        lastMediaDataChangeRequestHostTime = 0
        mediaDataChangeRequestCount = 0
        processingVideoOutput.requestNotificationOfMediaDataChange(withAdvanceInterval: 0.03)

        itemEndObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            self?.loopIfNeeded()
        }

        itemStallObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemPlaybackStalled,
            object: item,
            queue: .main
        ) { [weak self] _ in
            self?.handlePlaybackStalled()
        }

        itemFailedToEndObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemFailedToPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            self?.handleFailedToPlayToEnd()
        }

        itemErrorLogObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemNewErrorLogEntry,
            object: item,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            let message = item.errorLog()?.events.last?.errorComment ?? "unknown"
            self.setPlaybackDebugState("error-log: \(message)")
        }

        itemAccessLogObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemNewAccessLogEntry,
            object: item,
            queue: .main
        ) { [weak self] _ in
            self?.setPlaybackDebugState("access-log-update")
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

        let now = CACurrentMediaTime()
        guard now - lastAutoResumeAttemptHostTime >= 1.5 else { return }
        lastAutoResumeAttemptHostTime = now

        if player.rate == 0 || player.timeControlStatus != .playing {
            loadingMessage = "Recovering playback..."
            setPlaybackDebugState("recovering-\(reason)")
            PlaybackFaultLogger.shared.log("recover-playback", fields: ["reason": reason])
            logPlaybackSnapshot(context: "recover-\(reason)")
            signposter.emitEvent("RecoverPlayback", id: signposter.makeSignpostID())
            player.play()
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
        let stereoBridge = self.stereoBridge

        processingQueue.async { [weak self] in
            let result = autoreleasepool(invoking: {
                Result { try processor.process(pixelBuffer: pixelBuffer) }
            })

            Task { @MainActor [weak self] in
                guard let self else { return }

                self.isProcessingFrame = false

                switch result {
                case let .success(frame):
                    self.lastError = nil
                    self.loadingMessage = nil
                    self.hasProducedFirstFrame = true
                    self.setPlaybackDebugState("frame-processed")
                    if let startupIntervalState = self.startupIntervalState {
                        self.signposter.endInterval("StartupToFirstFrame", startupIntervalState)
                        self.startupIntervalState = nil
                        PlaybackFaultLogger.shared.log("startup-to-first-frame-finished", fields: [
                            "seconds": String(Int(self.playbackSeconds)),
                        ])
                    }
                    try? stereoBridge.enqueue(pixelBuffer: frame.stereoPixelBuffer, at: itemTime)
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

    nonisolated func outputMediaDataWillChange(_ sender: AVPlayerItemOutput) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.mediaDataChangeRequested = false
            self.mediaDataWillChangeCount &+= 1
            self.lastDeliveredFrameHostTime = CACurrentMediaTime()
            self.setPlaybackDebugState("media-data-will-change")
            if self.mediaDataWillChangeCount % 25 == 0 {
                PlaybackFaultLogger.shared.log("media-data-will-change-count", fields: [
                    "count": String(self.mediaDataWillChangeCount),
                    "seconds": String(Int(self.playbackSeconds)),
                ])
                self.logPlaybackSnapshot(context: "media-data-will-change-count")
            }
            self.signposter.emitEvent("MediaDataWillChange", id: self.signposter.makeSignpostID())
        }
    }

    nonisolated func outputSequenceWasFlushed(_ output: AVPlayerItemOutput) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.mediaDataChangeRequested = false
            self.setPlaybackDebugState("output-sequence-flushed")
            self.signposter.emitEvent("OutputSequenceFlushed", id: self.signposter.makeSignpostID())
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
        let hostNow = CACurrentMediaTime()
        let outputTime = processingVideoOutput.itemTime(forHostTime: hostNow).seconds
        let hasNewBuffer = processingVideoOutput.hasNewPixelBuffer(forItemTime: processingVideoOutput.itemTime(forHostTime: hostNow))

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
            "output_s": String(format: "%.2f", outputTime.isFinite ? outputTime : -1),
            "buffer_empty": item.isPlaybackBufferEmpty ? "true" : "false",
            "likely_to_keep_up": item.isPlaybackLikelyToKeepUp ? "true" : "false",
            "has_new_buffer": hasNewBuffer ? "true" : "false",
            "loaded_ranges": loadedRanges.isEmpty ? "none" : loadedRanges,
            "access": accessSummary,
            "request_count": String(mediaDataChangeRequestCount),
            "will_change_count": String(mediaDataWillChangeCount),
            "rebind_count": String(outputRebindCount),
        ])
    }

    private func maybeRebindVideoOutput(context: String) {
        guard let item = player.currentItem else { return }
        guard player.timeControlStatus == .playing else { return }
        guard item.isPlaybackLikelyToKeepUp else { return }
        guard !item.isPlaybackBufferEmpty else { return }

        let now = CACurrentMediaTime()
        guard now - lastOutputRebindHostTime > 4.0 else { return }
        lastOutputRebindHostTime = now
        outputRebindCount &+= 1

        item.remove(processingVideoOutput)
        item.add(processingVideoOutput)
        mediaDataChangeRequested = false
        processingVideoOutput.requestNotificationOfMediaDataChange(withAdvanceInterval: 0.03)

        PlaybackFaultLogger.shared.log("video-output-rebound", fields: [
            "context": context,
            "rebind_count": String(outputRebindCount),
            "seconds": String(Int(playbackSeconds)),
        ])
        setPlaybackDebugState("video-output-rebound")
        logPlaybackSnapshot(context: "video-output-rebound")
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
            guard let self else { return }
            self.lastDeliveredFrameHostTime = CACurrentMediaTime()
            self.lastObservedItemTimeSeconds = -1
            self.unchangedItemTimeTicks = 0
            self.mediaDataChangeRequested = false
            self.processingVideoOutput.requestNotificationOfMediaDataChange(withAdvanceInterval: 0.03)
            self.player.play()
            self.setPlaybackDebugState("loop-restarted")
        }
    }

    private func emitDiagnosticsIfNeeded(hostTime: CFTimeInterval, itemTime: CMTime) {
        let currentSecond = Int(playbackSeconds)
        if currentSecond < 0 || currentSecond == lastDetailedDiagnosticsSecond {
            return
        }

        let isDeepWindow = (28...40).contains(currentSecond)
        if !isDeepWindow && currentSecond % 15 != 0 {
            return
        }
        lastDetailedDiagnosticsSecond = currentSecond

        let outputTime = processingVideoOutput.itemTime(forHostTime: hostTime)
        let outputSeconds = CMTimeGetSeconds(outputTime)
        let hasNewBuffer = processingVideoOutput.hasNewPixelBuffer(forItemTime: itemTime)

        let memory = currentProcessMemoryStatsMB()
        var fields: [String: String] = [
            "seconds": String(currentSecond),
            "window": isDeepWindow ? "deep" : "periodic",
            "time_control": String(player.timeControlStatus.rawValue),
            "rate": String(format: "%.2f", player.rate),
            "item_s": String(format: "%.3f", CMTimeGetSeconds(itemTime)),
            "output_s": String(format: "%.3f", outputSeconds.isFinite ? outputSeconds : -1),
            "has_new_buffer": hasNewBuffer ? "true" : "false",
            "is_processing": isProcessingFrame ? "true" : "false",
            "pending_frame": pendingFrame == nil ? "false" : "true",
            "unchanged_ticks": String(unchangedItemTimeTicks),
            "request_count": String(mediaDataChangeRequestCount),
            "will_change_count": String(mediaDataWillChangeCount),
            "rebind_count": String(outputRebindCount),
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

        PlaybackFaultLogger.shared.log("pipeline-heartbeat", fields: fields)
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
        }
        .onDisappear {
            PlaybackFaultLogger.shared.log("immersive-view-on-disappear")
            stereoPresentation.setStatus("Immersive view disappeared")
        }
    }
}
#endif

struct StereoVideoPlayerView: View {
    private static let immersiveSignposter = OSSignposter(subsystem: "com.vision.depth-player", category: "immersive")
    @StateObject private var controller: StereoVideoPlayerController
    @Binding var isPlaying: Bool
    private let onUserStop: () -> Void
#if os(visionOS)
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    private let rendererConfiguration: Video3DConfiguration
    private let autoOpenImmersiveOnAppear: Bool
    @State private var rendererDiagnostics: String = "Renderer diagnostics pending..."
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
                maxDisparity: 12.0,
                temporalSmoothing: 0.85
            )
            _controller = StateObject(wrappedValue: playerController)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
        
        self._isPlaying = isPlaying
        self.onUserStop = onUserStop
        self.rendererConfiguration = rendererConfiguration
        self.autoOpenImmersiveOnAppear = autoOpenImmersiveOnAppear
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
                maxDisparity: 12.0,
                temporalSmoothing: 0.85
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
            Color.black.ignoresSafeArea()

            if let loadingMessage = controller.loadingMessage {
                ProgressView(loadingMessage)
                    .tint(.blue)
            } else {
                Text(stereoPresentation.immersiveStatus)
                    .foregroundColor(.gray)
            }

            VStack(spacing: 10) {
                Text("Stereo playback is presented in immersive space")
                    .font(.system(size: 15, weight: .semibold))
                Text("This window is only the control surface. The headset should no longer show a side-by-side preview here.")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
#if os(visionOS)
                if !didRequestImmersive {
                    Button("Enter Immersive Playback") {
                        requestImmersiveOpen()
                    }
                    .buttonStyle(.borderedProminent)
                }
#endif
#if os(visionOS)
                Text(rendererDiagnostics)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
#endif
            }
            .padding(18)
            .background(Color.black.opacity(0.55))
            .cornerRadius(10)
            
            if let error = controller.lastError {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.yellow)
                    
                    Text("Error")
                        .font(.system(size: 14, weight: .semibold))
                    
                    Text(error)
                        .font(.system(size: 12))
                        .multilineTextAlignment(.center)
                }
                .padding(16)
                .background(Color.black.opacity(0.8))
                .cornerRadius(8)
            }
            
            VStack {
                HStack {
                    Button(action: {
#if os(visionOS)
                        userInitiatedStop = true
#endif
                        onUserStop()
                        isPlaying = false
                    }) {
                        Label("Back", systemImage: "chevron.left")
                            .padding(8)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(6)
                    }
                    Spacer()
                }
                .padding(16)
                
                Spacer()
            }
        }
        .onAppear {
            PlaybackFaultLogger.shared.log("view-on-appear", fields: [
                "autoOpenImmersiveOnAppear": autoOpenImmersiveOnAppear ? "true" : "false",
            ])
            controller.attachStereoPresentation(stereoPresentation)
            controller.attachRendererConfiguration(rendererConfiguration)
            isPlaying = true
            controller.start()
            rendererDiagnostics = "\(rendererConfiguration.rendererDebugStatus) | sec=\(Int(controller.playbackSeconds)) vf=\(rendererConfiguration.receivedVideoFrameCount) rf=\(rendererConfiguration.renderedFrameCount) df=\(rendererConfiguration.depthFrameCount) | pb=\(controller.playbackDebugState)"
#if os(visionOS)
            rendererConfiguration.rendererDebugStatus = "Ready to open immersive on user action"
            userInitiatedStop = false
            diagnosticsTask?.cancel()
            diagnosticsTask = Task {
                while !Task.isCancelled {
                    await MainActor.run {
                        rendererDiagnostics = "\(rendererConfiguration.rendererDebugStatus) | sec=\(Int(controller.playbackSeconds)) vf=\(rendererConfiguration.receivedVideoFrameCount) rf=\(rendererConfiguration.renderedFrameCount) df=\(rendererConfiguration.depthFrameCount) | pb=\(controller.playbackDebugState)"
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
    }

#if os(visionOS)
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
                rendererConfiguration.rendererDebugStatus = "Immersive space opened; awaiting compositor start"
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
