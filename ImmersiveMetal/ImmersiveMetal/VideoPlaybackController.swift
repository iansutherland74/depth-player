import AVFoundation
import Foundation

@MainActor
final class VideoPlaybackController: ObservableObject {
    @Published var isPlaying = false
    @Published var currentTime = 0.0
    @Published var duration = 0.0
    @Published var videoTitle = "Sample Video"
    @Published var isMuted = false
    @Published var subtitlesEnabled = false
    @Published var hasSubtitles = false

    let player: AVPlayer
    let videoOutput: AVPlayerItemVideoOutput

    private weak var configuration: Video3DConfiguration?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?

    init(configuration: Video3DConfiguration) {
        self.configuration = configuration

        // Request Metal-compatible HDR-capable frames for direct texture upload.
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_64RGBAHalf),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: attrs)
        // Renderer performs presentation itself; disable AVPlayer's default rendering path.
        videoOutput.suppressesPlayerRendering = true

        if let videoURL = Bundle.main.url(forResource: "SampleVideo", withExtension: "mp4")
            ?? Bundle.main.url(forResource: "sample", withExtension: "mp4")
        {
            let item = AVPlayerItem(url: videoURL)
            item.add(videoOutput)
            player = AVPlayer(playerItem: item)
            player.actionAtItemEnd = .none
            configureMediaMetadata(item: item, url: videoURL)
            // Expose frame output to the native renderer.
            configuration.videoOutput = videoOutput

            endObserver = NotificationCenter.default.addObserver(
                forName: .AVPlayerItemDidPlayToEndTime,
                object: item,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.player.seek(to: .zero)
                    if self.isPlaying {
                        self.player.play()
                    }
                }
            }

            player.play()
            isPlaying = true
        } else {
            player = AVPlayer()
            configuration.videoOutput = nil
        }

        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTimeMakeWithSeconds(0.1, preferredTimescale: 600),
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshPlaybackState()
            }
        }

        refreshPlaybackState()
    }

    deinit {
        if let timeObserver {
            player.removeTimeObserver(timeObserver)
        }
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
        }
    }

    func togglePlayback() {
        isPlaying ? pause() : play()
    }

    func play() {
        player.play()
        isPlaying = true
        refreshPlaybackState()
    }

    func pause() {
        player.pause()
        isPlaying = false
        refreshPlaybackState()
    }

    func skip(by delta: Double) {
        seek(to: currentTime + delta)
    }

    func toggleMute() {
        player.isMuted.toggle()
        isMuted = player.isMuted
    }

    func toggleSubtitles() {
        // Subtitle track selection APIs are unavailable on current visionOS SDK.
        subtitlesEnabled = false
        hasSubtitles = false
    }

    func seek(to requestedTime: Double) {
        let clamped = min(max(0.0, requestedTime), max(duration, 0.0))
        let seekTime = CMTimeMakeWithSeconds(clamped, preferredTimescale: 600)
        player.seek(to: seekTime, toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshPlaybackState()
            }
        }
        currentTime = clamped
    }

    private func refreshPlaybackState() {
        let seconds = CMTimeGetSeconds(player.currentTime())
        currentTime = seconds.isFinite && seconds >= 0.0 ? seconds : 0.0

        if let item = player.currentItem {
            let durationSeconds = CMTimeGetSeconds(item.duration)
            duration = durationSeconds.isFinite && durationSeconds >= 0.0 ? durationSeconds : 0.0
        } else {
            duration = 0.0
        }

        isPlaying = player.rate != 0.0
        isMuted = player.isMuted
        refreshSubtitleState()
        // Keep renderer connected only when an item is active.
        configuration?.videoOutput = player.currentItem != nil ? videoOutput : nil
    }

    private func configureMediaMetadata(item: AVPlayerItem, url: URL) {
        // Try embedded title metadata first, then fall back to file name.
        let metadataTitle = item.asset.commonMetadata
            .first(where: { $0.commonKey?.rawValue == "title" })?
            .stringValue?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let fallback = url.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: "_", with: " ")
        videoTitle = (metadataTitle?.isEmpty == false) ? metadataTitle! : fallback
        refreshSubtitleState()
    }

    private func refreshSubtitleState() {
        hasSubtitles = false
        subtitlesEnabled = false
    }
}