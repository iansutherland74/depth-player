import SwiftUI
import RealityKit
import OSLog

struct ConverterControlsView: View {
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State private var showImmersiveSpace = false
    @State private var hasAutoStartedImmersiveSpace = false
    @State private var panelOffsetX = 0.0
    @State private var panelOffsetY = 0.0
    @State private var panelOffsetZ = -1.0
    @State private var panelScale = 1.0
    @State private var disparityStrength = 6.5
    @State private var stabilityAmount = 0.35
    @State private var colorBoost = 1.40
    @State private var playbackScrubTime = 0.0
    @State private var isScrubbingPlayback = false
    @State private var showSettings = false
    @State private var hasHUDAnchor = false
    @State private var anchorPanelX = 0.0
    @State private var anchorPanelY = 1.0
    @State private var anchorPanelZ = -1.0
    @State private var anchorPanelScale = 1.0
    @State private var anchorHUDX = 0.0
    @State private var anchorHUDY = -24.0

    // Direct bridge object to native renderer settings.
    private let rendererConfiguration: Video3DConfiguration
    private let isEmbeddedHUD: Bool
    private let showHUD: Bool
    private let hudLogger = Logger(subsystem: "com.iansutherland.ImmersiveMetal", category: "HUDPosition")
    private let panelLogger = Logger(subsystem: "com.iansutherland.ImmersiveMetal", category: "VideoPanelPosition")
    @ObservedObject private var playbackController: VideoPlaybackController

    init(_ rendererConfig: Video3DConfiguration,
         playbackController: VideoPlaybackController,
         isEmbeddedHUD: Bool = false,
         showHUD: Bool = true)
    {
        rendererConfiguration = rendererConfig
        self.isEmbeddedHUD = isEmbeddedHUD
        self.showHUD = showHUD
        self.playbackController = playbackController
        _panelOffsetX = State(initialValue: rendererConfig.panelOffsetX)
        _panelOffsetY = State(initialValue: rendererConfig.panelOffsetY)
        _panelOffsetZ = State(initialValue: rendererConfig.panelOffsetZ)
        _panelScale = State(initialValue: rendererConfig.panelScale)
        _disparityStrength = State(initialValue: rendererConfig.disparityStrength)
        _stabilityAmount = State(initialValue: rendererConfig.stabilityAmount)
        _colorBoost = State(initialValue: rendererConfig.colorBoost)
        _playbackScrubTime = State(initialValue: playbackController.currentTime)
    }

    // MARK: - Body

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .bottom) {
                // Invisible hit-target that covers the whole frame to capture taps.
                Color.clear
                    .ignoresSafeArea()
                    .contentShape(Rectangle())
                    .onTapGesture { revealControls() }

                if showHUD {
                    playerControls
                        .frame(width: max(proxy.size.width - 10, 320))
                        .padding(.bottom, max(44, proxy.safeAreaInsets.bottom + 30))
                        .offset(x: hudDynamicOffset(in: proxy.size).width,
                                y: hudDynamicOffset(in: proxy.size).height)
                        .transition(.opacity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
            .onAppear {
                if !hasHUDAnchor {
                    captureHUDAnchor(viewSize: proxy.size)
                }
                logVideoPanelPosition(reason: "appear")
                logHUDPosition(viewSize: proxy.size, reason: "appear")
            }
            .onChange(of: panelOffsetX) { _, _ in
                logVideoPanelPosition(reason: "panelOffsetX")
                logHUDPosition(viewSize: proxy.size, reason: "panelOffsetX")
            }
            .onChange(of: panelOffsetY) { _, _ in
                logVideoPanelPosition(reason: "panelOffsetY")
                logHUDPosition(viewSize: proxy.size, reason: "panelOffsetY")
            }
            .onChange(of: panelOffsetZ) { _, _ in
                logVideoPanelPosition(reason: "panelOffsetZ")
                logHUDPosition(viewSize: proxy.size, reason: "panelOffsetZ")
            }
            .onChange(of: panelScale) { _, _ in
                logVideoPanelPosition(reason: "panelScale")
                logHUDPosition(viewSize: proxy.size, reason: "panelScale")
            }
            .onChange(of: showSettings) { _, _ in
                logHUDPosition(viewSize: proxy.size, reason: "showSettings")
            }
        }
        // ── lifecycle tasks ──────────────────────────────────────────────────
        .task {
            if !isEmbeddedHUD && !hasAutoStartedImmersiveSpace {
                hasAutoStartedImmersiveSpace = true
                rendererConfiguration.panelOffsetX = panelOffsetX
                rendererConfiguration.panelOffsetY = panelOffsetY
                rendererConfiguration.panelOffsetZ = panelOffsetZ
                rendererConfiguration.panelScale   = panelScale
                let result = await openImmersiveSpace(id: "ImmersiveSpace")
                if case .opened = result { showImmersiveSpace = true }
            }
        }
        .task {
            guard showHUD else { return }
            while true {
                try? await Task.sleep(nanoseconds: 16_666_667)
                panelOffsetX = rendererConfiguration.panelOffsetX
                panelOffsetY = rendererConfiguration.panelOffsetY
                panelOffsetZ = rendererConfiguration.panelOffsetZ
                panelScale = rendererConfiguration.panelScale
                if !isScrubbingPlayback {
                    playbackScrubTime = playbackController.currentTime
                }
            }
        }
        // ── config sync ──────────────────────────────────────────────────────
        .onChange(of: showImmersiveSpace) { _, v in
            guard !isEmbeddedHUD else { return }
            Task {
                if v {
                    rendererConfiguration.panelOffsetX = panelOffsetX
                    rendererConfiguration.panelOffsetY = panelOffsetY
                    rendererConfiguration.panelOffsetZ = panelOffsetZ
                    rendererConfiguration.panelScale   = panelScale
                    await openImmersiveSpace(id: "ImmersiveSpace")
                } else {
                    await dismissImmersiveSpace()
                }
            }
        }
        .onChange(of: panelOffsetX)      { _, v in rendererConfiguration.panelOffsetX  = v; revealControls() }
        .onChange(of: panelOffsetY)      { _, v in rendererConfiguration.panelOffsetY  = v; revealControls() }
        .onChange(of: panelOffsetZ)      { _, v in rendererConfiguration.panelOffsetZ  = v; revealControls() }
        .onChange(of: panelScale)        { _, v in rendererConfiguration.panelScale    = v; revealControls() }
        .onChange(of: disparityStrength) { _, v in rendererConfiguration.disparityStrength = v; revealControls() }
        .onChange(of: stabilityAmount)   { _, v in rendererConfiguration.stabilityAmount   = v; revealControls() }
        .onChange(of: colorBoost)        { _, v in rendererConfiguration.colorBoost        = v; revealControls() }
        .onChange(of: playbackController.isPlaying) { _, _ in revealControls() }
        .onChange(of: showSettings) { _, _ in revealControls() }
    }

    // MARK: - Main control bar

    /// Compact HUD card centered near the panel bottom so it reads as an overlay on video.
    private var playerControls: some View {
        VStack(spacing: 10) {
            // ── Settings panel (hidden by default) ──────────────────────────
            if showSettings {
                settingsPanel
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // ── Bottom control card ─────────────────────────────────────────
            VStack(spacing: 12) {
                // Title row
                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("3D Video Converter")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.55))
                            .blackOutline()
                        Text(playbackController.videoTitle)
                            .font(.headline.weight(.semibold))
                            .foregroundStyle(.white)
                            .blackOutline()
                            .lineLimit(1)
                    }
                    Spacer()
                    // Immersive toggle
                    if !isEmbeddedHUD {
                        Button {
                            showImmersiveSpace.toggle()
                        } label: {
                            Label(showImmersiveSpace ? "3D On" : "3D Off",
                                  systemImage: showImmersiveSpace ? "cube.fill" : "cube")
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(.white.opacity(0.12), in: Capsule())
                                .foregroundStyle(.white)
                        }
                        .buttonStyle(.plain)
                    }
                }

                // Scrub bar row
                HStack(spacing: 10) {
                    Text(formatTime(playbackScrubTime))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.75))
                        .blackOutline()
                        .frame(minWidth: 40, alignment: .trailing)

                    Slider(value: $playbackScrubTime,
                           in: 0...max(playbackController.duration, 0.1),
                           onEditingChanged: handleScrub)
                        .accentColor(.white)

                    Text(formatTime(playbackController.duration))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.75))
                        .blackOutline()
                        .frame(minWidth: 40, alignment: .leading)
                }

                // Transport row
                HStack(spacing: 20) {
                    // Mute
                    playerButton(systemName: playbackController.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill") {
                        playbackController.toggleMute()
                        revealControls()
                    }

                    Spacer()

                    // Skip back 10 s
                    playerButton(systemName: "gobackward.10") {
                        playbackController.skip(by: -10)
                        revealControls()
                    }

                    // Play / Pause  (larger)
                    Button {
                        playbackController.togglePlayback()
                        revealControls()
                    } label: {
                        Image(systemName: playbackController.isPlaying ? "pause.fill" : "play.fill")
                            .font(.system(size: 30, weight: .bold))
                            .frame(width: 60, height: 60)
                            .background(.white.opacity(0.18), in: Circle())
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.plain)

                    // Skip forward 10 s
                    playerButton(systemName: "goforward.10") {
                        playbackController.skip(by: 10)
                        revealControls()
                    }

                    Spacer()

                    // Settings gear
                    playerButton(systemName: showSettings ? "gearshape.fill" : "gearshape") {
                        withAnimation(.easeInOut(duration: 0.2)) { showSettings.toggle() }
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 14)
            .padding(.top, 16)
            .padding(.bottom, 16)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(.black.opacity(0.50))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(.white.opacity(0.18), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.42), radius: 24, x: 0, y: 12)
        }
        .padding(.horizontal, 0)
    }

    // MARK: - Settings panel

    private var settingsPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            // ── 3D Depth ─────────────────────────────────────────────────────
            VStack(alignment: .leading, spacing: 8) {
                Text("3D DEPTH")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white.opacity(0.55))
                    .blackOutline()

                HStack(spacing: 10) {
                    Text("3D Strength")
                        .font(.callout)
                        .foregroundStyle(.white)
                        .blackOutline()
                        .frame(width: 110, alignment: .leading)
                    Slider(value: $disparityStrength, in: 0...50, step: 0.1)
                        .accentColor(.white)
                    Text("\(disparityStrength, specifier: "%.1f")")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.75))
                        .blackOutline()
                        .frame(width: 40, alignment: .trailing)
                }

                HStack(spacing: 8) {
                    ForEach([("Comfort", 2.0), ("Default", 6.5), ("Strong", 10.0), ("Max", 50.0)], id: \.0) { label, value in
                        Button(label) {
                            disparityStrength = value
                            revealControls()
                        }
                        .font(.caption.weight(.semibold))
                        .buttonStyle(.bordered)
                    }
                }

                HStack(spacing: 10) {
                    Text("Stability")
                        .font(.callout)
                        .foregroundStyle(.white)
                        .blackOutline()
                        .frame(width: 110, alignment: .leading)
                    Slider(value: $stabilityAmount, in: 0...1, step: 0.01)
                        .accentColor(.white)
                    Text("\(stabilityAmount, specifier: "%.2f")")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.75))
                        .blackOutline()
                        .frame(width: 40, alignment: .trailing)
                }
            }

            Divider().overlay(.white.opacity(0.15))

            // ── Colour ────────────────────────────────────────────────────────
            VStack(alignment: .leading, spacing: 8) {
                Text("COLOUR")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white.opacity(0.55))
                    .blackOutline()

                HStack(spacing: 10) {
                    Text("Color Boost")
                        .font(.callout)
                        .foregroundStyle(.white)
                        .blackOutline()
                        .frame(width: 110, alignment: .leading)
                    Slider(value: $colorBoost, in: 0.80...1.40, step: 0.01)
                        .accentColor(.white)
                    Text("\(colorBoost, specifier: "%.2f")x")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.75))
                        .blackOutline()
                        .frame(width: 50, alignment: .trailing)
                }
            }

            Divider().overlay(.white.opacity(0.15))

            // ── Panel Position ────────────────────────────────────────────────
            VStack(alignment: .leading, spacing: 8) {
                Text("PANEL POSITION")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white.opacity(0.55))
                    .blackOutline()

                settingsRow("Distance",  value: $panelOffsetZ,  range: -4.0 ... -0.6, unit: "m")
                settingsRow("Height",    value: $panelOffsetY,  range: 0.3 ... 2.2,   unit: "m")
                settingsRow("Offset X",  value: $panelOffsetX,  range: -2.5 ... 2.5,  unit: "m")
                settingsRow("Scale",     value: $panelScale,    range: 0.5 ... 2.0,   unit: "×")
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(.black.opacity(0.68))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(.white.opacity(0.16), lineWidth: 1)
        )
    }

    private func settingsRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>, unit: String) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.callout)
                .foregroundStyle(.white)
                .blackOutline()
                .frame(width: 110, alignment: .leading)
            Slider(value: value, in: range)
                .accentColor(.white)
            Text("\(value.wrappedValue, specifier: "%.2f")\(unit)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.75))
                .blackOutline()
                .frame(width: 60, alignment: .trailing)
        }
    }

    // MARK: - Helpers

    private func playerButton(systemName: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.title3.weight(.semibold))
                .frame(width: 44, height: 44)
                .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
    }

    private func hudDynamicOffset(in size: CGSize) -> CGSize {
        // Lock HUD to an anchor and then move by panel transform deltas.
        let deltaX = CGFloat(panelOffsetX - anchorPanelX) * size.width * 0.10
        let deltaYFromPanel = -CGFloat(panelOffsetY - anchorPanelY) * size.height * 0.06
        let deltaYFromDepth = CGFloat(panelOffsetZ - anchorPanelZ) * 46.0
        let deltaYFromScale = -CGFloat(panelScale - anchorPanelScale) * 26.0
        let rawX = CGFloat(anchorHUDX) + deltaX
        let rawY = CGFloat(anchorHUDY) + deltaYFromPanel + deltaYFromDepth + deltaYFromScale
        let clampedX = min(max(rawX, -160.0), 160.0)
        let clampedY = min(max(rawY, -140.0), -24.0)
        return CGSize(width: clampedX, height: clampedY)
    }

    private func captureHUDAnchor(viewSize: CGSize) {
        anchorPanelX = panelOffsetX
        anchorPanelY = panelOffsetY
        anchorPanelZ = panelOffsetZ
        anchorPanelScale = panelScale
        anchorHUDX = 0.0
        anchorHUDY = -24.0
        hasHUDAnchor = true

        let lockMessage = String(
            format: "LOCK anchor panel=(x=%.3f y=%.3f z=%.3f scale=%.3f) hud=(x=%.1f y=%.1f) view=(%.1f,%.1f)",
            anchorPanelX,
            anchorPanelY,
            anchorPanelZ,
            anchorPanelScale,
            anchorHUDX,
            anchorHUDY,
            viewSize.width,
            viewSize.height
        )
        panelLogger.notice("\(lockMessage, privacy: .public)")
    }

    private func logHUDPosition(viewSize: CGSize, reason: String) {
        guard showHUD else { return }
        let hudOffset = hudDynamicOffset(in: viewSize)
        let message = String(
            format: "reason=%@ view=(%.1f,%.1f) panel=(x=%.3f y=%.3f z=%.3f scale=%.3f) hudOffset=(x=%.1f y=%.1f)",
            reason,
            viewSize.width,
            viewSize.height,
            panelOffsetX,
            panelOffsetY,
            panelOffsetZ,
            panelScale,
            hudOffset.width,
            hudOffset.height
        )
        hudLogger.notice("\(message, privacy: .public)")
    }

    private func logVideoPanelPosition(reason: String) {
        let message = String(
            format: "reason=%@ panel=(x=%.3f y=%.3f z=%.3f scale=%.3f)",
            reason,
            panelOffsetX,
            panelOffsetY,
            panelOffsetZ,
            panelScale
        )
        panelLogger.notice("\(message, privacy: .public)")
    }

    private func revealControls() {
        // HUD is pinned visible; keep this helper for tap/action hooks.
    }

    private func handleScrub(_ isEditing: Bool) {
        isScrubbingPlayback = isEditing
        revealControls()
        if !isEditing {
            let clamped = min(max(0, playbackScrubTime), max(playbackController.duration, 0))
            playbackScrubTime = clamped
            playbackController.seek(to: clamped)
        }
    }

    private func formatTime(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds > 0 else { return "0:00" }
        let s = Int(seconds)
        return s >= 3600
            ? String(format: "%d:%02d:%02d", s / 3600, (s % 3600) / 60, s % 60)
            : String(format: "%d:%02d", s / 60, s % 60)
    }
}

private extension View {
    /// Four-direction black shadow to simulate a text border.
    func blackOutline() -> some View {
        self
            .shadow(color: .black.opacity(0.9), radius: 0, x: -1, y:  0)
            .shadow(color: .black.opacity(0.9), radius: 0, x:  1, y:  0)
            .shadow(color: .black.opacity(0.9), radius: 0, x:  0, y: -1)
            .shadow(color: .black.opacity(0.9), radius: 0, x:  0, y:  1)
    }
}
