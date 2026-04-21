import SwiftUI
import AVFoundation
import CoreML

struct ContentView: View {
#if os(visionOS)
    private let muxTestURLString = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
#else
    private let muxTestURLString = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
#endif
#if os(visionOS)
    private let rendererConfiguration: Video3DConfiguration
#endif
#if os(visionOS)
    @EnvironmentObject private var stereoPresentation: StereoPresentationCoordinator
#endif
    @State private var showURLInput = false
    @State private var streamURL: URL?
    @State private var isPlaying = false
    @State private var isMuxQuickTestSelected = false
    @State private var userRequestedStop = false
    @State private var showRendererMetrics = true

#if os(visionOS)
    init(rendererConfiguration: Video3DConfiguration) {
        self.rendererConfiguration = rendererConfiguration
    }
#else
    init() {}
#endif
    
    var body: some View {
        ZStack {
            if let url = streamURL {
#if os(visionOS)
                StereoVideoPlayerView(
                    hlsURL: url,
                    isPlaying: $isPlaying,
                    rendererConfiguration: rendererConfiguration,
                    showRendererMetrics: showRendererMetrics,
                    autoOpenImmersiveOnAppear: isMuxQuickTestSelected,
                    onUserStop: {
                        userRequestedStop = true
                    }
                )
#else
                StereoVideoPlayerView(hlsURL: url, isPlaying: $isPlaying)
#endif
            } else {
                LinearGradient(
                    colors: [
                        Color(red: 0.08, green: 0.12, blue: 0.20),
                        Color(red: 0.03, green: 0.05, blue: 0.10),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                Circle()
                    .fill(Color.cyan.opacity(0.18))
                    .frame(width: 380, height: 380)
                    .blur(radius: 36)
                    .offset(x: -180, y: -220)

                Circle()
                    .fill(Color.blue.opacity(0.22))
                    .frame(width: 320, height: 320)
                    .blur(radius: 32)
                    .offset(x: 180, y: 180)

                VStack(spacing: 18) {
                    VStack(spacing: 12) {
                        Image(systemName: "play.rectangle.fill")
                            .font(.system(size: 40, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(12)
                            .background(Color.white.opacity(0.16), in: RoundedRectangle(cornerRadius: 14, style: .continuous))

                        Text("DepthPlayer")
                            .font(.system(size: 34, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)

                        Text("Spatial 2D to 3D streaming")
                            .font(.system(size: 15, weight: .medium, design: .rounded))
                            .foregroundStyle(.white.opacity(0.82))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 22)
                    .padding(.horizontal, 20)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .stroke(Color.white.opacity(0.22), lineWidth: 1)
                    )

                    Button(action: { showURLInput = true }) {
                        Label("Load HLS Stream", systemImage: "plus.circle.fill")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                            .background(Color.white.opacity(0.2))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Quick Test")
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .foregroundColor(.white.opacity(0.7))

                        Button(action: {
                            isMuxQuickTestSelected = true
                            streamURL = URL(string: muxTestURLString)
                        }) {
                            HStack {
                                Text("RAW - Big Buck Bunny HLS Test Stream")
                                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                                Spacer()
                                Image(systemName: "arrow.right")
                            }
                            .padding(12)
                            .background(isMuxQuickTestSelected ? Color.green.opacity(0.82) : Color.white.opacity(0.12))
                            .foregroundColor(isMuxQuickTestSelected ? .white : .blue)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }

                        Toggle("Show sec / vf / rf / df", isOn: $showRendererMetrics)
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .foregroundColor(.white)
                            .tint(.blue)
                    }
                    .padding(14)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(Color.white.opacity(0.18), lineWidth: 1)
                    )

                    Text("Powered by Depth Anything V2 · AVPlayer")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(.white.opacity(0.66))
                }
                .padding(28)
                .frame(maxWidth: 620)
                .sheet(isPresented: $showURLInput) {
                    URLInputView(streamURL: $streamURL, onLoad: {
                        isMuxQuickTestSelected = false
                    })
                }
            }
        }
        .onChange(of: isPlaying) { _, isPlaying in
            guard !isPlaying else { return }
            guard userRequestedStop else {
                self.isPlaying = true
                return
            }
            streamURL = nil
            isMuxQuickTestSelected = false
            userRequestedStop = false
        }
    }
}

struct URLInputView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var streamURL: URL?
    let onLoad: () -> Void
    @State private var input = ""
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Stream URL")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.9))
                    
                    TextField("https://...", text: $input)
                        .textFieldStyle(.roundedBorder)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                .padding(14)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                
                HStack(spacing: 12) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(10)
                    .background(Color.white.opacity(0.14))
                    .cornerRadius(10)
                    .foregroundColor(.white)
                    
                    Button("Load") {
                        if let url = URL(string: input) {
                            onLoad()
                            streamURL = url
                            dismiss()
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(10)
                    .background(Color.blue.opacity(0.85))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(input.isEmpty)
                }
                
                Spacer()
            }
            .padding(16)
            .background(
                LinearGradient(
                    colors: [Color(red: 0.06, green: 0.09, blue: 0.16), Color(red: 0.03, green: 0.05, blue: 0.10)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .navigationTitle("Enter Stream URL")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

#Preview {
#if os(visionOS)
    ContentView(rendererConfiguration: Video3DConfiguration())
#else
    ContentView()
#endif
}
