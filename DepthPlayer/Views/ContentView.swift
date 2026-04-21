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
                    autoOpenImmersiveOnAppear: isMuxQuickTestSelected,
                    onUserStop: {
                        userRequestedStop = true
                    }
                )
#else
                StereoVideoPlayerView(hlsURL: url, isPlaying: $isPlaying)
#endif
            } else {
                VStack(spacing: 24) {
                    VStack(spacing: 12) {
                        Image(systemName: "video.fill")
                            .font(.system(size: 48))
                            .foregroundColor(.blue)
                        
                        Text("DepthPlayer")
                            .font(.system(size: 32, weight: .bold))
                        
                        Text("2D to 3D HLS Streaming")
                            .font(.system(size: 16))
                            .foregroundColor(.gray)
                    }
                    
                    VStack(spacing: 8) {
                        Text("Powered by Depth Anything V2")
                            .font(.system(size: 14, weight: .semibold))
                        Text("Real-time monocular depth estimation + stereo synthesis")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                    }
                    .padding(16)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    
                    Button(action: { showURLInput = true }) {
                        Label("Load HLS Stream", systemImage: "plus.circle.fill")
                            .frame(maxWidth: .infinity)
                            .padding(12)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Quick Test URLs")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.gray)
                        
                        Button(action: {
                            isMuxQuickTestSelected = true
                            streamURL = URL(string: muxTestURLString)
                        }) {
                            HStack {
                                Text("Mux Test Stream")
                                Spacer()
                                Image(systemName: "arrow.right")
                            }
                            .padding(10)
                            .background(isMuxQuickTestSelected ? Color.green : Color.gray.opacity(0.1))
                            .foregroundColor(isMuxQuickTestSelected ? .white : .blue)
                            .cornerRadius(6)
                        }
                    }
                }
                .padding(32)
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
                        .font(.system(size: 14, weight: .semibold))
                    
                    TextField("https://...", text: $input)
                        .textFieldStyle(.roundedBorder)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                
                HStack(spacing: 12) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(10)
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(6)
                    
                    Button("Load") {
                        if let url = URL(string: input) {
                            onLoad()
                            streamURL = url
                            dismiss()
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(10)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(6)
                    .disabled(input.isEmpty)
                }
                
                Spacer()
            }
            .padding(16)
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
