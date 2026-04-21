import SwiftUI
#if os(visionOS)
import CompositorServices
#endif

@main
struct DepthPlayerApp: App {
    @StateObject private var stereoPresentation = StereoPresentationCoordinator()
#if os(visionOS)
    private let rendererConfiguration = Video3DConfiguration()
#endif

    init() {
        AppleDiagnostics.shared.start()
    }

    var body: some Scene {
        WindowGroup {
#if os(visionOS)
            ContentView(rendererConfiguration: rendererConfiguration)
            .environmentObject(stereoPresentation)
        .glassBackgroundEffect(displayMode: .never)
#else
            ContentView()
            .environmentObject(stereoPresentation)
#endif
        }
#if os(visionOS)
    .windowStyle(.plain)
        .windowResizability(.contentSize)
    .defaultSize(width: 460, height: 180)
#endif

#if os(visionOS)
        ImmersiveSpace(id: "DepthPlayerStereoImmersive") {
            CompositorLayer { layerRenderer in
                rendererConfiguration.rendererDebugStatus = "Compositor layer attached"
                StartVideo3DRenderer(layerRenderer, rendererConfiguration)
            }
        }
#endif
    }
}
