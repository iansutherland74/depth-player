import SwiftUI
import CompositorServices

struct MetalLayerConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities,
                           configuration: inout LayerRenderer.Configuration)
    {
        // Prefer layered stereo rendering on device; fall back to dedicated when required.
        let supportsFoveation = capabilities.supportsFoveation
        let supportedLayouts = capabilities.supportedLayouts(options: supportsFoveation ? [.foveationEnabled] : [])
        
        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
        configuration.isFoveationEnabled = supportsFoveation
        configuration.colorFormat = .rgba16Float
    }
}

@main
struct FullyImmersiveMetalApp: App {
    // Shared mutable render settings consumed by both SwiftUI controls and the renderer thread.
    private let rendererConfig: Video3DConfiguration
    @StateObject private var playbackController: VideoPlaybackController

    init() {
        let configuration = Video3DConfiguration()
        rendererConfig = configuration
        _playbackController = StateObject(wrappedValue: VideoPlaybackController(configuration: configuration))
    }

    var body: some Scene {
        WindowGroup {
            ConverterControlsView(rendererConfig, playbackController: playbackController)
                .frame(minWidth: 520, minHeight: 520)
                .glassBackgroundEffect(displayMode: .never)
        }
        .windowStyle(.plain)
        .windowResizability(.automatic)

        ImmersiveSpace(id: "ImmersiveSpace") {
            // CompositorLayer owns the drawable stream that the Metal render thread consumes.
            CompositorLayer(configuration: MetalLayerConfiguration()) { layerRenderer in
                StartVideo3DRenderer(layerRenderer, rendererConfig)
            }
        }
    }
}
