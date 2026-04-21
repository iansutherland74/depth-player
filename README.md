# DepthPlayer - Live 2D to 3D HLS Video Player for Vision Pro

A native visionOS application that plays HEVC and HLS streams in real-time 3D stereo using DepthAnythingV2SmallF16 for monocular depth estimation.

## Features

- **Live HLS/HEVC Playback**: Supports streaming video from HTTP(S) sources
- **Real-time Depth Estimation**: Uses DepthAnythingV2SmallF16 Core ML model
- **Stereo Synthesis**: Converts 2D video to stereo side-by-side format on-the-fly
- **Vision Pro Optimized**: Native visionOS app with full spatial rendering
- **Temporal Stability**: Depth smoothing for comfortable 3D viewing

## Quick Start

### 1. Generate Xcode Project

```bash
cd /Users/sutherland/vision\ ui/DepthPlayer
xcodebuild -version  # Verify Xcode 15.4+ is installed
```

### 2. Create Xcode Project in Xcode

Since this is a visionOS app, you must build it in Xcode:

1. Open Xcode
2. Create a new visionOS App project
3. Copy these folders into your Xcode project:
   - `DepthPlayer/` (source code)
   - `DepthPlayer/DepthPlayer/` (app bundle)

### 3. Add the Core ML Model

1. In Xcode, select your app target
2. Build Phases → Copy Bundle Resources
3. Add the model:
   ```
   ../Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage
   ```

Xcode will automatically generate `DepthAnythingV2SmallF16.swift` (the model class).

### 4. Build & Run

```bash
# Build for visionOS
xcodebuild -scheme DepthPlayer -destination 'platform=visionOS' build

# Or use Xcode: Cmd+B
```

## Project Structure

```
DepthPlayer/
├── DepthPlayer/
│   ├── DepthPlayerApp.swift           # App entry point
│   ├── Views/
│   │   ├── ContentView.swift          # Home screen & URL input
│   │   └── StereoVideoPlayerView.swift # Video playback + depth conversion
│   └── Utilities/
│       └── DepthAnythingEstimator.swift # Depth inference wrapper
├── Package.swift                       # Swift package metadata
└── README.md                          # This file
```

## Usage

### Play a Test Stream

1. Launch DepthPlayer on Vision Pro (or simulator)
2. Tap "Load HLS Stream"
3. Use the quick test button: **Mux Test Stream**
   - URL: `https://test-streams.mux.dev/x36xhzz/main.m3u8`
4. Watch the video convert to stereo automatically

### Custom HLS URL

1. Tap "Load HLS Stream"
2. Enter your HLS URL (must end in `.m3u8`)
3. Tap "Load"

## Performance Tuning

Edit `StereoVideoPlayerView.swift` to adjust:

```swift
maxDisparity: 12.0        // Depth magnitude (8-16 comfortable range)
temporalSmoothing: 0.85   // Temporal stability (0.7-0.95)
```

### Recommendation: Start with these

- **maxDisparity = 8**: Conservative, very comfortable
- **maxDisparity = 12**: Balanced depth + comfort (default)
- **maxDisparity = 16**: Aggressive 3D effect
- **temporalSmoothing = 0.85**: Good temporal stability

## Important Notes

1. **Model Loading**: The DepthAnythingV2SmallF16 model (~50MB) is loaded once at startup
2. **Depth Format**: Outputs monocular relative depth (not true stereo reconstruction)
3. **Compute**: Uses Neural Engine on Apple Silicon for efficient inference
4. **Memory**: Typical memory usage 200-400MB during playback
5. **Frame Rate**: Targets 30fps; may vary based on video source

## Supported Formats

- **Video Codecs**: H.264, H.265 (HEVC)
- **Streaming**: HLS (.m3u8)
- **Containers**: MPEG-TS, MP4 (via HLS)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model fails to load | Ensure DepthAnythingV2SmallF16.mlpackage is in Bundle Resources |
| Black video | Check HLS URL is reachable; try test URL first |
| Stuttering | Reduce `maxDisparity` or increase `temporalSmoothing` |
| High memory | Model is normal size (~50MB); check available system RAM |

## Build Requirements

- **Xcode**: 15.4 or later
- **visionOS SDK**: 1.0+
- **Swift**: 5.9+
- **Deployment Target**: visionOS 1.0+

## References

- [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- [Apple Core ML](https://developer.apple.com/machine-learning/core-ml/)
- [visionOS Development](https://developer.apple.com/visionos/)
- [AVFoundation Streaming](https://developer.apple.com/documentation/avfoundation)

## License

DepthAnythingV2SmallF16 model is Apache-2.0. See the model repository for full license terms.
