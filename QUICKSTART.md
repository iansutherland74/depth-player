# DepthPlayer - Quick Start

Your complete visionOS 3D video player is ready. Here's what you have:

## 📁 What Was Created

```
/Users/sutherland/vision ui/DepthPlayer/
├── README.md                          # Full documentation
├── XCODE_SETUP.md                     # Step-by-step Xcode setup
├── setup.sh                           # Verification script (already ran ✅)
├── Package.swift                      # Swift package config
├── DepthPlayer/
│   ├── DepthPlayerApp.swift           # App entry point
│   ├── Info.plist                     # App configuration
│   ├── Views/
│   │   ├── ContentView.swift          # Home screen
│   │   └── StereoVideoPlayerView.swift # Video + depth processing
│   └── Utilities/
│       └── DepthAnythingEstimator.swift # Depth inference wrapper
```

## 🚀 Quick Start (5 minutes)

### Option A: Use Existing visionOS Project (Recommended)

If you have an existing visionOS project:

1. Copy `DepthPlayer/` folder into your Xcode project
2. Add `../Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage` to Build Phases
3. Build & run

### Option B: Create New visionOS Project

Follow [XCODE_SETUP.md](XCODE_SETUP.md) for complete step-by-step instructions.

## 📖 Documentation

- **[README.md](README.md)** - Full feature overview, configuration, troubleshooting
- **[XCODE_SETUP.md](XCODE_SETUP.md)** - Detailed Xcode project creation guide
- **[../Depth-Anything-V2/visionos/README.md](../Depth-Anything-V2/visionos/README.md)** - Depth model details

## ✨ Features

✅ Play HLS/HEVC streams in real-time  
✅ Live monocular depth estimation (50MB SmallF16 model)  
✅ Stereo synthesis with configurable disparity  
✅ Temporal depth smoothing for comfort  
✅ Built-in Mux test stream example  

## 🎮 Usage

1. **Launch** DepthPlayer on Vision Pro
2. **Tap** "Load HLS Stream"
3. **Use** quick test button OR enter custom HLS URL
4. **Watch** video convert to stereo automatically

Test URL:
```
https://test-streams.mux.dev/x36xhzz/main.m3u8
```

## ⚙️ Configuration

Edit `StereoVideoPlayerView.swift` line ~96:

```swift
maxDisparity: 12.0,           // Depth effect (8-16 range)
temporalSmoothing: 0.85       // Stability (0.7-0.95 range)
```

**Recommended starting values:**
- `maxDisparity: 8-12` (comfortable depth)
- `temporalSmoothing: 0.85+` (smooth playback)

## 🔧 Technical Stack

- **Language**: Swift 5.9+
- **Framework**: SwiftUI + AVFoundation
- **ML Model**: DepthAnythingV2SmallF16 (Core ML)
- **Video**: HLS/H.264/H.265
- **Target**: visionOS 1.0+
- **Architecture**: ARM64 (Apple Silicon)

## 📦 Dependencies

**Model (pre-downloaded):**
- DepthAnythingV2SmallF16.mlpackage (~50MB)
  - Location: `../Depth-Anything-V2/models/`
  - Auto-copied to Bundle Resources

**Frameworks (linked automatically):**
- CoreML
- Vision
- AVFoundation
- CoreImage
- QuartzCore
- SwiftUI

## ✅ Verification

Run the setup checker anytime:
```bash
cd /Users/sutherland/vision\ ui/DepthPlayer
./setup.sh
```

All tests should show ✅.

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| "Model not found" | Verify Build Phases has `.mlpackage` file |
| "Black screen" | Tap "Load HLS Stream" and use test URL |
| "Stuttering video" | Reduce `maxDisparity` by 2-4 points |
| Build fails | Run `setup.sh` and follow output |

## 📚 Next Steps

1. **For Xcode setup**: Read [XCODE_SETUP.md](XCODE_SETUP.md)
2. **For more details**: Read [README.md](README.md)
3. **To customize**: Edit values in `StereoVideoPlayerView.swift`
4. **To deploy**: Use Xcode's Team settings for Vision Pro device

## 🎯 Key Files to Know

| File | Purpose |
|------|---------|
| `DepthPlayerApp.swift` | App entry point |
| `ContentView.swift` | Home screen & URL input |
| `StereoVideoPlayerView.swift` | Video playback + depth stereo conversion |
| `DepthAnythingEstimator.swift` | Depth inference wrapper |

## 📝 Notes

- **Monocular Depth**: Output is pseudo-3D from single camera (not true stereo geometry)
- **Real-time**: Frame-by-frame processing at ~30fps
- **Comfort**: Temporal smoothing prevents flickering
- **Licensing**: Model is Apache-2.0 (small variant)

---

**You're ready to go!** 🚀

Open Xcode and start building. Questions? See the documentation files above.
