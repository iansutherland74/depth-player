# Setting Up DepthPlayer in Xcode

This guide walks you through creating and building the DepthPlayer visionOS app from the source files.

## Prerequisites

- ✅ Xcode 15.4 or later
- ✅ visionOS SDK (included with Xcode 15.2+)
- ✅ Apple Silicon Mac (recommended)
- ✅ All source files present (verified by `setup.sh`)

## Step 1: Create a New visionOS Project

1. Open **Xcode**
2. Click **File → New → Project**
3. Select **visionOS** (not iOS)
4. Choose **App** template
5. Fill in project settings:
   - **Product Name**: `DepthPlayer`
   - **Team**: Your Apple Developer account
   - **Organization Identifier**: `com.vision`
   - **Bundle Identifier**: `com.vision.depth-player`
   - **Interface**: `SwiftUI`
   - **Lifecycle**: `SwiftUI App`
6. Create in a **temporary location** (we'll use source from the cloned files)

## Step 2: Replace Generated Source Files

After Xcode creates the project, replace the auto-generated Swift files:

### Delete These Auto-Generated Files
```
DepthPlayerApp.swift
ContentView.swift
Preview Content/ (folder)
```

### Copy Source Files from Cloned Repository

Copy these files from `/Users/sutherland/vision ui/DepthPlayer/DepthPlayer/` into your Xcode project:

```
DepthPlayer/DepthPlayerApp.swift
DepthPlayer/Views/ContentView.swift
DepthPlayer/Views/StereoVideoPlayerView.swift
DepthPlayer/Utilities/DepthAnythingEstimator.swift
DepthPlayer/Info.plist
```

In Xcode:
1. In the project navigator, select the folder group where you deleted files
2. **File → Add Files to "DepthPlayer"...**
3. Navigate to the cloned source folder
4. Select the files above
5. Check "Copy items if needed"
6. Add to target: **DepthPlayer**

## Step 3: Add the Core ML Model

The model is at: `/Users/sutherland/vision ui/Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage`

### In Xcode:

1. Select your **DepthPlayer** target (in the project editor)
2. Go to **Build Phases** tab
3. Expand **Copy Bundle Resources**
4. Click **+** to add files
5. Navigate to `Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage`
6. Select it and click **Add**

**Xcode will automatically generate** `DepthAnythingV2SmallF16.swift` (the model interface class).

Verify it was added:
- In Xcode, open the Build Report after building
- Search for "DepthAnythingV2SmallF16.mlpackage"
- Should see it copied to Bundle

## Step 4: Verify Framework Linking

Select **DepthPlayer** target → **Build Phases** → **Link Binary With Libraries**

Ensure these frameworks are present:
- ✅ CoreML.framework
- ✅ Vision.framework
- ✅ AVFoundation.framework
- ✅ CoreImage.framework
- ✅ QuartzCore.framework
- ✅ SwiftUI.framework

If any are missing, click **+** and add them.

## Step 5: Configure Deployment

1. Select **DepthPlayer** target
2. **General** tab
3. Set:
   - **Minimum Deployments**: `visionOS 1.0`
   - **Device**: visionOS (all devices)

## Step 6: Build & Test

### Build for visionOS Simulator

```bash
# In terminal at project root
xcodebuild -scheme DepthPlayer -destination 'platform=visionOS Simulator,name=Apple Vision Pro' build
```

Or in Xcode:
1. Select **DepthPlayer** scheme (top left)
2. Select **Apple Vision Pro** (device picker)
3. Press **Cmd+B** (Build)
4. Wait for build to complete

### Run on Simulator

1. Select **DepthPlayer** scheme
2. Select **Apple Vision Pro** simulator
3. Press **Cmd+R** (Run)

### Test with HLS Stream

Once the app launches:
1. Tap **"Load HLS Stream"**
2. Tap **"Mux Test Stream"** for quick test
3. Watch video play in stereo!

## Step 7: Debug Common Issues

### Issue: "Cannot find module 'CoreML'"

**Solution**: Ensure Build Phases → Link Binary With Libraries includes CoreML.framework

### Issue: "Model DepthAnythingV2SmallF16 not found"

**Solution**: 
1. Verify model is in Build Phases → Copy Bundle Resources
2. Run Clean Build Folder (Cmd+Shift+K)
3. Rebuild (Cmd+B)

### Issue: Black screen at startup

**Solution**:
1. App is waiting for HLS URL
2. Tap "Load HLS Stream"
3. Use test URL from quick buttons

### Issue: Crash on model load

**Solution**: Check that DepthAnythingV2SmallF16.mlpackage is exactly at:
```
../Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage
```

## Advanced: Custom HLS Sources

To test with your own HLS URL:

1. Launch DepthPlayer
2. Tap **"Load HLS Stream"**
3. Enter URL in text field (must end in `.m3u8`)
4. Tap **"Load"**

Example working URLs:
- `https://test-streams.mux.dev/x36xhzz/main.m3u8`
- Any valid HLS (.m3u8) endpoint

## Build Artifacts

After successful build:
- **Simulator**: `./DerivedData/DepthPlayer-{hash}/Build/Products/Debug-visionOS/DepthPlayer.app`
- **Device**: Use Xcode's "Run" to deploy to real Vision Pro

## Next Steps

- Adjust depth parameters in `StereoVideoPlayerView.swift`
- Add custom HLS sources
- Deploy to real Vision Pro (requires developer certificate)
- Customize UI/UX for your use case

## References

- [Xcode visionOS Support](https://developer.apple.com/xcode/visionos/)
- [Core ML Integration](https://developer.apple.com/documentation/coreml)
- [AVFoundation HLS](https://developer.apple.com/documentation/avfoundation/media_playback_and_selection)
