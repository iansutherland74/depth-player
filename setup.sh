#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DepthPlayer - visionOS 3D Video Player Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Xcode
if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "❌ Xcode not found. Install from App Store or run:"
    echo "   xcode-select --install"
    exit 1
fi

XCODE_VERSION=$(xcodebuild -version | head -n1)
echo "✅ $XCODE_VERSION"

# Check Swift
SWIFT_VERSION=$(swift --version)
echo "✅ $SWIFT_VERSION"

# Verify model exists
MODEL_PATH="../Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo ""
    echo "   Download the model using:"
    echo "   cd ../Depth-Anything-V2/visionos"
    echo "   ./download_model.sh DepthAnythingV2SmallF16 ../models"
    exit 1
fi
echo "✅ Model found: $MODEL_PATH"

# Verify source files
REQUIRED_FILES=(
    "DepthPlayer/DepthPlayerApp.swift"
    "DepthPlayer/Views/ContentView.swift"
    "DepthPlayer/Views/StereoVideoPlayerView.swift"
    "DepthPlayer/Utilities/DepthAnythingEstimator.swift"
)

echo ""
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$REPO_ROOT/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file"
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next Steps:"
echo "1. Open Xcode"
echo "2. Create a new visionOS App project"
echo "3. Copy the DepthPlayer/ folder into your Xcode project"
echo "4. Add model to Build Phases:"
echo "   ../Depth-Anything-V2/models/DepthAnythingV2SmallF16.mlpackage"
echo "5. Build & run (Cmd+B then Cmd+R)"
echo ""
echo "Or, open in Xcode now:"
echo "   open -a Xcode ."
echo ""
