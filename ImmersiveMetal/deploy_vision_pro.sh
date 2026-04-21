#!/usr/bin/env bash
set -euo pipefail

SCHEME="ImmersiveMetal"
CONFIGURATION="Release"
BUNDLE_ID="com.iansutherland.ImmersiveMetal"
APP_NAME="Video3DConverter.app"
DERIVED_DATA="/tmp/vision-pro-deploy"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

find_connected_vision_pro_udid() {
  xcrun xctrace list devices 2>/dev/null | awk '
    /^== Devices ==/ { in_devices = 1; next }
    /^== Devices Offline ==/ { in_devices = 0 }
    in_devices && /Vision/ && /Pro/ && $0 !~ /Simulator/ {
      line = $0
      sub(/^.*\(/, "", line)
      sub(/\)[[:space:]]*$/, "", line)
      if (line ~ /^[0-9A-Fa-f-]+$/ && length(line) >= 25) {
        print line
        exit
      }
    }
  '
}

DEVICE_UDID="$(find_connected_vision_pro_udid)"
if [[ -z "$DEVICE_UDID" ]]; then
  echo "Error: No connected physical Vision Pro found."
  echo "Tip: Put on headset, enable Developer Mode, and ensure it is connected to this Mac."
  exit 1
fi

echo "Target Vision Pro UDID: $DEVICE_UDID"
echo "Cleaning and building $SCHEME ($CONFIGURATION)..."
rm -rf "$DERIVED_DATA"
cd "$PROJECT_DIR"
xcodebuild \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination 'generic/platform=visionOS' \
  -derivedDataPath "$DERIVED_DATA" \
  -allowProvisioningUpdates \
  clean build >/tmp/vision-pro-deploy-build.log 2>&1

APP_PATH="$DERIVED_DATA/Build/Products/Release-xros/$APP_NAME"
if [[ ! -d "$APP_PATH" ]]; then
  echo "Error: Built app not found at $APP_PATH"
  echo "Build log: /tmp/vision-pro-deploy-build.log"
  exit 1
fi

echo "Uninstalling previous app (if present)..."
xcrun devicectl device uninstall app --device "$DEVICE_UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true

echo "Installing fresh build..."
xcrun devicectl device install app --device "$DEVICE_UDID" "$APP_PATH"

echo "Verifying installed app version..."
xcrun devicectl device info apps --device "$DEVICE_UDID" --filter "bundleIdentifier == '$BUNDLE_ID'"

echo "Done."
