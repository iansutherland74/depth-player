#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ID="com.iansutherland.ImmersiveMetal"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "Deploying latest build..."
"$SCRIPT_DIR/deploy_vision_pro.sh"

DEVICE_UDID="$(find_connected_vision_pro_udid)"
if [[ -z "$DEVICE_UDID" ]]; then
  echo "Error: Could not resolve connected Vision Pro UDID for launch step."
  exit 1
fi

echo "Launching app on Vision Pro ($DEVICE_UDID)..."
set +e
LAUNCH_OUTPUT="$(xcrun devicectl device process launch --device "$DEVICE_UDID" "$BUNDLE_ID" 2>&1)"
LAUNCH_EXIT=$?
set -e

echo "$LAUNCH_OUTPUT"

if [[ $LAUNCH_EXIT -eq 0 ]]; then
  echo "Launch succeeded."
  exit 0
fi

if echo "$LAUNCH_OUTPUT" | grep -qi "profile has not been explicitly trusted\|invalid code signature\|inadequate entitlements\|RequestDenied\|Security"; then
  cat <<'EOF'

Launch blocked by device trust/signing policy.
On Vision Pro:
1. Open Settings.
2. Go to Privacy & Security and verify Developer Mode is enabled.
3. Go to General > VPN & Device Management (or Device Management).
4. Trust the Apple Development profile for this app.
5. Relaunch the app.
EOF
  exit 2
fi

echo "Launch failed for another reason. Review output above."
exit $LAUNCH_EXIT
