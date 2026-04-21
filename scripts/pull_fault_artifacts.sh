#!/usr/bin/env bash
set -euo pipefail

DEVICE_ID="${1:-5B4FC2E8-AC48-5A6B-9C16-FB3C213C4796}"
BUNDLE_ID="${2:-com.vision.depth-player}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/device-pulls/fault-artifacts-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR"

echo "Pulling app cache artifacts to: $OUT_DIR/app-caches"
xcrun devicectl device copy from \
  --device "$DEVICE_ID" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --source "Library/Caches" \
  --destination "$OUT_DIR/app-caches" \
  --remove-existing-content true

echo "Pulling system crash logs to: $OUT_DIR/system-crash-logs"
if ! xcrun devicectl device copy from \
  --device "$DEVICE_ID" \
  --domain-type systemCrashLogs \
  --source "." \
  --destination "$OUT_DIR/system-crash-logs" \
  --remove-existing-content true; then
  echo "Warning: system crash log pull failed (continuing)."
fi

echo "Done. Artifacts saved under: $OUT_DIR"

if [[ -x "$ROOT_DIR/scripts/analyze_fault_artifacts.sh" ]]; then
  "$ROOT_DIR/scripts/analyze_fault_artifacts.sh" "$OUT_DIR"
else
  echo "Analyzer script is not executable. Run: chmod +x scripts/analyze_fault_artifacts.sh"
fi
