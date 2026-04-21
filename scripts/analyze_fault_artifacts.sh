#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${1:-}"

if [[ -z "$TARGET_DIR" ]]; then
  latest_dir="$(ls -dt "$ROOT_DIR"/device-pulls/fault-artifacts-* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_dir" ]]; then
    echo "No fault artifact directory found under $ROOT_DIR/device-pulls"
    exit 1
  fi
  TARGET_DIR="$latest_dir"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Target directory does not exist: $TARGET_DIR"
  exit 1
fi

echo "Analyzing artifacts in: $TARGET_DIR"

audit_app_log="$(find "$TARGET_DIR" -type f -name 'depthplayer-fault-log.ndjson' | head -n 1 || true)"
metrickit_log="$(find "$TARGET_DIR" -type f -name 'depthplayer-metrickit-diagnostics.ndjson' | head -n 1 || true)"
crash_root="$TARGET_DIR/system-crash-logs"

if [[ -n "$audit_app_log" && -f "$audit_app_log" ]]; then
  echo ""
  echo "=== App Fault Log Summary ==="
  python3 - "$audit_app_log" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime

path = sys.argv[1]
last_by_run = {}
counts_by_run = Counter()
state_counts = Counter()
max_seconds_by_run = {}
latest_ts_by_run = {}


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        run = obj.get("run_id", "(missing)")
        evt = obj.get("event", "(missing)")
        counts_by_run[run] += 1

        seconds = obj.get("seconds")
        if seconds is not None:
            try:
                max_seconds_by_run[run] = max(max_seconds_by_run.get(run, -1), int(float(seconds)))
            except Exception:
                pass

        ts = parse_ts(obj.get("ts"))
        if ts and (run not in latest_ts_by_run or ts > latest_ts_by_run[run]):
            latest_ts_by_run[run] = ts

        if evt == "playback-state":
            state = obj.get("state")
            if state:
                state_counts[state] += 1
        last_by_run[run] = obj

for run, count in counts_by_run.most_common(5):
    last = last_by_run.get(run, {})
    print(
        "run={run} events={events} max_seconds={max_s} last_event={evt} last_state={state} seq={seq}".format(
            run=run,
            events=count,
            max_s=max_seconds_by_run.get(run, "n/a"),
            evt=last.get("event", "(none)"),
            state=last.get("state", "(n/a)"),
            seq=last.get("seq", "(n/a)"),
        )
    )

if latest_ts_by_run:
    print("latest_runs=")
    ordered = sorted(latest_ts_by_run.items(), key=lambda item: item[1], reverse=True)
    for run, ts in ordered[:5]:
        last = last_by_run.get(run, {})
        print(
            "  run={run} ts={ts} max_seconds={max_s} last_event={evt} last_state={state} seq={seq}".format(
                run=run,
                ts=last.get("ts", "(none)"),
                max_s=max_seconds_by_run.get(run, "n/a"),
                evt=last.get("event", "(none)"),
                state=last.get("state", "(n/a)"),
                seq=last.get("seq", "(n/a)"),
            )
        )

if state_counts:
    print("top_playback_states=")
    for state, count in state_counts.most_common(10):
        print(f"  {state}: {count}")
PY
else
    echo "No app fault log found in: $TARGET_DIR"
fi

if [[ -n "$metrickit_log" && -f "$metrickit_log" ]]; then
  echo ""
  echo "=== MetricKit Diagnostic Payloads ==="
  python3 - "$metrickit_log" <<'PY'
import json
import sys

path = sys.argv[1]
count = 0
for line in open(path, "r", encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    count += 1
    source = obj.get("source", "unknown")
    ts = obj.get("ts", "unknown")
    print(f"payload[{count}] ts={ts} source={source}")

if count == 0:
    print("No valid MetricKit payloads parsed")
PY
else
    echo "No MetricKit diagnostics file found in: $TARGET_DIR"
fi

echo ""
echo "=== Crash and Jetsam Classification ==="
python3 - "$crash_root" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
ips_files = sorted(glob.glob(os.path.join(root, "**", "*.ips"), recursive=True), key=os.path.getmtime, reverse=True)

if not ips_files:
    print(f"No .ips files found under {root}")
    sys.exit(0)

seen = 0
for path in ips_files[:30]:
    seen += 1
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            rest = f.read().strip()
        if not first or not rest:
            continue
        metadata = json.loads(first)
        report = json.loads(rest)
    except Exception:
        continue

    bug_type = str(metadata.get("bug_type", ""))
    name = metadata.get("name") or metadata.get("procName") or "(unknown)"
    ts = metadata.get("timestamp") or report.get("captureTime") or "(unknown)"

    if bug_type == "309":
        exc = report.get("exception", {})
        term = report.get("termination", {})
        print(f"[{seen}] CRASH file={os.path.basename(path)}")
        print(f"    proc={name} time={ts}")
        print(f"    exception_type={exc.get('type', '(none)')} signal={exc.get('signal', '(none)')}")
        print(f"    termination_namespace={term.get('namespace', '(none)')} code={term.get('code', '(none)')} indicator={term.get('indicator', '(none)')}")
    elif "jetsam" in os.path.basename(path).lower() or bug_type in {"298", "385"}:
        mem = report.get("memoryStatus", {})
        page_size = mem.get("pageSize") or 0
        killed = None
        for proc in report.get("processes", []):
            if "reason" in proc:
                killed = proc
                break
        print(f"[{seen}] JETSAM file={os.path.basename(path)}")
        print(f"    proc={name} time={ts} page_size={page_size}")
        if killed:
            rpages = int(killed.get("rpages") or 0)
            used_bytes = rpages * int(page_size or 0)
            used_gib = used_bytes / float(1024 ** 3)
            print(f"    victim={killed.get('name', '(unknown)')} reason={killed.get('reason', '(unknown)')} rpages={rpages} approx_gib={used_gib:.3f}")
    else:
        print(f"[{seen}] OTHER bug_type={bug_type} file={os.path.basename(path)} proc={name} time={ts}")
PY

echo ""
echo "Analysis complete."
