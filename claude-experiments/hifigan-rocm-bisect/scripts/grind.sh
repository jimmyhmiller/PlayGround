#!/usr/bin/env bash
# Run real_capture.py over many fresh processes until we hit an anomaly.
#
# Per CHATTERBOX_DEBUG.md "anomaly rate is process-dependent": some processes
# are high-rate (25-42% per chunk), some are 0%. Single 5-trial process gives
# 0 hits on low-rate processes (which is most of them). Solution: many short
# fresh processes; stop on first anomaly.
#
# Each process: 3 trials of chunk19 with the env that produced anomalies in
# the historical record. Fresh process = fresh chatterbox load = fresh MIOpen.
#
# Usage: scripts/grind.sh [max_processes] [trials_per_process]

set -uo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"

MAX_PROCESSES="${1:-50}"
TRIALS="${2:-3}"

VENV_PY="$HOME/.cache/paper-audiobooks/venvs/chatterbox/bin/python"
LOG_DIR="$PROJECT/grind_logs"
mkdir -p "$LOG_DIR" "$PROJECT/captures/grind"

# Match env of the historical anomaly-producing scripts (eager_attn_test.py etc.)
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HIFIGAN_BISECT_OUT="$PROJECT/captures/grind"

echo "[grind] starting up to $MAX_PROCESSES processes × $TRIALS trials, stop on first anomaly"
echo "[grind] env: PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF"
echo "[grind] log dir: $LOG_DIR"
echo "[grind] capture dir: $HIFIGAN_BISECT_OUT"

START=$(date +%s)
for i in $(seq 1 "$MAX_PROCESSES"); do
    LOG="$LOG_DIR/proc_$(printf '%03d' "$i").log"
    BEFORE=$(ls "$HIFIGAN_BISECT_OUT" 2>/dev/null | wc -l)

    PROC_START=$(date +%s)
    echo "[grind] process $i / $MAX_PROCESSES at $(date +%H:%M:%S) ..."
    "$VENV_PY" scripts/real_capture.py --n "$TRIALS" >"$LOG" 2>&1

    PROC_DT=$(($(date +%s) - PROC_START))
    AFTER=$(ls "$HIFIGAN_BISECT_OUT" 2>/dev/null | wc -l)
    NEW_BUNDLES=$((AFTER - BEFORE))

    # Pull the per-trial summary lines for the run-log.
    LAST_TRIALS=$(grep -E "capture_hook\] mel|real_capture\] trial" "$LOG" | tail -n "$((TRIALS * 2))")
    echo "[grind] process $i: ${PROC_DT}s, $NEW_BUNDLES new bundles"
    echo "$LAST_TRIALS" | sed 's/^/[grind]   /'

    if [ "$NEW_BUNDLES" -gt 0 ]; then
        echo "[grind] !!! ANOMALY HIT in process $i, bundle saved. Stopping."
        TOTAL_DT=$(($(date +%s) - START))
        echo "[grind] total: ${TOTAL_DT}s across $i processes"
        exit 0
    fi
done

TOTAL_DT=$(($(date +%s) - START))
echo "[grind] no anomalies in $MAX_PROCESSES processes (${TOTAL_DT}s)."
exit 1
