#!/bin/bash
#
# heartbeat_status.sh — linux-cuda worker heartbeat (editor task 2026-06-21).
#
# Gathers GPU / training / queue status, PRINTS it to the terminal (preserving the
# old periodic terminal report), and PUBLISHES the same summary to the coupler-queue
# repo at status/linux-cuda.md (commit + push) so the browser Editor can see
# "what's running?" without a relay. Routine/non-destructive (no kill/rm/force-push).
#
# Run by the 3-hourly status-wake cron, and runnable by hand.
set -u
PROJ=/home/owner/bpe_vs_char_model_comparison
CQ=/home/owner/coupler-queue
UTC=$(date -u '+%Y-%m-%dT%H:%MZ')
LOCAL=$(date '+%Y-%m-%d %H:%M %Z')

cd "$PROJ" || exit 1
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
TRAIN=$(pgrep -af "py/train.py" | grep -v pgrep | grep -oE "output pt/[^ ]+" | sed 's#output ##' | head -1)
TRAINPID=$(pgrep -f "py/train.py" | grep -v "heartbeat\|pgrep" | head -1)

if [ -n "$TRAIN" ]; then
    BASE=$(basename "$TRAIN" .pt)
    LOG=$(ls -t terminal_logs/terminal_log_for_${BASE}_*.txt 2>/dev/null | head -1)
    ITER=$(grep -oE "iter [0-9]+" "$LOG" 2>/dev/null | tail -1)
    EVAL=$(grep -E "Step .* val loss" "$LOG" 2>/dev/null | tail -1 | sed -E 's/^\[[^]]*\] (\[[^]]*\] )?//')
    RATE=$(grep "iters/sec" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+ iters/sec")
else
    BASE="(none — GPU idle)"; LOG=""; ITER="-"; EVAL="-"; RATE="-"
fi

cd "$CQ" && git pull --rebase -q origin main 2>/dev/null
PEND=$(ls pending/ 2>/dev/null | grep -v '^\.' | tr '\n' ' ')
RUN=$(ls running/ 2>/dev/null | grep -v '^\.' | tr '\n' ' ')
DONE=$(ls done/ 2>/dev/null | grep -v '^\.' | tr '\n' ' ')
FAIL=$(ls failed/ 2>/dev/null | grep -v '^\.' | tr '\n' ' ')
DISK=$(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')

# anomaly heuristics
ANOM=""
[ -z "$TRAIN" ] && [ -n "$PEND" ] && ANOM="${ANOM}IDLE GPU with pending queue item(s); "
if [ -n "$LOG" ]; then
    tail -30 "$LOG" 2>/dev/null | grep -qiE "Traceback|RuntimeError|out of memory|CUDA error|enforce fail" && ANOM="${ANOM}error/exception in recent log tail; "
fi
case "$DISK" in *9[5-9]%*|*100%*) ANOM="${ANOM}disk nearly full; ";; esac
[ -z "$ANOM" ] && ANOM="none"

mkdir -p "$CQ/status"
cat > "$CQ/status/linux-cuda.md" <<EOF
---
worker: linux-cuda
host: A6000
updated_utc: $UTC
updated_local: $LOCAL
---

# linux-cuda heartbeat

- **GPU:** ${GPU:-unknown}
- **Training process:** PID ${TRAINPID:-none} -> ${TRAIN:-idle}
- **Current run:** $BASE
  - iter: $ITER
  - latest eval: $EVAL
  - rate: ${RATE:-n/a}
  - log: ${LOG:-none}
- **Disk (/):** $DISK
- **Queue (after fresh pull):**
  - pending: ${PEND:-(empty)}
  - running: ${RUN:-(empty)}
  - done: ${DONE:-(empty)}
  - failed: ${FAIL:-(empty)}
- **Anomalies:** $ANOM
EOF

echo "===== linux-cuda heartbeat $LOCAL ($UTC) ====="
cat "$CQ/status/linux-cuda.md"

cd "$CQ" && git add status/linux-cuda.md \
    && git commit -q -m "heartbeat linux-cuda $UTC: ${BASE} ${ITER}" \
    && git push origin main 2>&1 | tail -1
