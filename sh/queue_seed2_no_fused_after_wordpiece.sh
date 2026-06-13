#!/bin/bash
#
# queue_seed2_no_fused_after_wordpiece.sh — experiment-queue runner for
# the A6000 (diary 103 follow-up). Waits for the WordPiece-pair queue
# runner (PID passed as $1 — the running instance of
# queue_wordpiece_pair_after_seed2.sh) to EXIT, which happens only after
# BOTH WordPiece runs (control + no-GELU) have finished, then launches:
#   sh/train_char_uppercase_16L_1280_seed2_no_fused_CUDA.sh
# the seed-2 stability-fix trial 1 (non-fused AdamW on CUDA).
#
# Launch (on the A6000, from the repo root):
#   nohup sh/queue_seed2_no_fused_after_wordpiece.sh 152043 \
#       > terminal_logs/queue_seed2_no_fused.log 2>&1 &
#
# Waiting on the WordPiece-pair queue PID (not an individual training
# PID) is deliberate: that script exits only when the whole pair is done,
# so the GPU is guaranteed free. The trial is itself nohup'd by
# train_cuda.sh; this script parses the child PID and waits on it. If the
# child dies within 120 s of launch (startup crash) the queue aborts.
# Because this trial is meant to be INSPECTED early (is the val curve
# smooth by iter ~6-10K?), it intentionally does NOT auto-stop — it runs
# until the val floor or until a human SIGTERMs it. The "Training
# complete" grep below only warns; it does not gate anything.

set -u

WAIT_PID="${1:?usage: $0 <wordpiece-pair-queue-pid-to-wait-for>}"
POLL=300

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] [queue] waiting for PID $WAIT_PID (WordPiece-pair queue runner) to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep "$POLL"; done
echo "[$(stamp)] [queue] PID $WAIT_PID gone — WordPiece pair complete, GPU should be free."

echo "[$(stamp)] [queue] launching: sh/train_char_uppercase_16L_1280_seed2_no_fused_CUDA.sh"
out=$(sh/train_char_uppercase_16L_1280_seed2_no_fused_CUDA.sh)
echo "$out"
pid=$(echo "$out" | sed -n 's/^Training started with PID //p')
log=$(echo "$out" | sed -n 's/^Log:  //p')

if [[ -z "$pid" ]]; then
    echo "[$(stamp)] [queue] ERROR: could not parse training PID; trial not confirmed launched."
    exit 1
fi

sleep 120
if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$(stamp)] [queue] ERROR: PID $pid died within 120 s (startup crash?). Check $log"
    exit 1
fi
echo "[$(stamp)] [queue] PID $pid healthy after 120 s."
echo "[$(stamp)] [queue] INSPECT: tail the log and confirm the val curve is smooth"
echo "[$(stamp)] [queue]          (no 1.3<->2.7 oscillation) by iter ~6K-10K, then decide"
echo "[$(stamp)] [queue]          whether to let it run to the floor or 'kill -TERM $pid'."
echo "[$(stamp)] [queue] no-fused trial armed and running; queue runner exiting."
