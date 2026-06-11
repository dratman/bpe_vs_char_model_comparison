#!/bin/bash
#
# queue_wordpiece_pair_after_seed2.sh — experiment-queue runner for the
# A6000 (diary 102). Waits for the seed-2 char replication (PID passed
# as $1) to finish, then runs the WordPiece pair back to back:
#   1. sh/train_wordpiece_uppercase_16L_1280_b2_CUDA.sh        (control, GELU on)
#   2. sh/train_wordpiece_uppercase_16L_1280_b2_no_gelu_CUDA.sh (ablation)
# The no-GELU run loads the control's tokenizer (--tokenizer_from), so
# run 2 must follow run 1 — order is not arbitrary.
#
# Launch (on the A6000, from the repo root):
#   nohup sh/queue_wordpiece_pair_after_seed2.sh 151814 \
#       > terminal_logs/queue_wordpiece_pair.log 2>&1 &
#
# Each training is itself nohup'd by train_cuda.sh; this script parses
# the child PID from train_cuda.sh's output and waits on it. If a child
# dies within 120 s of launch (startup crash), the queue aborts rather
# than blindly starting the next run. A child that dies LATER is
# indistinguishable from normal completion by PID alone, so the queue
# also greps the child's log for "Training complete" and warns loudly
# if it is missing — check the run's own log before trusting results.

set -u

WAIT_PID="${1:?usage: $0 <pid-to-wait-for>}"
POLL=300

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_and_wait() {  # $1 = launch script path
    echo "[$(stamp)] [queue] launching: $1"
    local out
    out=$("$1")
    echo "$out"
    local pid
    pid=$(echo "$out" | sed -n 's/^Training started with PID //p')
    if [[ -z "$pid" ]]; then
        echo "[$(stamp)] [queue] ERROR: could not parse training PID; aborting queue."
        exit 1
    fi
    local log
    log=$(echo "$out" | sed -n 's/^Log:  //p')
    sleep 120
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$(stamp)] [queue] ERROR: PID $pid died within 120 s (startup crash?). Aborting queue. Check $log"
        exit 1
    fi
    echo "[$(stamp)] [queue] PID $pid healthy; waiting for it to finish..."
    while kill -0 "$pid" 2>/dev/null; do sleep "$POLL"; done
    echo "[$(stamp)] [queue] PID $pid exited."
    if [[ -n "$log" ]] && ! grep -q "Training complete" "$log"; then
        echo "[$(stamp)] [queue] WARNING: '$log' has no 'Training complete' line — the run may have crashed mid-training. Continuing anyway; verify before using results."
    fi
}

echo "[$(stamp)] [queue] waiting for PID $WAIT_PID (seed-2 char run) to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep "$POLL"; done
echo "[$(stamp)] [queue] PID $WAIT_PID gone."

run_and_wait sh/train_wordpiece_uppercase_16L_1280_b2_CUDA.sh
run_and_wait sh/train_wordpiece_uppercase_16L_1280_b2_no_gelu_CUDA.sh

echo "[$(stamp)] [queue] WordPiece pair complete."
