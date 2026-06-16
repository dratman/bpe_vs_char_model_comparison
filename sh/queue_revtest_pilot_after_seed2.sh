#!/bin/bash
#
# queue_revtest_pilot_after_seed2.sh — experiment-queue runner for coupler-queue
# item 0002 (reversed-text learnability pilot).
#
# Waits for the currently-live A6000 run — the seed-2 --no_fused stability trial
# (PID passed as $1) — to EXIT so the GPU is free, then runs the matched pilot
# pair BACK-TO-BACK (each auto-terminates at max_iters 10000, ~tens of minutes on
# the idle A6000):
#   1. sh/train_char_revtest_pilot_6L_768_forward_CUDA.sh   (forward baseline)
#   2. sh/train_char_revtest_pilot_6L_768_reversed_CUDA.sh  (reversed)
#
# This honors spec 0002's "do NOT run concurrently with a GPU-saturated training;
# slot it into an idle window" — it never launches while the live trial holds the
# GPU.
#
# Launch (on the A6000, from the repo root):
#   nohup sh/queue_revtest_pilot_after_seed2.sh 2314695 \
#       > terminal_logs/queue_revtest_pilot.log 2>&1 &
#
# Double-launch guard below: if a pilot is already training, abort cleanly.

set -u

WAIT_PID="${1:?usage: $0 <pid-of-live-run-to-wait-for>}"
POLL=300

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

if pgrep -f "py/train.py.*char_revtest_pilot" > /dev/null; then
    echo "[$(stamp)] [queue] a revtest pilot is already running; aborting duplicate launch."
    exit 0
fi

echo "[$(stamp)] [queue] waiting for PID $WAIT_PID (live A6000 run) to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep "$POLL"; done
echo "[$(stamp)] [queue] PID $WAIT_PID gone — GPU should be free."

run_and_wait() {
    local script="$1" label="$2"
    echo "[$(stamp)] [queue] launching $label: $script"
    local out pid log
    out=$("$script")
    echo "$out"
    pid=$(echo "$out" | sed -n 's/^Training started with PID //p')
    log=$(echo "$out" | sed -n 's/^Log:  //p')
    if [[ -z "$pid" ]]; then
        echo "[$(stamp)] [queue] ERROR: could not parse PID for $label; aborting."
        return 1
    fi
    sleep 120
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$(stamp)] [queue] ERROR: $label PID $pid died within 120 s (startup crash?). Check $log"
        return 1
    fi
    echo "[$(stamp)] [queue] $label PID $pid healthy after 120 s; waiting for completion..."
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "[$(stamp)] [queue] $label PID $pid finished. Log: $log"
    return 0
}

run_and_wait sh/train_char_revtest_pilot_6L_768_forward_CUDA.sh  "FORWARD"  || exit 1
run_and_wait sh/train_char_revtest_pilot_6L_768_reversed_CUDA.sh "REVERSED" || exit 1

echo "[$(stamp)] [queue] revtest pilot pair COMPLETE. Next: collect val-loss series"
echo "[$(stamp)] [queue]   (convert nats->bits: bpc = nats / ln 2), plot both curves,"
echo "[$(stamp)] [queue]   pull early/mid/late samples (reverse the reversed run's"
echo "[$(stamp)] [queue]   samples back to normal reading order), write result.md."
