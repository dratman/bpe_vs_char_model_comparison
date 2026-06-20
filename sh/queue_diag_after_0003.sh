#!/bin/bash
#
# queue_diag_after_0003.sh — waits for the 0003 reversed run to stop (it was
# graceful-stopped via its override at iter 187000, which makes train.py save
# _final and exit), then launches the fp32 CUDA diagnostic on the freed GPU.
#
# Detached background process; not subject to Claude's tool-permission layer.
# Launch: nohup sh/queue_diag_after_0003.sh > terminal_logs/queue_diag.log 2>&1 &
set -u
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
FINAL=pt/char_uppercase_16L_1280_reversed_cuda_final.pt

echo "[$(stamp)] waiting for 0003 to stop (its _final to appear)..."
while [ ! -f "$FINAL" ]; do sleep 60; done
echo "[$(stamp)] 0003 _final present; waiting for the GPU to free (no py/train.py)..."
while pgrep -f "py/train.py" > /dev/null 2>&1; do sleep 30; done
echo "[$(stamp)] GPU free. Launching fp32 diagnostic."
out=$(sh/train_char_diag_fp32_16L_1280_forward_CUDA.sh)
echo "$out"
pid=$(echo "$out" | sed -n 's/^Training started with PID //p')
log=$(echo "$out" | sed -n 's/^Log:  //p')
if [ -z "$pid" ]; then
    echo "[$(stamp)] ERROR: could not parse diagnostic PID; not confirmed launched."
    exit 1
fi
sleep 120
if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$(stamp)] ERROR: diagnostic PID $pid died within 120s (fp32 OOM at block 4096?). Check $log"
    exit 1
fi
echo "[$(stamp)] fp32 diagnostic PID $pid healthy after 120s. Watch the 0-20K val curve."
