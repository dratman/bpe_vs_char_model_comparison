#!/bin/bash
#
# queue_0003_reversed_after_seed2.sh
#
# Armed, NON-DESTRUCTIVE waiter for coupler-queue item 0003 (full-length
# reversed-char training). It only WAITS and LAUNCHES — it deletes nothing.
#
# WHY THIS EXISTS — scheduling decision (2026-06-15, worker linux-cuda, logged in
# HANDOFF): the seed-2 --no_fused trial is coasting to its 500K floor. Its
# diagnostic verdict is already in hand; the remaining value is only the diary-094
# error bar, which materializes ONLY at its floor. Rather than stop it mid-run
# (which would write a misleading partial _final.pt, perturb its cosine LR, and
# trip the resume guard), we let it finish and launch 0003 the moment the GPU
# frees. Same total GPU time, zero interruption hazards.
#
# DISK: 0003 is configured (save_interval=100000) to fit in the free space left
# after seed-2 finishes WITHOUT deleting any checkpoints — its footprint is
# ~35 GB (tokens + best + rolling + 4 intermediate + final). The dead, regenerable
# token caches were reclaimed separately (foreground, supervised). So this waiter
# does NO cleanup; it just launches.
#
# What it does, in order:
#   1. Poll every 5 min until the seed-2 training process is gone (it exits on its
#      own at iter 500K) AND the GPU is idle (<3 GB resident).
#   2. Launch 0003 via sh/train_char_uppercase_16L_1280_reversed_CUDA.sh (which has
#      its own completed/running guard).
#
# Detached: launch with setsid+nohup so it survives the Claude session ending.
# Double-launch guard below prevents two waiters racing.
# Log: terminal_logs/queue_0003_waiter.log

set -u
cd "$(dirname "$0")/.." || exit 1
LOG=terminal_logs/queue_0003_waiter.log
SEED2_PAT="py/train.py.*char_uppercase_16L_1280_seed2_no_fused_cuda"
REV_FINAL=pt/char_uppercase_16L_1280_reversed_cuda_final.pt

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# --- Double-launch guard: single-instance flock (immune to wrapper-process
# false positives from setsid/nohup) ----------------------------------------
ME=$$
LOCK=/tmp/queue_0003_waiter.lock
exec 9>"$LOCK" || { log "Cannot open lockfile $LOCK; exiting."; exit 1; }
if ! flock -n 9; then
    log "Another waiter already holds the lock; exiting."
    exit 0
fi
if [ -f "$REV_FINAL" ]; then
    log "0003 already completed (_final.pt exists); waiter exiting."
    exit 0
fi

log "Waiter armed (PID $ME). Waiting for seed-2 to free the GPU..."

# --- 1. Wait for seed-2 to finish + GPU to go idle --------------------------
while pgrep -f "$SEED2_PAT" > /dev/null 2>&1; do
    sleep 300
done
log "seed-2 training process is gone. Confirming GPU is idle..."
for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$used" ] && [ "$used" -lt 3000 ] 2>/dev/null; then
        break
    fi
    log "GPU still shows ${used} MiB used; waiting 60s..."
    sleep 60
done

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$used" ] && [ "$used" -ge 3000 ] 2>/dev/null; then
    log "GPU still busy (${used} MiB) after wait; NOT launching 0003. Re-arm a waiter manually."
    exit 1
fi

# --- 2. Launch 0003 (its own guard prevents double-launch) ------------------
log "GPU idle. df avail before launch:"
df -BG --output=avail . | tail -1 >> "$LOG"
log "Launching 0003 reversed-char training..."
bash sh/train_char_uppercase_16L_1280_reversed_CUDA.sh >> "$LOG" 2>&1
log "0003 launch script returned. Waiter done."
