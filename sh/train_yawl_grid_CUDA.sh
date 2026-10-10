#!/bin/bash
# train_yawl_grid_CUDA.sh
#
# coupler-queue 0007: YAWL width x depth grid, word-mode char models.
# Arms (smallest first): n_embd 64/128/256 x 1/2/4 layers, plus 32 x 2L and 32 x 4L.
# 4 heads, block_size 17, 20K iters, seed 1337 (seeds the split too).
#
# Idempotent: skips any arm whose _final.pt already exists.
# Usage (from project root): bash sh/train_yawl_grid_CUDA.sh [--dry_run]

set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniforge3/envs/bpe_char/bin/python}"
INPUT="txt_local/yawl_word_list_width_26.txt"
LOG_DIR="terminal_logs"
PT_DIR="pt"
ARMS="32:2 32:4 64:1 64:2 64:4 128:1 128:2 128:4 256:1 256:2 256:4"

[[ -x "$PYTHON" ]] || { echo "Error: Python not found at $PYTHON"; exit 1; }
[[ -f "$INPUT" ]] || { echo "Error: $INPUT not found"; exit 1; }
mkdir -p "$LOG_DIR" "$PT_DIR"

DRY_RUN=0
[[ "${1:-}" == "--dry_run" ]] && DRY_RUN=1
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")

for ARM in $ARMS; do
    N_EMBD=${ARM%%:*}; N_LAYER=${ARM##*:}
    OUTPUT_BASE="pt/yawl_word_L${N_LAYER}H4_e${N_EMBD}_untied_cuda"
    LOG_FILE="${LOG_DIR}/terminal_log_for_yawl_word_L${N_LAYER}H4_e${N_EMBD}_untied_cuda_${TIMESTAMP}.txt"
    if [[ -f "${OUTPUT_BASE}_final.pt" ]]; then
        echo "=== e${N_EMBD} L${N_LAYER}: done already, skipping ==="; continue
    fi
    echo "=== e${N_EMBD} L${N_LAYER}: starting $(date) ==="
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "[DRY RUN]"; continue; fi
    echo "Started: $(date)  Host: $(hostname)" > "$LOG_FILE"
    "$PYTHON" -u py/train.py \
        --input "$INPUT" --output "$OUTPUT_BASE" --checkpoints_to "$PT_DIR" \
        --mode word --tokenizer char --untie_weights \
        --n_layer "$N_LAYER" --n_head 4 --n_embd "$N_EMBD" \
        --block_size 17 --max_iters 20000 --seed 1337 --no_fused \
        --batch_size 64 --learning_rate 6e-4 --warmup_iters 2000 \
        --dropout 0.0 --val_split 0.1 --precision float32 \
        --eval_interval 500 --eval_iters 20 --save_interval 2500 \
        --log_interval 500 --sample_interval 100000000 \
        >> "$LOG_FILE" 2>&1
    echo "=== e${N_EMBD} L${N_LAYER}: finished $(date) ==="
done
echo "All arms complete."
