#!/bin/bash
# train_yawl_sweep_CUDA.sh
#
# Six-arm YAWL n_embd sweep (coupler-queue 0006).
# Runs sequentially (widest first): n_embd = 32, 28, 24, 20, 16, 12.
# Each arm: 1L 1H word-mode char model, block_size 17, 20K iters, seed 1337.
#
# Idempotent: skips any arm whose _final.pt already exists.
# Usage: bash sh/train_yawl_sweep_CUDA.sh [--dry_run]
#
# Run from the project root:
#   bash sh/train_yawl_sweep_CUDA.sh

set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniforge3/envs/bpe_char/bin/python}"
INPUT="txt_local/yawl_word_list_width_26.txt"
LOG_DIR="terminal_logs"
PT_DIR="pt"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python not found at $PYTHON"
    exit 1
fi
if [[ ! -f "$INPUT" ]]; then
    echo "Error: YAWL word list not found at $INPUT"
    exit 1
fi

mkdir -p "$LOG_DIR" "$PT_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry_run" ]]; then DRY_RUN=1; fi

TIMESTAMP=$(date +"%Y_%m_%d_%H%M")

for N_EMBD in 32 28 24 20 16 12; do
    OUTPUT_BASE="pt/yawl_word_1L1H_e${N_EMBD}_untied_cuda"
    FINAL_PT="${OUTPUT_BASE}_final.pt"
    LOG_FILE="${LOG_DIR}/terminal_log_for_yawl_word_1L1H_e${N_EMBD}_untied_cuda_${TIMESTAMP}.txt"

    if [[ -f "$FINAL_PT" ]]; then
        echo "=== n_embd=${N_EMBD}: already done (${FINAL_PT} exists), skipping ==="
        continue
    fi

    echo "========================================"
    echo "=== n_embd=${N_EMBD}: starting ==="
    echo "  output: ${OUTPUT_BASE}.pt"
    echo "  log:    ${LOG_FILE}"
    echo "========================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY RUN — would run python train.py for n_embd=${N_EMBD}]"
        continue
    fi

    # Log the command
    cat > "$LOG_FILE" << EOF
Command: $PYTHON -u py/train.py --checkpoints_to "$PT_DIR" (n_embd=${N_EMBD})
Started: $(date)
Python:  $PYTHON
Host:    $(hostname)
========================================
EOF

    # Run FOREGROUND so arms are sequential
    "$PYTHON" -u py/train.py \
        --input "$INPUT" \
        --output "$OUTPUT_BASE" \
        --checkpoints_to "$PT_DIR" \
        --mode word \
        --tokenizer char \
        --untie_weights \
        --n_layer 1 \
        --n_head 1 \
        --n_embd "$N_EMBD" \
        --block_size 17 \
        --max_iters 20000 \
        --seed 1337 \
        --no_fused \
        --batch_size 64 \
        --learning_rate 6e-4 \
        --warmup_iters 2000 \
        --dropout 0.0 \
        --val_split 0.1 \
        --precision float32 \
        --eval_interval 500 \
        --eval_iters 20 \
        --save_interval 2500 \
        --log_interval 500 \
        --sample_interval 100000000 \
        2>&1 | tee -a "$LOG_FILE"

    echo "=== n_embd=${N_EMBD}: finished at $(date) ==="
    echo ""
done

echo "========================================"
echo "All arms complete."
echo "========================================"
