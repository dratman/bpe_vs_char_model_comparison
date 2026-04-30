#!/bin/zsh
#
# train_imitator_L8_kl.sh - Imitator with downstream KL loss
#
# Instead of matching vectors (cosine similarity), this trains the
# imitator to produce vectors that, when decoded through the frozen
# model's back half, give the same token probability distribution.
#
# Full 2048 dimensions, split at layer 8.
#

LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/terminal_log_for_imitator_L8_kl_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"
mkdir -p "pt"

echo "========================================"
echo "Imitator training: layer 8, downstream KL loss"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

echo "Command: python3 -u py/train_imitator.py --loss_type kl ..." > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

python3 -u py/train_imitator.py \
    --small_model pt/bpe_16L16H_iter170000.pt \
    --input txt_local/corpus_books_shuffled_2026_04_18.txt \
    --output pt/imitator_L8_kl.pt \
    --token_cache txt_local/corpus_tokens_32k.pt \
    --split_layer 8 \
    --d_model 2048 \
    --n_layer 6 \
    --n_head 16 \
    --block_size 512 \
    --batch_size 4 \
    --max_iters 5000 \
    --learning_rate 1e-4 \
    --warmup_iters 200 \
    --eval_interval 200 \
    --log_interval 10 \
    --save_interval 1000 \
    --loss_type kl \
    --small_model_precision float32 \
    --sanity_check >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

trap "echo ''; echo 'Stopping training (PID $TRAIN_PID)...'; kill $TRAIN_PID 2>/dev/null; exit" INT TERM

echo "Training started with PID $TRAIN_PID"
echo "Output is being logged to: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop both tail AND training"
echo ""

while [[ ! -f "$LOG_FILE" ]]; do
    sleep 0.1
done

tail -f "$LOG_FILE"
