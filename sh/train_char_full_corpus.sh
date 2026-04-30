#!/bin/zsh
#
# train_char_full_corpus.sh - Character model on full combined corpus
# (Gutenberg high-quality + Wikipedia prose)
#
# 152M params, 12 layers, 1024 dim, 8 heads, character-level.
# Running on Mac Studio.
#

LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/terminal_log_for_char_full_corpus_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"
mkdir -p "pt"

echo "========================================"
echo "Training character model on full combined corpus"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

echo "Command: python3 -u py/train.py ..." > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

python3 -u py/train.py \
    --input txt_local/corpus_full_2026_04_29.txt \
    --output pt/char_full_corpus.pt \
    --checkpoints_to pt \
    --mode continuous \
    --tokenizer char \
    --n_layer 12 \
    --n_head 8 \
    --n_embd 1024 \
    --block_size 512 \
    --batch_size 16 \
    --max_iters 500000 \
    --learning_rate 3e-4 \
    --warmup_iters 2000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --eval_interval 2000 \
    --save_interval 20000 \
    --log_interval 100 \
    --sample_interval 10000 \
    --sample_max_tokens 300 >> "$LOG_FILE" 2>&1 &

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
