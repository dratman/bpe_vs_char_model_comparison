#!/bin/zsh
#
# sample_char_uppercase_16L_1280.sh — generate samples from the Studio
# char training checkpoint pt/char_uppercase_16L_1280.pt.
#
# Run from the Studio (where the checkpoint lives). Output goes to a
# timestamped log in terminal_logs/ AND echoes to the terminal via tee.
#
# Any extra CLI args pass through to py/sample.py. Example overrides:
#   sh/sample_char_uppercase_16L_1280.sh --prompt "It was a dark and stormy"
#   sh/sample_char_uppercase_16L_1280.sh --temperature 0.6 --num_samples 10
#   sh/sample_char_uppercase_16L_1280.sh --max_tokens 1000 --no_compile
#
# sample.py auto-detects case-preserved tokenizers (this one is) and
# disables the default prompt-lowercasing + post-hoc capitalize_sentences.

MODEL="pt/char_uppercase_16L_1280.pt"
LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/sample_char_uppercase_16L_1280_${TIMESTAMP}.txt"
PYTHON=/Users/RalphDratman/miniforge3/bin/python3

mkdir -p "$LOG_DIR"

echo "model:  $MODEL"
echo "output: $LOG_FILE"
echo ""

"$PYTHON" py/sample.py \
    --model "$MODEL" \
    --num_samples 5 \
    --max_tokens 500 \
    --temperature 0.8 \
    --top_k 40 \
    "$@" 2>&1 | tee "$LOG_FILE"
