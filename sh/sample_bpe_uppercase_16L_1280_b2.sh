#!/bin/zsh
#
# sample_bpe_uppercase_16L_1280_b2.sh — generate samples from the M3
# BPE training checkpoint pt/bpe_uppercase_16L_1280_b2.pt.
#
# Run from the M3 (where the checkpoint lives). Output goes to a
# timestamped log in terminal_logs/ on the M3 AND echoes to the
# terminal via tee.
#
# Any extra CLI args pass through to py/sample.py. Example overrides:
#   sh/sample_bpe_uppercase_16L_1280_b2.sh --prompt "It was a dark and stormy"
#   sh/sample_bpe_uppercase_16L_1280_b2.sh --temperature 0.6 --num_samples 10
#   sh/sample_bpe_uppercase_16L_1280_b2.sh --max_tokens 500
#
# sample.py auto-detects case-preserved tokenizers (this one is) and
# disables the default prompt-lowercasing + post-hoc capitalize_sentences.
#
# Note on units: max_tokens here is in BPE tokens, not characters.
# 300 BPE tokens corresponds to roughly 1,350 characters of generated
# text at this corpus's ~4.5 chars-per-token ratio.

MODEL="pt/bpe_uppercase_16L_1280_b2.pt"
LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/sample_bpe_uppercase_16L_1280_b2_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"

echo "model:  $MODEL"
echo "output: $LOG_FILE"
echo ""

# Use `python` from PATH; on the M3 this resolves to the homebrew Python
# 3.13 that has torch installed (same one the training is running under).
python py/sample.py \
    --model "$MODEL" \
    --num_samples 5 \
    --max_tokens 300 \
    --temperature 0.8 \
    --top_k 40 \
    "$@" 2>&1 | tee "$LOG_FILE"
