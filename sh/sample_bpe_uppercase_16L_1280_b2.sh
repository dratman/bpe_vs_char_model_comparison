#!/bin/zsh
#
# sample_bpe_uppercase_16L_1280_b2.sh — generate samples from the M3
# BPE training checkpoint, but **run on the Studio**. The M3 is short
# on memory while training; sampling there forces the system into
# swap. The Studio's 192 GB has plenty of headroom.
#
# What this script does:
#   1. rsyncs the M3's best-val checkpoint and its tokenizer metadata
#      to the Studio's pt/ directory. If the M3 hasn't saved a new
#      checkpoint since last run, rsync skips the transfer (default
#      size+mtime check).
#   2. Runs py/sample.py against the local copy.
#   3. Tees output to a timestamped log in terminal_logs/.
#
# Any extra CLI args pass through to py/sample.py. Example overrides:
#   sh/sample_bpe_uppercase_16L_1280_b2.sh --prompt "It was a dark and stormy"
#   sh/sample_bpe_uppercase_16L_1280_b2.sh --temperature 0.6 --num_samples 10
#   sh/sample_bpe_uppercase_16L_1280_b2.sh --max_tokens 500
#
# sample.py auto-detects case-preserved tokenizers (this one is) and
# disables prompt-lowercasing + capitalize_sentences accordingly.
#
# Note on units: max_tokens here is in BPE tokens, not characters.
# 300 BPE tokens corresponds to roughly 1,350 characters of generated
# text at this corpus's ~4.5 chars-per-token ratio.

M3_HOST="RalphDratman@192.168.1.177"
M3_REPO="0-Home-Working-on-M3-Pro/bpe_vs_char_model_comparison"
RUN="bpe_uppercase_16L_1280_b2"

MODEL="pt/${RUN}.pt"
LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/sample_${RUN}_${TIMESTAMP}.txt"
PYTHON=/Users/RalphDratman/miniforge3/bin/python3

mkdir -p "$LOG_DIR" pt

echo "rsync M3 → Studio (skips files that have not changed)…"
RSYNC_OPTS=(-avh --partial -e "ssh -o ConnectTimeout=5")
for FILE in "${RUN}.pt" "${RUN}_meta.pkl" "${RUN}_meta.json"; do
    rsync "${RSYNC_OPTS[@]}" \
        "${M3_HOST}:${M3_REPO}/pt/${FILE}" \
        "pt/" || { echo "rsync of ${FILE} failed; cannot sample"; exit 1; }
done

echo
echo "model:  $MODEL"
echo "output: $LOG_FILE"
echo

"$PYTHON" py/sample.py \
    --model "$MODEL" \
    --num_samples 5 \
    --max_tokens 300 \
    --temperature 0.8 \
    --top_k 40 \
    "$@" 2>&1 | tee "$LOG_FILE"
