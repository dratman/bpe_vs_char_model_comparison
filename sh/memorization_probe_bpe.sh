#!/bin/zsh
#
# memorization_probe_bpe.sh — run py/memorization_probe.py across the three
# BPE overtraining contrast points {132K best, 168K true-min, 220K final},
# **on the Studio** (the M3 is memory-tight; the Studio has headroom).
#
# Background (see HANDOFF.md, BPE-resumed section, and diary 094): the BPE
# run was resumed past its early stop to test whether samples grow more
# memorized as training proceeds past the val-loss minimum. The true val
# minimum turned out to be iter 168K (epoch 5.34), not the original-run
# stop near 132K. The meaningful contrast points are therefore:
#   - iter 132K best  -> pt/bpe_uppercase_16L_1280_b2.pt        (original-run best)
#   - iter 168K best  -> pt/bpe_uppercase_16L_1280_b2_resumed.pt (resumed-run best = true min)
#   - iter 220K final -> pt/bpe_uppercase_16L_1280_b2_resumed_final.pt
# If extractable-memorization (mean matched chars / extract%) rises
# monotonically across these, the overtraining hypothesis is confirmed.
#
# What this script does:
#   1. rsyncs the three checkpoints + tokenizer metadata from the M3 to
#      the Studio's pt/ (skips unchanged files via default size+mtime).
#   2. Runs py/memorization_probe.py over all three at once.
#   3. Tees output to a timestamped log in terminal_logs/.
#
# Meta-naming quirk (same as the sample scripts): the tokenizer was not
# re-saved on resume, so the M3's meta files are all named
# bpe_uppercase_16L_1280_b2_meta.{pkl,json}. memorization_probe.py (like
# sample.py) derives the meta path from the model path, so the resumed
# checkpoints expect *_resumed_meta.*. We rename in transit for those.
#
# CORPUS: the probe draws prompts + true continuations from the SAME corpus
# the models were trained on. It must be present locally on the Studio at
# txt_local/corpus_high_quality_uppercase_2026_05_08.txt (it is — the char
# run trained from it there).
#
# Any extra CLI args pass through to py/memorization_probe.py, e.g.:
#   sh/memorization_probe_bpe.sh --num_prompts 400 --continue_chars 512
#
# NOTE: this wrapper was authored on the Linux workstation and has NOT been
# run cross-machine. Verify the M3 hostname/paths below before first use.

M3_HOST="RalphDratman@MacBookProM3Max.local"
M3_REPO="0-Home-Working-on-M3-Pro/bpe_vs_char_model_comparison"
CORPUS="txt_local/corpus_high_quality_uppercase_2026_05_08.txt"

# (model_base_on_M3, local_meta_base_for_derivation)
# The first column is the .pt file (same name both sides). The second is
# the meta base name the probe will derive from that .pt path — we make
# sure a meta file exists under that name locally.
M132="bpe_uppercase_16L_1280_b2"                 # iter 132K best; meta base = itself
M168="bpe_uppercase_16L_1280_b2_resumed"         # iter 168K best; meta base = itself
M220="bpe_uppercase_16L_1280_b2_resumed_final"   # iter 220K final; meta base = b2_resumed

LOG_DIR="terminal_logs"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_FILE="${LOG_DIR}/memorization_probe_bpe_${TIMESTAMP}.txt"
PYTHON=$HOME/miniforge3/bin/python3

mkdir -p "$LOG_DIR" pt

RSYNC_OPTS=(-avh --partial -e "ssh -o ConnectTimeout=5")

stage_model () {  # $1 = base name of the .pt on the M3 (no extension)
    rsync "${RSYNC_OPTS[@]}" \
        "${M3_HOST}:${M3_REPO}/pt/${1}.pt" "pt/${1}.pt" \
        || { echo "rsync of ${1}.pt failed; cannot probe"; exit 1; }
}

stage_meta () {  # $1 = M3 meta base, $2 = local meta base (rename in transit)
    for EXT in pkl json; do
        rsync "${RSYNC_OPTS[@]}" \
            "${M3_HOST}:${M3_REPO}/pt/${1}_meta.${EXT}" \
            "pt/${2}_meta.${EXT}" \
            || { echo "rsync of ${1}_meta.${EXT} failed; cannot probe"; exit 1; }
    done
}

echo "rsync M3 → Studio (skips unchanged files)…"
stage_model "$M132"
stage_model "$M168"
stage_model "$M220"
# All three derive their tokenizer from one of two meta bases. The M3 has
# only the single pre-resume meta (b2); copy it to both expected names.
stage_meta "bpe_uppercase_16L_1280_b2" "bpe_uppercase_16L_1280_b2"
stage_meta "bpe_uppercase_16L_1280_b2" "bpe_uppercase_16L_1280_b2_resumed"

echo
echo "corpus: $CORPUS"
echo "output: $LOG_FILE"
echo

"$PYTHON" py/memorization_probe.py \
    --models "pt/${M132}.pt" "pt/${M168}.pt" "pt/${M220}.pt" \
    --corpus "$CORPUS" \
    --num_prompts 200 \
    --prompt_chars 512 \
    --continue_chars 256 \
    --extract_threshold 50 \
    --seed 42 \
    "$@" 2>&1 | tee "$LOG_FILE"
