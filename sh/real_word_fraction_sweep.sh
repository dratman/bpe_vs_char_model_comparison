#!/bin/zsh
#
# real_word_fraction_sweep.sh — diary 099 step 1: run py/real_word_fraction.py
# across the baseline char run and the no-GELU matched-LR run, then run the
# long-pending BPE memorization probe (HANDOFF TODO since 2026-06-06).
#
# Runs ON THE STUDIO from the repo root. Designed to be launched under
# nohup so it survives SSH disconnect:
#   nohup sh/real_word_fraction_sweep.sh > terminal_logs/rwf_sweep_nohup.log 2>&1 &
#
# Three sequential jobs (sequential on purpose — the Studio GPU is also
# running the live no-GELU training; one extra MPS consumer at a time):
#   1. Real-word-fraction sweep over the BASELINE char checkpoints
#      (every 40K iters after 80K, plus best at 482K and final at 500K).
#   2. Real-word-fraction sweep over the NO-GELU matched-LR checkpoints
#      (every 5K iters, 5K-80K — the window where invented words fade).
#   3. py/memorization_probe.py over the three BPE overtraining contrast
#      points {132K original best, 168K true min, 220K final}. This calls
#      the python directly rather than sh/memorization_probe_bpe.sh because
#      that wrapper insists on rsyncing from the M3 first and fails if the
#      M3 is asleep; the checkpoints must already be staged in pt/ (they
#      were, fresh from the M3, on 2026-06-10).
#
# Each job tees to its own timestamped file in terminal_logs/, and the
# sweeps also write a TSV (one row per checkpoint) for plotting.

PYTHON=/Users/RalphDratman/miniforge3/bin/python3
CORPUS=txt_local/corpus_high_quality_uppercase_2026_05_08.txt
TS=$(date +"%Y_%m_%d_%H%M")

BASELINE=(
    pt/char_uppercase_16L_1280_iter20000.pt
    pt/char_uppercase_16L_1280_iter40000.pt
    pt/char_uppercase_16L_1280_iter60000.pt
    pt/char_uppercase_16L_1280_iter80000.pt
    pt/char_uppercase_16L_1280_iter120000.pt
    pt/char_uppercase_16L_1280_iter160000.pt
    pt/char_uppercase_16L_1280_iter200000.pt
    pt/char_uppercase_16L_1280_iter240000.pt
    pt/char_uppercase_16L_1280_iter280000.pt
    pt/char_uppercase_16L_1280_iter320000.pt
    pt/char_uppercase_16L_1280_iter360000.pt
    pt/char_uppercase_16L_1280_iter400000.pt
    pt/char_uppercase_16L_1280_iter440000.pt
    pt/char_uppercase_16L_1280_iter480000.pt
    pt/char_uppercase_16L_1280.pt
    pt/char_uppercase_16L_1280_final.pt
)

NOGELU=(
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter5000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter10000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter15000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter20000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter25000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter30000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter35000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter40000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter45000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter50000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter55000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter60000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter65000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter70000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter75000.pt
    pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter80000.pt
)

echo "=== Job 1/3: real-word fraction, baseline char run ($TS) ==="
"$PYTHON" py/real_word_fraction.py \
    --models "${BASELINE[@]}" \
    --corpus "$CORPUS" \
    --num_samples 16 --sample_chars 512 --seed 42 --progress_every 8 \
    --tsv "terminal_logs/real_word_fraction_baseline_${TS}.tsv" \
    2>&1 | tee "terminal_logs/real_word_fraction_baseline_${TS}.txt"

echo
echo "=== Job 2/3: real-word fraction, no-GELU matched-LR run ==="
"$PYTHON" py/real_word_fraction.py \
    --models "${NOGELU[@]}" \
    --corpus "$CORPUS" \
    --num_samples 16 --sample_chars 512 --seed 42 --progress_every 8 \
    --tsv "terminal_logs/real_word_fraction_no_gelu_${TS}.tsv" \
    2>&1 | tee "terminal_logs/real_word_fraction_no_gelu_${TS}.txt"

echo
echo "=== Job 3/3: BPE memorization probe (132K / 168K / 220K) ==="
"$PYTHON" py/memorization_probe.py \
    --models pt/bpe_uppercase_16L_1280_b2.pt \
             pt/bpe_uppercase_16L_1280_b2_resumed.pt \
             pt/bpe_uppercase_16L_1280_b2_resumed_final.pt \
    --corpus "$CORPUS" \
    --num_prompts 200 --prompt_chars 512 --continue_chars 256 \
    --extract_threshold 50 --seed 42 \
    2>&1 | tee "terminal_logs/memorization_probe_bpe_${TS}.txt"

echo
echo "=== Sweep complete ==="
