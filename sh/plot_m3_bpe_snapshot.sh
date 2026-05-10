#!/bin/zsh
#
# plot_m3_bpe_snapshot.sh — Pull the M3's in-progress BPE training log
# to the Studio's terminal_logs/ and run plot_current_run.py against it.
#
# Run from the Studio. Requires SSH key auth to the M3 (set up
# 2026-05-09; key in ~/.ssh/id_ed25519). The plot lands at
# plots/bpe_uppercase_16L_1280_b2_loss.png — same stem as the
# checkpoint pt/bpe_uppercase_16L_1280_b2.pt — and overwrites any
# previous snapshot.
#
# This script targets the *currently running* M3 BPE training. If a
# different M3 BPE run is started later, edit LOG_NAME and OUT below
# or generalize this into a parameterized version.

set -e

M3_HOST="RalphDratman@192.168.1.177"
M3_REPO="0-Home-Working-on-M3-Pro/bpe_vs_char_model_comparison"
LOG_NAME="terminal_log_for_bpe_uppercase_16L_1280_b2_2026_05_09_0926.txt"
OUT="plots/bpe_uppercase_16L_1280_b2_loss.png"

echo "rsync M3 log → Studio…"
rsync -avh --quiet \
    "${M3_HOST}:${M3_REPO}/terminal_logs/${LOG_NAME}" \
    "terminal_logs/"

echo "plot…"
/Users/RalphDratman/miniforge3/bin/python3 py/plot_current_run.py \
    --log "terminal_logs/${LOG_NAME}" \
    --out "${OUT}"
