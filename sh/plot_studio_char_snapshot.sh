#!/bin/zsh
#
# plot_studio_char_snapshot.sh — Plot the in-progress Studio char training.
#
# The char training log is local to this Studio (in terminal_logs/), so
# unlike the M3 BPE snapshot script there is no rsync step. Just point
# plot_current_run.py at the right log file.
#
# Output: plots/char_uppercase_16L_1280_loss.png — same stem as the
# checkpoint pt/char_uppercase_16L_1280.pt. Re-run any time to refresh.
#
# If a different Studio char run is started later, edit LOG_NAME and
# OUT below.

set -e

LOG_NAME="terminal_log_for_char_uppercase_16L_1280_2026_05_09_0038.txt"
OUT="plots/char_uppercase_16L_1280_loss.png"

/Users/RalphDratman/miniforge3/bin/python3 py/plot_current_run.py \
    --log "terminal_logs/${LOG_NAME}" \
    --out "${OUT}"
