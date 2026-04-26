#!/bin/zsh
#
# train_imitator.sh - Wrapper for train_imitator.py with logging
#
# Usage: ./sh/train_imitator.sh [all train_imitator.py arguments]
#
# Same pattern as train.sh:
# - Creates a timestamped log file in terminal_logs/
# - Runs train_imitator.py in background with output to log
# - Runs tail -f so you can watch progress
# - Ctrl+C stops both tail AND training

# Extract --output filename for log naming
OUTPUT_FILE=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--output" ]]; then
        OUTPUT_FILE="$arg"
    fi
    prev_arg="$arg"
done

if [[ -z "$OUTPUT_FILE" ]]; then
    LOG_BASENAME="imitator"
else
    LOG_BASENAME=$(basename "$OUTPUT_FILE" | sed 's/\.[^.]*$//')
fi

TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
LOG_DIR="terminal_logs"
LOG_FILE="${LOG_DIR}/terminal_log_for_${LOG_BASENAME}_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"
mkdir -p "pt"

echo "========================================"
echo "Imitator training wrapper started"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

echo "Command: python -u py/train_imitator.py $@" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

python -u py/train_imitator.py "$@" >> "$LOG_FILE" 2>&1 &
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
