#!/bin/bash
#
# train_cuda.sh - Linux/CUDA port of train.sh (for the RTX A6000 workstation).
#
# Differences from sh/train.sh (the macOS/zsh wrapper):
#   - #!/bin/bash instead of #!/bin/zsh (no zsh on the Linux box).
#   - Runs the conda-env interpreter ($HOME/miniforge3/envs/bpe_char/bin/python)
#     instead of a bare `python`, so it works without activating the env first.
#     Override by exporting PYTHON=... before calling.
#   - Backgrounds train.py with nohup + disown and then EXITS instead of
#     `tail -f`-ing the log. The tail-follow in train.sh is what left orphaned
#     wrapper processes holding a static log on the Macs (see HANDOFF). Here the
#     wrapper returns immediately, printing the PID and log path; tail the log
#     yourself to watch progress.
#
# py/train.py auto-selects device: cuda -> mps -> cpu (train.py:661), and
# handles bf16 on CUDA (train.py:682), so no Python changes are needed.
#
# Usage: sh/train_cuda.sh [all train.py arguments]
# Stop a run with SIGTERM: kill -TERM <PID> (train.py exits cleanly; the rolling
# checkpoint means at most eval_interval iters of work are lost).

PYTHON="${PYTHON:-$HOME/miniforge3/envs/bpe_char/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: python interpreter not found/executable at: $PYTHON"
    echo "Set PYTHON=... or create the bpe_char conda env first."
    exit 1
fi

# Extract --input / --output filenames for log naming
INPUT_FILE=""
OUTPUT_FILE=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--input" ]]; then
        INPUT_FILE="$arg"
    elif [[ "$prev_arg" == "--output" ]]; then
        OUTPUT_FILE="$arg"
    fi
    prev_arg="$arg"
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: --input argument is required"
    echo "Usage: $0 --input <file> [other train.py arguments]"
    exit 1
fi

if [[ -n "$OUTPUT_FILE" ]]; then
    LOG_BASENAME=$(basename "$OUTPUT_FILE" | sed 's/\.[^.]*$//')
else
    LOG_BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
fi

TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
PT_DIR="pt"
LOG_DIR="terminal_logs"
LOG_FILE="${LOG_DIR}/terminal_log_for_${LOG_BASENAME}_${TIMESTAMP}.txt"

mkdir -p "$PT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Training wrapper (CUDA) started"
echo "Input file:  $INPUT_FILE"
echo "Python:      $PYTHON"
echo "Checkpoints: $PT_DIR"
echo "Log file:    $LOG_FILE"
echo "========================================"

# Log the command line first (self-documenting log)
echo "Command: $PYTHON -u py/train.py --checkpoints_to \"$PT_DIR\" $*" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Python:  $PYTHON" >> "$LOG_FILE"
echo "Host:    $(hostname)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Launch in background, detached, output appended to the log
nohup "$PYTHON" -u py/train.py --checkpoints_to "$PT_DIR" "$@" >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
disown

echo "Training started with PID $TRAIN_PID"
echo "Log:  $LOG_FILE"
echo "Watch: tail -f $LOG_FILE"
echo "Stop:  kill -TERM $TRAIN_PID"
