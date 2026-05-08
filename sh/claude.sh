#!/bin/zsh
#
# claude.sh - Wrapper for Claude Code that logs the session transcript
#
# Usage: ./sh/claude.sh [any claude arguments]
#
# This script:
# - Creates a timestamped log file in claude_code_sessions/
# - Uses the macOS `script` command to capture all terminal output
# - When the session ends, strips ANSI escape codes for readability
#
# The raw (with ANSI codes) version is also kept as .raw.txt in case
# the stripping loses important formatting.

LOG_DIR="claude_code_sessions"
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")
RAW_FILE="${LOG_DIR}/SESSION_${TIMESTAMP}.raw.txt"
CLEAN_FILE="${LOG_DIR}/SESSION_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "Claude Code session logging"
echo "Log file: $CLEAN_FILE"
echo "========================================"
echo ""

# Use script to capture the full terminal session.
# -q = quiet (no "Script started/done" messages in file)
# On macOS, syntax is: script -q <file> <command> [args...]
# Use the full path to the claude binary to avoid alias/function loops.
if [ -x /usr/local/bin/claude ]; then
    CLAUDE_BIN="/usr/local/bin/claude"
elif [ -x /opt/homebrew/bin/claude ]; then
    CLAUDE_BIN="/opt/homebrew/bin/claude"
elif [ -x "$HOME/.npm-global/bin/claude" ]; then
    CLAUDE_BIN="$HOME/.npm-global/bin/claude"
else
    echo "ERROR: Cannot find claude binary. Install with: npm install -g @anthropic-ai/claude-code"
    exit 1
fi
echo "Using claude binary: $CLAUDE_BIN"
script -q "$RAW_FILE" "$CLAUDE_BIN" "$@"

echo ""
echo "Raw session saved to: $RAW_FILE"

# Strip ANSI escape codes and control characters for a readable version.
# This handles: color codes, cursor movement, clearing, etc.
LC_ALL=C sed \
    -e $'s/\x1b\[[0-9;]*[a-zA-Z]//g' \
    -e $'s/\x1b\][^\x07]*\x07//g' \
    -e $'s/\x1b[()][B0-2]//g' \
    -e $'s/\x0f//g' \
    -e $'s/\r//g' \
    "$RAW_FILE" > "$CLEAN_FILE"

echo "Cleaned session saved to: $CLEAN_FILE"
echo ""
echo "You can delete the .raw.txt file if the cleaned version looks good."
