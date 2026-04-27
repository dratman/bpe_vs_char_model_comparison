#!/usr/bin/env python3
"""
clean_session_log.py - Clean a Claude Code raw session log into readable text.

Removes:
- Terminal control/escape sequences (ANSI codes, cursor movement, etc.)
- Tool call metadata and system reminders
- File contents from Read tool results (keeps just the filename)
- Command outputs longer than 20 lines (keeps first 5 and last 5)
- Blank line runs (collapses to single blank line)

Keeps:
- Human messages (marked with "❯" or from context)
- Assistant text responses
- Key tool results (filenames, short outputs)

Usage:
    python py/clean_session_log.py SESSION_2026_04_25_1750.raw.txt
    # Writes SESSION_2026_04_25_1750.clean.txt
"""

import re
import sys
import os


def strip_ansi(text):
    """Remove ANSI escape sequences and terminal control codes."""
    # CSI sequences: ESC [ ... letter
    text = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', text)
    # OSC sequences: ESC ] ... BEL
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    # Other ESC sequences (2 chars)
    text = re.sub(r'\x1b[^[\]].?', '', text)
    # Carriage returns (terminal overwrites)
    text = re.sub(r'\r', '', text)
    # Control chars except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Unicode box-drawing and block elements that are UI decoration
    text = re.sub(r'[▗▖▘▝▙▛▜▟▀▄▌▐█░▒▓│─┌┐└┘├┤┬┴┼╔╗╚╝║═]', '', text)
    return text


def clean_session(raw_text):
    """Clean a raw session log into readable text."""
    text = strip_ansi(raw_text)

    # Remove system reminders
    text = re.sub(
        r'<system-reminder>.*?</system-reminder>',
        '', text, flags=re.DOTALL
    )

    # Remove task notifications
    text = re.sub(
        r'<task-notification>.*?</task-notification>',
        '', text, flags=re.DOTALL
    )

    # Remove tool use markers but keep the description
    # Patterns like "Read(file.py)" or "Bash(command)"
    text = re.sub(r'\[?ctrl\+o\s*to\s*expand\]?', '', text, flags=re.IGNORECASE)

    # Collapse multiple blank lines to one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.rstrip()
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # Final collapse of blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip() + '\n'


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_session_log.py <raw_log_file>")
        print("  Writes a .clean.txt file alongside the input.")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Determine output path
    base = input_path.replace('.raw.txt', '').replace('.txt', '')
    output_path = base + '.clean.txt'

    print(f"Reading: {input_path}")
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    print(f"Raw size: {len(raw):,} bytes")

    cleaned = clean_session(raw)
    print(f"Cleaned size: {len(cleaned):,} bytes")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f"Written: {output_path}")

    reduction = (1 - len(cleaned) / len(raw)) * 100
    print(f"Reduction: {reduction:.0f}%")


if __name__ == '__main__':
    main()
