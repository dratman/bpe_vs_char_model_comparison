#!/usr/bin/env python3
"""
handoff_conversation.py — Automated handoff between two Claude Code instances.

Runs a conversation where a new instance interrogates the outgoing instance
about the current state of the project. Each instance runs via `claude -p`
(non-interactive mode).

The outgoing instance gets context by continuing its existing session.
The new instance starts fresh but reads HANDOFF.md and can ask questions.

Usage:
    python intercom/handoff_conversation.py [--turns 10]

This script runs in a third terminal. It alternates:
  1. New instance asks a question about the project state
  2. Old instance (continuing its session) answers
  3. Repeat for N turns
  4. New instance summarizes what it learned

The full conversation is saved to intercom/handoff_transcript.txt
"""

import subprocess
import argparse
import os
import sys
from datetime import datetime

TRANSCRIPT = os.path.join(os.path.dirname(__file__), "handoff_transcript.txt")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def claude_ask(prompt, continue_session=False, system_prompt=None):
    """Run claude -p with a prompt and return the response."""
    cmd = ["claude", "-p"]
    if continue_session:
        cmd.append("--continue")
    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])
    cmd.append(prompt)

    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=PROJECT_DIR,
        timeout=300
    )
    return result.stdout.strip()


def log_turn(speaker, message, f):
    """Log a turn to the transcript file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n[{timestamp}] {speaker}:"
    print(header)
    print(message)
    print()
    f.write(header + "\n")
    f.write(message + "\n\n")
    f.flush()


def main():
    parser = argparse.ArgumentParser(description="Automated handoff conversation")
    parser.add_argument("--turns", type=int, default=10,
                        help="Number of question/answer turns (default: 10)")
    args = parser.parse_args()

    new_system = (
        "You are a NEW Claude Code instance taking over this project. "
        "You are interviewing the OUTGOING instance to understand the "
        "current state. Ask specific, probing questions. Don't accept "
        "vague answers. Your goal is to be able to continue the work "
        "without any gaps. Start by reading HANDOFF.md, then ask about "
        "anything unclear, unfinished, or potentially wrong."
    )

    old_system = (
        "You are the OUTGOING Claude Code instance for this project. "
        "A new instance is interviewing you about the project state. "
        "Answer honestly and completely. Flag anything you're uncertain "
        "about. Mention things you forgot to document. The goal is a "
        "complete transfer of your working knowledge."
    )

    with open(TRANSCRIPT, 'w') as f:
        f.write(f"Handoff conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Turns: {args.turns}\n")
        f.write("=" * 60 + "\n")

        # New instance reads HANDOFF.md and asks first question
        first_prompt = (
            "Read HANDOFF.md and CLAUDE.md. Then ask your first question "
            "to the outgoing instance about the project state. Focus on "
            "what's most critical to know right now."
        )
        question = claude_ask(first_prompt, system_prompt=new_system)
        log_turn("NEW INSTANCE", question, f)

        for turn in range(args.turns):
            # Old instance answers
            answer_prompt = (
                f"The new instance asks:\n\n{question}\n\n"
                "Answer completely and honestly."
            )
            answer = claude_ask(answer_prompt, continue_session=True,
                                system_prompt=old_system)
            log_turn("OUTGOING INSTANCE", answer, f)

            if turn < args.turns - 1:
                # New instance asks follow-up
                followup_prompt = (
                    f"The outgoing instance answered:\n\n{answer}\n\n"
                    "Ask your next question. Probe deeper on anything "
                    "that seems incomplete or uncertain."
                )
                question = claude_ask(followup_prompt, system_prompt=new_system)
                log_turn("NEW INSTANCE", question, f)
            else:
                # Final turn: new instance summarizes
                summary_prompt = (
                    f"The outgoing instance's final answer:\n\n{answer}\n\n"
                    "Now summarize everything you've learned in this handoff. "
                    "List: (1) current state, (2) pending tasks, (3) things "
                    "to verify, (4) any concerns."
                )
                summary = claude_ask(summary_prompt, system_prompt=new_system)
                log_turn("NEW INSTANCE (SUMMARY)", summary, f)

        f.write("=" * 60 + "\n")
        f.write("Handoff conversation complete.\n")

    print(f"\nTranscript saved to {TRANSCRIPT}")


if __name__ == "__main__":
    main()
