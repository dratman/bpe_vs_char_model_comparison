#!/usr/bin/env python3
"""Extract early/mid/late samples from the two pilot logs. The reversed run's
samples are character-reversed back to normal reading order so they are legible
(and so the raw reversed gibberish never needs to be surfaced)."""
import re

LOGS = {
    "forward": "terminal_logs/terminal_log_for_char_revtest_pilot_6L_768_forward_cuda_2026_06_15_1026.txt",
    "reversed": "terminal_logs/terminal_log_for_char_revtest_pilot_6L_768_reversed_cuda_2026_06_15_1053.txt",
}
ITERS = ["2000", "4000", "6000", "8000", "final"]
NUM = re.compile(r"^  (\d)\. (.*)$")


def blocks(path):
    """Return list of blocks; each block is list of (n, text) sample lines."""
    out, cur = [], []
    with open(path) as f:
        for line in f:
            m = NUM.match(line.rstrip("\n"))
            if m:
                n = int(m.group(1))
                if n == 1 and cur:
                    out.append(cur); cur = []
                cur.append((n, m.group(2)))
            else:
                if cur and len(cur) >= 1 and NUM.match("  1. x"):
                    pass
    if cur:
        out.append(cur)
    return out


def main():
    fwd = blocks(LOGS["forward"])
    rev = blocks(LOGS["reversed"])
    bymark = {"forward": dict(zip(ITERS, fwd)), "reversed": dict(zip(ITERS, rev))}

    lines = []
    for mark in ["2000", "6000", "final"]:
        label = {"2000": "EARLY (iter 2000)", "6000": "MID (iter 6000)",
                 "final": "LATE (final, ~iter 10000)"}[mark]
        lines.append(f"========== {label} ==========")
        lines.append("--- FORWARD (sample 1, as generated) ---")
        lines.append("  " + bymark["forward"][mark][0][1])
        lines.append("--- REVERSED (sample 1, RE-REVERSED to normal reading order) ---")
        rev_text = bymark["reversed"][mark][0][1]
        lines.append("  " + rev_text[::-1])
        lines.append("")

    with open("plots/revtest_pilot_samples.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote plots/revtest_pilot_samples.txt")


if __name__ == "__main__":
    main()
