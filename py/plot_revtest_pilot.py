#!/usr/bin/env python3
"""
plot_revtest_pilot.py — coupler-queue 0002 analysis.

Parse the forward + reversed pilot logs, build the val-loss series in bits per
character (bpc = nats / ln 2), write a CSV, render the comparison plot, and print
a compact table + headline numbers for 0002.result.md.
"""
import re
import math
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2.0)
LOGS = {
    "forward": "terminal_logs/terminal_log_for_char_revtest_pilot_6L_768_forward_cuda_2026_06_15_1026.txt",
    "reversed": "terminal_logs/terminal_log_for_char_revtest_pilot_6L_768_reversed_cuda_2026_06_15_1053.txt",
}
# eval summary line: "... Step   200 | Epoch ... | train loss X | val loss Y | lr Z"
PAT = re.compile(r"Step\s+(\d+)\s+\|.*\|\s*val loss\s+([0-9.]+)\s*\|")


def parse(path):
    series = {}
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                series[int(m.group(1))] = float(m.group(2))
    return series


def main():
    fwd = parse(LOGS["forward"])
    rev = parse(LOGS["reversed"])
    steps = sorted(set(fwd) | set(rev))

    # CSV (bpc)
    with open("plots/revtest_pilot_val_bpc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "forward_val_nats", "reversed_val_nats",
                    "forward_val_bpc", "reversed_val_bpc"])
        for s in steps:
            fv, rv = fwd.get(s), rev.get(s)
            w.writerow([s, fv, rv,
                        f"{fv/LN2:.4f}" if fv is not None else "",
                        f"{rv/LN2:.4f}" if rv is not None else ""])

    # best
    fb_s = min(fwd, key=fwd.get); rb_s = min(rev, key=rev.get)
    print(f"forward  best val: {fwd[fb_s]:.4f} nats = {fwd[fb_s]/LN2:.4f} bpc  @ step {fb_s}")
    print(f"reversed best val: {rev[rb_s]:.4f} nats = {rev[rb_s]/LN2:.4f} bpc  @ step {rb_s}")
    print(f"final-step gap: {abs(fwd[max(fwd)]-rev[max(rev)]):.4f} nats")

    # compact table every 1000 steps
    print("\nstep   fwd_bpc  rev_bpc   d_bpc")
    for s in steps:
        if s % 1000 == 0 and s in fwd and s in rev:
            print(f"{s:5d}  {fwd[s]/LN2:7.4f}  {rev[s]/LN2:7.4f}  {(rev[s]-fwd[s])/LN2:+.4f}")

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot([s for s in steps if s in fwd], [fwd[s]/LN2 for s in steps if s in fwd],
             label="forward", lw=2)
    plt.plot([s for s in steps if s in rev], [rev[s]/LN2 for s in steps if s in rev],
             label="reversed (within-split)", lw=2, ls="--")
    plt.xlabel("iteration")
    plt.ylabel("validation loss (bits per character)")
    plt.title("Reversed-text pilot: forward vs reversed char model (6L/768, 10K iters)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/revtest_pilot_forward_vs_reversed_bpc.png", dpi=130)
    print("\nwrote plots/revtest_pilot_forward_vs_reversed_bpc.png")
    print("wrote plots/revtest_pilot_val_bpc.csv")


if __name__ == "__main__":
    main()
