#!/usr/bin/env python3
"""Analyze the full-length reversed fp32-attention run (coupler-queue 0003 re-run):
build the val curve in bits/char, plot it against the forward floor, and extract
early/mid/late samples re-reversed to normal reading order."""
import re, math, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2.0)
FWD_FLOOR_NATS = 0.7152          # forward char_uppercase_16L_1280 best val (per-char)
LOG = "terminal_logs/terminal_log_for_char_uppercase_16L_1280_reversed_fp32attn_cuda_2026_06_20_2331.txt"
PAT = re.compile(r"Step\s+(\d+)\s+\|.*\|\s*val loss\s+([0-9.]+)\s*\|")
NUM = re.compile(r"^  (\d)\. (.*)$")


def parse_series(path):
    s = {}
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                s[int(m.group(1))] = float(m.group(2))
    return s


def parse_blocks(path):
    out, cur = [], []
    with open(path) as f:
        for line in f:
            m = NUM.match(line.rstrip("\n"))
            if m:
                n = int(m.group(1))
                if n == 1 and cur:
                    out.append(cur); cur = []
                cur.append(m.group(2))
            elif cur and len(cur) >= 5:
                out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def main():
    s = parse_series(LOG)
    steps = sorted(s)
    best_it = min(s, key=s.get); best = s[best_it]
    print(f"reversed best val: {best:.4f} nats = {best/LN2:.4f} bpc  p={math.exp(-best):.4f}  @iter {best_it}")
    print(f"forward floor:     {FWD_FLOOR_NATS:.4f} nats = {FWD_FLOOR_NATS/LN2:.4f} bpc  p={math.exp(-FWD_FLOOR_NATS):.4f}")
    print(f"gap (reversed - forward): {best-FWD_FLOOR_NATS:+.4f} nats = {(best-FWD_FLOOR_NATS)/LN2:+.4f} bpc")

    with open("plots/reversed_fp32attn_val_bpc.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter", "val_nats", "val_bpc"])
        for it in steps:
            w.writerow([it, s[it], f"{s[it]/LN2:.4f}"])

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [s[it]/LN2 for it in steps], lw=1.5, label="reversed (fp32-attention)")
    plt.axhline(FWD_FLOOR_NATS/LN2, ls="--", color="k", lw=1,
                label=f"forward floor {FWD_FLOOR_NATS/LN2:.3f} bpc")
    plt.ylim(0.9, 2.2)
    plt.xlabel("iteration"); plt.ylabel("validation loss (bits/char)")
    plt.title("0003 reversed 16L/1280 (fp32-attention) vs forward floor")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("plots/reversed_fp32attn_val_bpc.png", dpi=130)

    blocks = parse_blocks(LOG)
    # 50 blocks @ sample_interval 10000 -> iters 10k..500k; pick early/mid/late
    picks = {"EARLY (~iter 10k)": 0, "MID (~iter 250k)": min(24, len(blocks)-1),
             "LATE (final, ~iter 500k)": len(blocks)-1}
    lines = []
    for label, idx in picks.items():
        lines.append(f"========== {label} ==========")
        lines.append("REVERSED sample 1, RE-REVERSED to normal reading order:")
        lines.append("  " + blocks[idx][0][::-1])
        lines.append("")
    with open("plots/reversed_fp32attn_samples.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote plots/reversed_fp32attn_{val_bpc.csv,val_bpc.png,samples.txt}")


if __name__ == "__main__":
    main()
