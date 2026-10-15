"""Build doc/0009_mps_probe_table.md and the A6000:Studio ratio tables from
the two probe jsonl files. Read-only over the jsonl; prints the ratio/estimate
tables to stdout for pasting into 0009.result.md. (coupler-queue 0009)"""
import json, os

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'doc')
MPS = os.path.join(DOC, '0009_mps_probe.jsonl')
CUDA = os.path.join(DOC, '0008_memory_probe.jsonl')
SIZES = ['S125', 'M350', 'L760']
MODES = ['fp32', 'bf16', 'bf16+fp32attn']


def load(path):
    rows = {}
    meta = []
    for line in open(path):
        r = json.loads(line)
        if r.get('kind'):
            meta.append(r)
            continue
        if r.get('vocab') != 8192:
            continue
        rows[(r['size'], r['mode'], r['mb'])] = r
    return rows, meta


def cell(r):
    if r is None:
        return '-'
    if r.get('status') == 'OOM':
        return 'OOM'
    if r.get('status') == 'timeout':
        return 'timeout'
    if r.get('status') != 'ok':
        return r.get('status', '?')
    return f"{r['peak_reserved_gib']:.1f}/{r['peak_alloc_gib']:.1f} / {r['tok_per_s']/1000:.1f}k / {r['tflops']:.1f}"


def main():
    mps, mmeta = load(MPS)
    cuda, _ = load(CUDA)
    mbs = [4, 8, 16, 32]

    lines = []
    lines.append("# 0009 Mac Studio (MPS) memory/throughput probe\n")
    v = next((m for m in mmeta if m.get('kind') == 'versions'), {})
    lines.append(f"torch {v.get('torch')}, macOS {v.get('macos')}, MPS available {v.get('mps_available')}. "
                 "Real py/model.py GPT, block 1024, vocab 8192, bias on, tied weights, dropout 0, "
                 "AdamW unfused (wd 0.1, lr 6e-4, betas 0.9/0.95), random token ids, 2 warm-up + 10 "
                 "timed steps, one fresh process per config. TFLOPS = 6 x params x tokens/s.\n")
    lines.append("Memory cells: **driver_allocated (pool high-water) / current_allocated (post-step)** GiB. "
                 "MPS has no peak-stats API; current understates (post-step), driver overstates (greedy pool). "
                 "Then tokens/s / TFLOPS.\n")
    lines.append("| size | mode | mb=4 | mb=8 | mb=16 | mb=32 |")
    lines.append("|---|---|---|---|---|---|")
    for size in SIZES:
        for mode in MODES + ['bf16native']:
            row = [cell(mps.get((size, mode, mb))) for mb in mbs]
            lines.append(f"| {size} | {mode} | " + " | ".join(row) + " |")
    table = "\n".join(lines)
    with open(os.path.join(DOC, '0009_mps_probe_table.md'), 'w') as f:
        f.write(table + "\n")
    print(table)

    # ---- ratio table: A6000 tok/s / Studio tok/s, matching cells (spec modes) ----
    print("\n\n### A6000 : Studio tokens/s ratio (per matching cell, vocab 8192)\n")
    print("| size | mode | mb=4 | mb=8 | mb=16 | mb=32 |")
    print("|---|---|---|---|---|---|")
    for size in SIZES:
        for mode in MODES:
            cells = []
            for mb in mbs:
                a = cuda.get((size, mode, mb))
                m = mps.get((size, mode, mb))
                if a and m and a.get('status') == 'ok' and m.get('status') == 'ok':
                    cells.append(f"{a['tok_per_s']/m['tok_per_s']:.2f}x")
                else:
                    cells.append('-')
            print(f"| {size} | {mode} | " + " | ".join(cells) + " |")

    # ---- best-setting ratio ----
    print("\n### Ratio at each machine's own best working setting (tokens/s)\n")
    print("| size | A6000 best (mode/mb -> tok/s) | Studio best (mode/mb -> tok/s) | A6000:Studio |")
    print("|---|---|---|---|")
    est = {}
    for size in SIZES:
        def best(rows):
            ok = [r for (s, mo, mb), r in rows.items()
                  if s == size and r.get('status') == 'ok']
            return max(ok, key=lambda r: r['tok_per_s']) if ok else None
        ba = best(cuda)
        bm = best(mps)
        est[size] = bm
        if ba and bm:
            print(f"| {size} | {ba['mode']}/{ba['mb']} -> {ba['tok_per_s']/1000:.1f}k | "
                  f"{bm['mode']}/{bm['mb']} -> {bm['tok_per_s']/1000:.1f}k | "
                  f"{ba['tok_per_s']/bm['tok_per_s']:.2f}x |")

    # ---- training-time estimates on Studio best working setting ----
    print("\n### Studio training-time estimates at its best working setting\n")
    print("(tokens / best Studio tok/s; ignores eval, data loading, LR schedule, grad-accum)\n")
    print("| size | tokens | Studio best | tok/s | wall-clock |")
    print("|---|---|---|---|---|")
    for size, toks in [('S125', 2.5e9), ('M350', 7e9), ('L760', 15e9)]:
        bm = est.get(size)
        if bm:
            hrs = toks / bm['tok_per_s'] / 3600
            days = hrs / 24
            tag = f"{hrs:.1f} h" + (f" ({days:.1f} d)" if hrs >= 48 else "")
            print(f"| {size} | {toks/1e9:.1f}B | {bm['mode']}/{bm['mb']} | {bm['tok_per_s']/1000:.1f}k | {tag} |")


if __name__ == '__main__':
    main()
