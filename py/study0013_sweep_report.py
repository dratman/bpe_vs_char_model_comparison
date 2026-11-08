"""coupler-queue 0013 part 3: aggregate the LR sweep -> table, jsonl, plot, best rates, Check B."""
import json, math, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'pt/study0013/sweep'
LRS = ['0.00025', '0.0005', '0.001', '0.002', '0.004']
LENS = [3050, 12200]
wall = {}
for l in open(f'{R}/runs.log'):
    m = re.match(r'(\S+) wall_seconds=(\d+)', l)
    if m:
        wall[m.group(1)] = int(m.group(2))

rows = []
for tk in ('bpe', 'char'):
    for n in LENS:
        for lr in LRS:
            name = f'{tk}_{n}_{lr}'
            sj = f'{R}/{name}/run_summary.json'
            if not os.path.exists(sj):
                continue
            s = json.load(open(sj))
            probes = [json.loads(x) for x in open(f'{R}/{name}/probe_log.jsonl')]
            last = probes[-1]
            rows.append(dict(tokenizer=tk, length=n, lr=float(lr), name=name, final_val_bpc=s['final_val_bpc'],
                             min_val_bpc=s['min_val_bpc'], final_probe_prose_bpc=last['bpc']['prose'],
                             final_probe_step=last['step'], steps_done=s['steps_done'], diverged=s['diverged'],
                             diverged_reason=s['diverged_reason'], wall_seconds=wall.get(name),
                             train_seconds=s['train_seconds']))
with open('doc/0013_lr_sweep.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')

print('| tokenizer | length | lr | final val bpc | final probe prose bpc | diverged | wall-clock |')
print('|---|---|---|---|---|---|---|')
for r in rows:
    w = r['wall_seconds']
    fv = 'n/a' if r['final_val_bpc'] is None else f"{r['final_val_bpc']:.4f}"
    print(f"| {r['tokenizer']} | {r['length']} | {r['lr']:g} | {fv} | {r['final_probe_prose_bpc']:.4f} | "
          f"{'YES (' + r['diverged_reason'] + ')' if r['diverged'] else 'no'} | {w//60}m{w%60:02d}s |")

best = {}
for tk in ('bpe', 'char'):
    for n in LENS:
        ok = [r for r in rows if r['tokenizer'] == tk and r['length'] == n and not r['diverged'] and r['final_val_bpc'] is not None]
        if ok:
            b = min(ok, key=lambda r: r['final_val_bpc'])
            best[(tk, n)] = b
            edge = ' (EDGE OF GRID)' if b['lr'] in (0.00025, 0.004) else ''
            print(f"best {tk} {n}: lr {b['lr']:g}  val bpc {b['final_val_bpc']:.4f}{edge}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
sty = {('bpe', 3050): ('#1f77b4', '--'), ('bpe', 12200): ('#1f77b4', '-'),
       ('char', 3050): ('#d95f02', '--'), ('char', 12200): ('#d95f02', '-')}
for (tk, n), (c, ls) in sty.items():
    pts = sorted((r['lr'], r['final_val_bpc']) for r in rows
                 if r['tokenizer'] == tk and r['length'] == n and not r['diverged'] and r['final_val_bpc'] is not None)
    if pts:
        ax.plot(*zip(*pts), marker='o', color=c, ls=ls, label=f"{tk}, {n} steps ({n/610:.0f}% of a run)".replace('(5', '(').replace(' ', ' ', 1))
    for r in rows:
        if r['tokenizer'] == tk and r['length'] == n and r['diverged']:
            ax.plot(r['lr'], ax.get_ylim()[1], marker='x', color=c, ls='none')
ax.set_xscale('log')
ax.set_xlabel('peak learning rate')
ax.set_ylabel('final held-out val bits per character')
ax.set_title('Learning-rate sweep: 8 layers, width 512 (x = diverged)')
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('doc/0013_lr_sweep.png', dpi=150)

if ('bpe', 12200) in best and ('char', 12200) in best:
    b, c = best[('bpe', 12200)]['lr'], best[('char', 12200)]['lr']
    ratio = max(b, c) / min(b, c)
    gm = math.sqrt(b * c)
    print(f"20% optima: bpe {b:g}, char {c:g}; ratio {ratio:g}; geometric mean {gm:.5f}")
    if ratio <= 2:
        cand = min((float(x) for x in LRS), key=lambda x: abs(math.log(x / gm)))
        print(f"decision rule: within a factor of 2 -> candidate shared rate {cand:g}")
    else:
        cand = None
        print("decision rule: optima differ by more than 2x -> run the main comparison at both rates")
    # Check B at the candidate (or, if none, at each tokenizer's own best and the shared grid rate nearest gm)
    cb = cand if cand is not None else min((float(x) for x in LRS), key=lambda x: abs(math.log(x / gm)))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    series = {}
    for tk, ax in zip(('bpe', 'char'), axs):
        name = f'{tk}_12200_{cb:g}'
        recs = [json.loads(x) for x in open(f'{R}/{name}/probe_log.jsonl')]
        recs = [r for r in recs if r['update_rel'] is not None]
        steps = [r['step'] for r in recs]
        L = len(recs[0]['update_rel'])
        for li in range(L):
            ax.plot(steps, [r['update_rel'][li] for r in recs], label=f'layer {li}', color=plt.cm.viridis(li / (L - 1)))
        ax.set_title(f'{tk}, lr {cb:g}')
        ax.set_xlabel('step')
        series[tk] = {r['step']: r['update_rel'] for r in recs}
    axs[0].set_ylabel('relative update size ||dW||/||W||, last step')
    axs[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig('doc/0013_checkB_update_size.png', dpi=150)
    print(f"\nCheck B at lr {cb:g} (20% length): per-layer mean relative update, ratio char/bpe by phase")
    common = sorted(set(series['bpe']) & set(series['char']))
    for lo, hi, lab in ((0, 610, 'warmup (<610)'), (610, 4000, '610-4000'), (4000, 12201, '4000-end')):
        st = [s for s in common if lo <= s < hi]
        if not st:
            continue
        rat = [[series['char'][s][li] / series['bpe'][s][li] for s in st] for li in range(L)]
        print(f"  {lab:14s} steps {len(st):3d}: per-layer median char/bpe = " + ' '.join(f"{sorted(x)[len(x)//2]:.2f}" for x in rat))
    with open('doc/0013_checkB_update_size.json', 'w') as f:
        json.dump(dict(lr=cb, bpe=series['bpe'], char=series['char']), f)
