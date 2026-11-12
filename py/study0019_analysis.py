"""coupler-queue 0019: log-only analysis of the 0018 probe logs. No training.
  /home/owner/miniforge3/envs/bpe_char/bin/python py/study0019_analysis.py
Writes doc/0019_*.png, doc/0019_numbers.json, doc/0019_tables.md
"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
D = '/data/study/0018'
BPE = ['B1', 'B2', 'B3']; CHR = ['C1', 'C2', 'C3']; ALL = BPE + CHR + ['C4']
COL = {'bpe': '#1f77b4', 'char': '#d95f02'}
KINDS = ['prose', 'morphology', 'short_sentence', 'repetition']
OUT = os.path.join(ROOT, 'doc')

L = {r: [json.loads(l) for l in open(f'{D}/{r}/probe_log.jsonl')] for r in ALL}
for r in ALL: assert len(L[r]) == 245, r
steps = np.array([x['step'] for x in L['B1']])
chars = {r: np.array([x['chars_seen'] for x in L[r]], float) for r in ALL}
tab = []  # markdown lines
num = {}

def arr(r, f): return np.array([f(x) for x in L[r]], float)
def tok(r): return 'bpe' if r.startswith('B') else 'char'

def band(ax, runs, y, color, label, x=None):
    """mean line + min-max band over seeds (runs share steps; x = chars of first run (identical per tokenizer))."""
    Y = np.array([y[r] for r in runs]); X = chars[runs[0]]
    m = X > 0
    ax.fill_between(X[m], Y.min(0)[m], Y.max(0)[m], color=color, alpha=0.25, lw=0)
    ax.plot(X[m], Y.mean(0)[m], color=color, lw=1.6, label=label)

def c4(ax, y, color='#7a3300', label='char C4'):
    m = chars['C4'] > 0
    ax.plot(chars['C4'][m], y['C4'][m], color=color, lw=0.7, label=label)

def fmtc(c): return f'{c/1e6:.1f}M'

# chars seen identical across seeds within a tokenizer?
for g in (BPE, CHR):
    for r in g[1:]: assert np.allclose(chars[r], chars[g[0]]), 'chars differ'
for a, b in ((chars['C4'], chars['C1']),): assert np.allclose(a, b)
num['chars_seen_final'] = {'bpe': chars['B1'][-1], 'char': chars['C1'][-1]}

# ---------------- tokenizer statistics ----------------
from tokenizer import BPETokenizer
bpe = BPETokenizer(8192); bpe.load(os.path.join(ROOT, 'pt/study0012/bpe8192.pkl'))
batch = json.load(open(os.path.join(OUT, '0018_probe_batch_v3.json'), encoding='utf-8'))
spans = json.load(open(os.path.join(OUT, '0018_probe_batch_v3_spans.json')))['spans']
cpt = {}   # chars per BPE token over scored spans, per kind
for k in KINDS:
    ss = [s for s in spans if s['kind'] == k]
    cpt[k] = sum(s['length'] for s in ss) / sum(s['bpe_pred_tokens'] for s in ss)
ntk = {k: 1 / cpt[k] for k in KINDS}
cpt_nonrep = (sum(s['length'] for s in spans if s['kind'] != 'repetition') /
              sum(s['bpe_pred_tokens'] for s in spans if s['kind'] != 'repetition'))
num['bpe_chars_per_token_scored_span'] = cpt; num['bpe_tokens_per_char_scored_span'] = ntk
num['bpe_chars_per_token_nonrep'] = cpt_nonrep

def word_stats(texts):
    n = split = ntoks = 0; nch = 0
    for t in texts:
        for w in re.findall(r" ?[A-Za-z]+", t):
            e = bpe.tokenizer.encode(w).ids
            n += 1; split += len(e) >= 2; ntoks += len(e); nch += len(w)
    return dict(words=n, frac_split=split / n, tokens_per_word=ntoks / n, chars_per_token=nch / ntoks)
rep_texts = [p['text'][:512] for p in batch['passages'] if p['kind'] == 'repetition']
ws = {'repetition (random words, first 512 chars)': word_stats(rep_texts)}
for k in ('prose', 'morphology', 'short_sentence'):
    ws[k] = word_stats([p['text'] for p in batch['passages'] if p['kind'] == k])
ws['prose+morphology+short_sentence'] = word_stats([p['text'] for p in batch['passages'] if p['kind'] != 'repetition'])
# distinct random words only (each passage draws from 1000 most common words; count distinct vocabulary used)
dist = set(w for t in rep_texts for w in re.findall(r" ?[A-Za-z]+", t))
ws['repetition distinct words'] = dict(words=len(dist), frac_split=float(np.mean([len(bpe.tokenizer.encode(w).ids) >= 2 for w in dist])))
num['word_split_stats'] = ws

# ---------------- Analysis 1 ----------------
first = {r: arr(r, lambda x: x['rep_first_bpc']) for r in ALL}
second = {r: arr(r, lambda x: x['rep_second_bpc']) for r in ALL}
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for ax, (y, ttl) in zip(axs, ((first, 'first copy (512 random words, unseen)'), (second, 'second copy (exact repeat)'),
                              ({r: first[r] - second[r] for r in ALL}, 'first minus second (repetition improvement)'))):
    band(ax, BPE, y, COL['bpe'], 'BPE (B1-B3 mean, min-max band)'); band(ax, CHR, y, COL['char'], 'char (C1-C3 mean, min-max band)')
    c4(ax, y); ax.set_xscale('log'); ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('characters seen in training (log)'); ax.set_ylabel('bits per character (bpc)'); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.suptitle('0019 Analysis 1: repetition passages, first vs second copy (probe v3, 16 passages)')
fig.tight_layout(); fig.savefig(f'{OUT}/0019_repetition.png', dpi=130); plt.close(fig)

tab.append('### A1 final-step repetition bpc (step 61000)\n\n| run | first-copy bpc | second-copy bpc | first - second |\n|---|---|---|---|')
for r in ALL:
    tab.append(f'| {r} | {first[r][-1]:.4f} | {second[r][-1]:.4f} | {first[r][-1]-second[r][-1]:.4f} |')
def grp(runs, y): v = [y[r][-1] for r in runs]; return np.mean(v), min(v), max(v)
tab.append('\n| group | first (mean [min-max]) | second | diff |\n|---|---|---|---|')
for nm, runs in (('BPE B1-3', BPE), ('char C1-3', CHR)):
    f = grp(runs, first); s = grp(runs, second); d = grp(runs, {r: first[r] - second[r] for r in ALL})
    tab.append(f'| {nm} | {f[0]:.3f} [{f[1]:.3f}-{f[2]:.3f}] | {s[0]:.3f} [{s[1]:.3f}-{s[2]:.3f}] | {d[0]:.3f} [{d[1]:.3f}-{d[2]:.3f}] |')
    num[f'A1_final_{nm}'] = dict(first=f, second=s, diff=d)
# check consistency: logged repetition bpc vs mean of the two copies (if the same span)
num['A1_check_rep_bpc_vs_mean_of_copies_B1_final'] = (L['B1'][-1]['bpc']['repetition'], (first['B1'][-1] + second['B1'][-1]) / 2)
num['A1_check_rep_bpc_vs_mean_of_copies_C1_final'] = (L['C1'][-1]['bpc']['repetition'], (first['C1'][-1] + second['C1'][-1]) / 2)

tab.append('\n### A1 tokenizer statistics (probe v3)\n')
tab.append('BPE chars/token (scored spans): ' + ', '.join(f'{k} {cpt[k]:.3f} ({ntk[k]:.3f} tok/char)' for k in KINDS) + '\n')
tab.append('| text | words | fraction split into 2+ BPE tokens | BPE tokens/word | chars/token (words) |\n|---|---|---|---|---|')
for k, v in ws.items():
    tab.append(f"| {k} | {v['words']} | {v['frac_split']:.3f} | {v.get('tokens_per_word', float('nan')):.2f} | {v.get('chars_per_token', float('nan')):.2f} |")

# ---------------- Analysis 2 ----------------
IND = {r: np.array([x['induction'] for x in L[r]], float) for r in ALL}   # (245, 8, 8)
nh = {r: (IND[r] > 0.2).sum((1, 2)) for r in ALL}
def onset(r, thr=0.2, hold=4):
    A = IND[r].reshape(245, -1) > thr
    best = None
    for h in range(A.shape[1]):
        for i in range(245 - hold):
            if A[i:i + hold + 1, h].all():
                if best is None or i < best[0]: best = (i, h)
                break
    return best
imp = {r: first[r] - second[r] for r in ALL}
def steepest(r, w=5):
    y = np.convolve(imp[r], np.ones(w) / w, mode='valid')     # centered w-point moving mean
    d = np.diff(y); i = int(np.argmax(d)); return i + w // 2 + 1  # index of the log point at the end of the steepest rise
ons = {}
tab.append('\n### A2 induction onsets (head > 0.2 for 5 consecutive log points; steepest rise of 5-pt-smoothed first-second bpc)\n')
tab.append('| run | onset step | onset chars seen | onset head (layer.head) | steepest-rise step | steepest chars seen | heads>0.2 at final | layers of those heads |\n|---|---|---|---|---|---|---|---|')
for r in ALL:
    o = onset(r); s = steepest(r)
    fin = np.argwhere(IND[r][-1] > 0.2)
    layers = sorted(set(int(a) for a in fin[:, 0]))
    lcount = {int(l): int((fin[:, 0] == l).sum()) for l in layers}
    ons[r] = dict(onset_idx=None if o is None else o[0], onset_step=None if o is None else int(steps[o[0]]),
                  onset_chars=None if o is None else float(chars[r][o[0]]), head=None if o is None else (o[1] // 8, o[1] % 8),
                  steep_step=int(steps[s]), steep_chars=float(chars[r][s]), n_final=int(len(fin)), layers_final=lcount)
    q = ons[r]
    tab.append(f"| {r} | {q['onset_step']} | {fmtc(q['onset_chars']) if q['onset_chars'] else '-'} | {q['head'][0]}.{q['head'][1]} | {q['steep_step']} | {fmtc(q['steep_chars'])} | {q['n_final']} | {lcount} |")
num['A2'] = ons
tab.append('\n| group | onset chars (mean [min-max]) | steepest-rise chars (mean [min-max]) |\n|---|---|---|')
for nm, runs in (('BPE B1-3', BPE), ('char C1-3', CHR)):
    o = [ons[r]['onset_chars'] for r in runs]; s = [ons[r]['steep_chars'] for r in runs]
    tab.append(f'| {nm} | {fmtc(np.mean(o))} [{fmtc(min(o))}-{fmtc(max(o))}] | {fmtc(np.mean(s))} [{fmtc(min(s))}-{fmtc(max(s))}] |')

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, runs, tk in ((axs[0], BPE, 'bpe'), (axs[1], CHR, 'char')):
    band(ax, runs, nh, COL[tk], f'{tk} mean (B/C1-3), band = min-max')
    c4(ax, nh) if tk == 'char' else None
    for r in runs:
        if ons[r]['onset_chars']:
            ax.axvline(ons[r]['onset_chars'], color=COL[tk], ls=':', lw=1)
    if tk == 'char' and ons['C4']['onset_chars']: ax.axvline(ons['C4']['onset_chars'], color='#7a3300', ls=':', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('characters seen in training (log)'); ax.set_title(f'{tk}: heads with induction score > 0.2 (dotted = onset per run)', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper left')
axs[0].set_ylabel('number of heads (of 64) above 0.2')
fig.tight_layout(); fig.savefig(f'{OUT}/0019_induction.png', dpi=130); plt.close(fig)

# ---------------- Analysis 3 ----------------
NR = ['prose', 'morphology', 'short_sentence']
EFF = {r: np.mean([np.array([x['attn_eff_n'][k] for x in L[r]], float) for k in NR], 0) for r in ALL}   # (245,8,8) unweighted mean over 3 kinds
CS = {r: EFF[r] * (cpt_nonrep if r.startswith('B') else 1.0) for r in ALL}
def phase(r):
    p = EFF[r]; c = CS[r]
    single = p < 1.5
    avg = (~single) & (c > 40)
    clus = (~single) & (c <= 40)
    return np.stack([single, clus, avg], -1)      # (245,8,8,3)
PH = {r: phase(r) for r in ALL}
share = {r: PH[r].mean(2) for r in ALL}   # (245, layer, 3)
names = ['single-pattern', 'cluster', 'averaging']; pcol = ['#4c4c4c', '#2ca02c', '#c9c9c9']
fig, axs = plt.subplots(2, 8, figsize=(26, 7), sharey=True)
for row, (runs, tk) in enumerate(((BPE, 'BPE'), (CHR, 'char'))):
    X = chars[runs[0]]; m = X > 0
    for l in range(8):
        S = np.mean([share[r][:, l, :] for r in runs], 0)
        ax = axs[row, l]
        ax.stackplot(X[m], S[m].T, colors=pcol, labels=names, lw=0)
        ax.set_xscale('log'); ax.set_ylim(0, 1); ax.set_title(f'{tk}  layer {l}', fontsize=9)
        ax.set_xlabel('chars seen (log)', fontsize=8); ax.tick_params(labelsize=7)
        if l == 0: ax.set_ylabel('share of the 8 heads')
axs[0, 0].legend(fontsize=7, loc='lower left')
fig.suptitle('0019 Analysis 3: attention phases per layer (mean of 3 seeds; prose+morphology+short-sentence passages; single <1.5 positions, cluster <=40 chars spanned, averaging >40 chars)')
fig.tight_layout(); fig.savefig(f'{OUT}/0019_phases.png', dpi=110); plt.close(fig)

tab.append('\n### A3 final-step phase shares per layer (mean [min-max over 3 seeds]; C4 listed separately)\n')
tab.append('| layer | tok | single | cluster | averaging |\n|---|---|---|---|---|')
A3 = {}
for l in range(8):
    for nm, runs in (('BPE', BPE), ('char', CHR)):
        v = np.array([share[r][-1, l, :] for r in runs])
        A3[(l, nm)] = v
        tab.append(f'| {l} | {nm} | ' + ' | '.join(f'{v[:, j].mean():.2f} [{v[:, j].min():.2f}-{v[:, j].max():.2f}]' for j in range(3)) + ' |')
    tab.append(f"| {l} | C4 | " + ' | '.join(f'{share["C4"][-1, l, j]:.2f}' for j in range(3)) + ' |')
tab.append('\nEarly layers (0-3 pooled, cluster share of heads), per seed: ')
early = {}
for nm, runs in (('BPE', BPE), ('char', CHR + ['C4'])):
    early[nm] = [float(share[r][-1, :4, 1].mean()) for r in runs]
    tab.append(f'- {nm}: {[round(x, 3) for x in early[nm]]}')
late = {nm: [float(share[r][-1, 4:, 1].mean()) for r in runs] for nm, runs in (('BPE', BPE), ('char', CHR + ['C4']))}
tab.append(f'- late layers 4-7 cluster share: BPE {[round(x,3) for x in late["BPE"]]}, char {[round(x,3) for x in late["char"]]}')
num['A3_early_cluster'] = early; num['A3_late_cluster'] = late
# final eff-n summary
num['A3_final_effn_positions_median_by_layer'] = {tk: [float(np.median(np.mean([EFF[r][-1, l, :] for r in runs], 0))) for l in range(8)]
                                                  for tk, runs in (('BPE', BPE), ('char', CHR))}
num['A3_final_chars_spanned_median_by_layer'] = {tk: [float(np.median(np.mean([CS[r][-1, l, :] for r in runs], 0))) for l in range(8)]
                                                 for tk, runs in (('BPE', BPE), ('char', CHR))}

# ---------------- Analysis 4 ----------------
BP = {k: {r: arr(r, lambda x, k=k: x['bpc'][k]) for r in ALL} for k in KINDS}
fig, axs = plt.subplots(1, 4, figsize=(22, 5))
for ax, k in zip(axs, KINDS):
    band(ax, BPE, BP[k], COL['bpe'], 'BPE (mean, min-max band)'); band(ax, CHR, BP[k], COL['char'], 'char (mean, min-max band)')
    c4(ax, BP[k]); ax.set_xscale('log'); ax.set_title(k, fontsize=10); ax.set_xlabel('characters seen in training (log)')
    ax.set_ylabel('probe bits per character (bpc)'); ax.set_ylim(top=3.5 if k != 'repetition' else None); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.suptitle('0019 Analysis 4: probe bpc by passage kind (probe v3)'); fig.tight_layout()
fig.savefig(f'{OUT}/0019_curves.png', dpi=120); plt.close(fig)
# second zoomed version, late training
fig, axs = plt.subplots(1, 4, figsize=(22, 5))
for ax, k in zip(axs, KINDS):
    band(ax, BPE, BP[k], COL['bpe'], 'BPE'); band(ax, CHR, BP[k], COL['char'], 'char'); c4(ax, BP[k])
    ax.set_xscale('log'); ax.set_xlim(2e7, 2.2e9); lo = min(BP[k][r][60:].min() for r in ALL); ax.set_ylim(lo - 0.02, lo + (0.9 if k == 'repetition' else 0.5))
    ax.set_title(k + ' (zoom)', fontsize=10); ax.set_xlabel('characters seen in training (log)'); ax.set_ylabel('probe bpc'); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{OUT}/0019_curves_zoom.png', dpi=120); plt.close(fig)

tab.append('\n### A4 BPE-vs-char comparison by kind (mean over seeds; gap = BPE - char, negative = BPE ahead)\n')
tab.append('| kind | BPE ahead at first log point (chars seen) | ever behind? | permanently ahead from (chars) | fraction of log points BPE ahead | final gap |\n|---|---|---|---|---|---|')
for k in KINDS:
    gap = np.mean([BP[k][r] for r in BPE], 0) - np.mean([BP[k][r] for r in CHR], 0)
    X = chars['B1']; ahead = gap < 0
    ahead_idx = np.where(ahead[1:])[0] + 1
    first_ahead = fmtc(X[ahead_idx[0]]) if len(ahead_idx) else 'never'
    behind = np.where(~ahead)[0]
    perm = fmtc(X[behind[-1] + 1]) if len(behind) and behind[-1] + 1 < 245 else ('start' if not len(behind) else 'never')
    cross = int((np.diff(ahead.astype(int)) != 0).sum())
    num[f'A4_{k}'] = dict(first_ahead=first_ahead, perm=perm, frac_ahead=float(ahead[1:].mean()), crossings=cross, final_gap=float(gap[-1]),
                          gap_at=dict((fmtc(X[i]), round(float(gap[i]), 3)) for i in (1, 5, 10, 20, 40, 80, 120, 160, 200, 244)))
    tab.append(f'| {k} | {first_ahead} | {"yes" if (~ahead[1:]).any() else "no"} | {perm} | {ahead[1:].mean():.2f} | {gap[-1]:+.3f} |')
    tab.append('') if False else None
tab.append('\nMean gap (BPE - char) at selected chars-seen points: ')
for k in KINDS: tab.append(f'- {k}: {num["A4_"+k]["gap_at"]}  (sign changes: {num["A4_"+k]["crossings"]})')

open(f'{OUT}/0019_tables.md', 'w').write('\n'.join(tab) + '\n')
json.dump(num, open(f'{OUT}/0019_numbers.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('\n'.join(tab))
