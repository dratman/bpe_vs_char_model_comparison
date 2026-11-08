"""coupler-queue 0013 part 2: PASS/FAIL report. Run with the .venv python from the repo root."""
import glob, json, math, os, re, sys
sys.path.insert(0, 'py')
from tokenizer import load_tokenizer
import study_probe

R = 'pt/study0013/part2'
P = 'doc/0013_probe_batch_v2.json'
steps = [int(t) for t in re.split(r'[,\s]+', open('doc/0012_snapshot_steps.txt').read().strip()) if t]
steps = [s for s in steps if s <= 500]

# 1. identical spans
rb = study_probe.ProbeRunner(P, load_tokenizer('pt/study0012/bpe8192.pkl'), 'cpu', max_input=256)
rc = study_probe.ProbeRunner(P, load_tokenizer('pt/study0012/char.pkl'), 'cpu', max_input=1003)
same = rb.n_chars == rc.n_chars
print(f"[{'PASS' if same else 'FAIL'}] per-passage scored-span length identical for BPE and char (64 passages)")
for k in ('prose', 'morphology', 'short_sentence', 'repetition'):
    L = [n for n, kk in zip(rb.n_chars, rb.kinds) if kk == k]
    print(f"    {k:15s} span chars min {min(L)} mean {sum(L)/len(L):.1f} max {max(L)}")
# spans start identical too: BPE scores chars after token 0; char scored from lo+1
ok = all(rc.lo[i] + 1 == rb.spans[i]['start'] for i in range(64))
print(f"[{'PASS' if ok else 'FAIL'}] char scoring starts at the same first character as BPE (after token 0)")

# 2. snapshots and probe logs
def finite(o):
    if isinstance(o, dict): return all(finite(v) for v in o.values())
    if isinstance(o, list): return all(finite(v) for v in o)
    if isinstance(o, float): return math.isfinite(o)
    return True
for tk in ('bpe', 'char'):
    d = f'{R}/A_{tk}'
    have = sorted(int(re.search(r'snap_step(\d+)_', f).group(1)) for f in glob.glob(d + '/snap_step*.pt'))
    print(f"[{'PASS' if have == steps else 'FAIL'}] {tk}: snapshots at all {len(steps)} scheduled steps <=500 (found {len(have)})")
    recs = [json.loads(l) for l in open(d + '/probe_log.jsonl')]
    st = [r['step'] for r in recs]
    # update_rel is None at step 0 by design
    fin = all(finite({k: v for k, v in r.items() if k != 'update_rel'}) and (r['update_rel'] is None or finite(r['update_rel'])) for r in recs)
    print(f"[{'PASS' if st == [0, 250, 500] and fin else 'FAIL'}] {tk}: probe log lines at {st}, all fields finite={fin}")

# 4. val loss repeat
def val500(f):
    m = re.findall(r'Final held-out val loss ([\d.]+) \(val bpc ([\d.]+)\)', open(f).read())
    return m[-1]
for tk in ('bpe', 'char'):
    a, c = val500(f'{R}/A_{tk}.log'), val500(f'{R}/C_{tk}.log')
    pa = [json.loads(l) for l in open(f'{R}/A_{tk}/probe_log.jsonl')][-1]['bpc']
    pc = [json.loads(l) for l in open(f'{R}/C_{tk}/probe_log.jsonl')][-1]['bpc']
    dp = max(abs(pa[k] - pc[k]) for k in pa)
    print(f"[{'PASS' if a == c else 'FAIL'}] {tk}: val loss/bpc at step 500, run A {a} vs run C {c} (printed 4/4 digits); max |diff| probe bpc A vs C {dp:.2e}")
