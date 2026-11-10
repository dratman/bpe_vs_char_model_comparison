"""coupler-queue 0018 helpers.  .venv/bin/python3 py/study0018_tools.py gate|line|table RUN_ROOT [runs...]"""
import glob, json, math, os, re, subprocess, sys
ROOT = sys.argv[2] if len(sys.argv) > 2 else '/data/study/0018'
THRESH = {'bpe': 1.2346, 'char': 1.2961}
STEPS = [int(t) for t in re.split(r'[,\s]+', open('doc/0012_snapshot_steps.txt').read().strip()) if t]
TK = {'B': 'bpe', 'C': 'char'}

def finite(o):
    if isinstance(o, dict): return all(finite(v) for v in o.values())
    if isinstance(o, list): return all(finite(v) for v in o)
    if isinstance(o, float): return math.isfinite(o)
    return True

def load(run):
    d = f'{ROOT}/{run}'
    s = json.load(open(d + '/run_summary.json'))
    recs = [json.loads(l) for l in open(d + '/probe_log.jsonl')]
    return d, s, recs

def gate_one(run):
    d, s, recs = load(run); tk = TK[run[0]]; msgs = []
    have = sorted(int(re.search(r'snap_step(\d+)_', f).group(1)) for f in glob.glob(d + '/snap_step*.pt'))
    if have != STEPS: msgs.append(f'snapshots {len(have)} != {len(STEPS)}')
    st = [r['step'] for r in recs]
    if st != list(range(0, 61001, 250)): msgs.append(f'probe steps wrong ({len(st)} lines)')
    if not all(finite({k: v for k, v in r.items() if k != 'update_rel'}) and (r['update_rel'] is None or finite(r['update_rel'])) for r in recs):
        msgs.append('non-finite probe field')
    if s['diverged']: msgs.append('diverged: ' + str(s['diverged_reason']))
    if s['steps_done'] != 61000: msgs.append(f"steps_done {s['steps_done']}")
    v = s['final_val_bpc']
    if not (v is not None and v < THRESH[tk]): msgs.append(f'final val bpc {v} not < {THRESH[tk]}')
    return msgs

def row(run):
    d, s, recs = load(run); last = recs[-1]
    ind = max(max(l) if isinstance(l, list) else l for l in last['induction'])
    du = subprocess.run(['du', '-sh', d], capture_output=True, text=True).stdout.split()[0]
    return dict(run=run, val=s['final_val_bpc'], bpc=last['bpc'], ind=ind, wall=s['train_seconds'], du=du, diverged=s['diverged'])

def fmt_wall(x): return f'{int(x//3600)}h{int(x%3600//60):02d}m'

cmd = sys.argv[1]
if cmd == 'gate':
    bad = {r: gate_one(r) for r in ('B1', 'C1')}
    print(json.dumps(bad)); sys.exit(1 if any(bad.values()) else 0)
elif cmd == 'line':
    r = row(sys.argv[3]); b = r['bpc']
    print(f"- {r['run']}: val bpc {r['val']:.4f}; probe bpc prose {b['prose']:.4f} / morphology {b['morphology']:.4f} / short_sentence {b['short_sentence']:.4f} / repetition {b['repetition']:.4f}; train wall-clock {fmt_wall(r['wall'])}")
elif cmd == 'table':
    rows = [row(r) for r in sys.argv[3:]]
    print('| run | val bpc | prose | morphology | short_sentence | repetition | max induction (final) | train wall-clock | disk |\n|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        b = r['bpc']; print(f"| {r['run']} | {r['val']:.4f} | {b['prose']:.4f} | {b['morphology']:.4f} | {b['short_sentence']:.4f} | {b['repetition']:.4f} | {r['ind']:.4f} | {fmt_wall(r['wall'])} | {r['du']} |")
