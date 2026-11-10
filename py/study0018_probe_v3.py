"""coupler-queue 0018 part 1: probe batch v3. Stage 'cands' lists ranked candidates; stage 'build' writes v3 from chosen held_offsets.
  .venv/bin/python3 py/study0018_probe_v3.py cands [N]
  .venv/bin/python3 py/study0018_probe_v3.py build off1 off2 ... (16 held_offsets, in order)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_data, study0012_prep as prep  # noqa
import study0013_probe_v2 as v2  # noqa
from tokenizer import BPETokenizer  # noqa

V1 = 'doc/0012_probe_batch.json'; V2 = 'doc/0013_probe_batch_v2.json'
CORPUS = 'txt_local/corpus_high_quality_uppercase_2026_05_08.txt'
OUT = 'doc/0018_probe_batch_v3.json'


def extra_ok(t):
    if len(re.findall(r"\.{4,}", t)) >= 2: return False
    if t.count('--') > 2: return False
    if len(re.findall(r"\b[A-Z]{3,}\s*,", t)) >= 3: return False
    return True


def candidates():
    v1 = json.load(open(V1, encoding='utf-8'))
    keep = [p for p in v1['passages'] if p['kind'] != 'short_sentence']
    cut = v1['meta']['heldout_cut_offset']
    text = open(CORPUS, encoding='utf-8').read()
    held = text[cut:]
    taken = [(p['held_offset'], p['held_offset'] + 1024) for p in keep if p['held_offset'] is not None]
    cands = []
    for s in prep.tile_windows(held):
        t = held[s:s + 1024]
        if not prep.proselike(t) or any(s < b and a < s + 1024 for a, b in taken): continue
        ok, _ = v2.prose_like_not_list(t, held[s - 1] == '\n', (s + 1024 < len(held) and held[s + 1024] == '\n'))
        if not ok or not extra_ok(t): continue
        m, nseg = prep.mean_sentence_words(t)
        if nseg >= 12: cands.append((m, s, t))
    cands.sort()
    return v1, keep, cut, held, cands


if sys.argv[1] == 'cands':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    *_, cands = candidates()
    print(len(cands), "candidates")
    for r, (m, s, t) in enumerate(cands[:n]):
        print(f"\n=== rank {r} offset {s} meanSent {m:.2f}\n{t[:700]}\n  [...]\n{t[-250:]}")
else:
    offs = [int(x) for x in sys.argv[2:]]
    assert len(offs) == 16 and all(abs(a - b) >= 1024 for i, a in enumerate(offs) for b in offs[:i])
    v1, keep, cut, held, cands = candidates()
    cd = {s: (m, t) for m, s, t in cands}
    new_short = [dict(kind='short_sentence', held_offset=s, corpus_offset=cut + s, text=cd[s][1],
                      morph_score=prep.morph_score(cd[s][1]), mean_sentence_words=round(cd[s][0], 2)) for s in offs]
    passages = ([p for p in keep if p['kind'] in ('prose', 'morphology')] + new_short +
                [p for p in keep if p['kind'] == 'repetition'])
    meta = dict(v1['meta']); meta['version'] = 3
    meta['derived_from'] = V2
    meta['short_sentence_v3'] = ("v2 rules plus: reject >=2 runs of 4+ periods, >2 '--', >=3 all-caps 3+-letter words followed by comma; "
                                 "then hand-read in mean-sentence-length order, keeping running prose/dialogue only")
    json.dump(dict(meta=meta, passages=passages), open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    sha = study_data.sha256_file(OUT); print("probe v3 sha256:", sha)
    bpe = BPETokenizer(8192); bpe.load('pt/study0012/bpe8192.pkl')
    spans = []
    for i, p in enumerate(passages):
        t = p['text']; offs_ = bpe.tokenizer.encode(t).offsets
        assert all(offs_[j][1] == offs_[j + 1][0] for j in range(len(offs_) - 1))
        m = min(len(offs_) - 1, 256)
        while offs_[m][1] > 1003: m -= 1
        s, e = offs_[1][0], offs_[m][1]
        spans.append(dict(index=i, kind=p['kind'], start=s, end=e, length=e - s, bpe_pred_tokens=m))
    json.dump(dict(meta=dict(bpe_ctx=256, char_ctx=1003, probe_sha256=sha,
               rule="start = end of BPE token 0; end = end of last BPE token predicted inside the BPE window (<=256 predictions) and ending within 1003 characters"),
               spans=spans), open(OUT[:-5] + '_spans.json', 'w'), indent=1)
    for k in ('prose', 'morphology', 'short_sentence', 'repetition'):
        L = [x['length'] for x in spans if x['kind'] == k]
        print(f"span length {k}: min {min(L)} mean {sum(L)/len(L):.1f} max {max(L)}")
    print("\nFIRST 150 CHARS OF EACH NEW PASSAGE:")
    for j, p in enumerate(new_short): print(f"[{j}] off={p['held_offset']}: {p['text'][:150]!r}")
