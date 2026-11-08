"""
coupler-queue 0013, part 1b/1c: probe batch v2 + shared scored spans.

  .venv/bin/python3 py/study0013_probe_v2.py

Copies the 48 prose / morphology / repetition passages of doc/0012_probe_batch.json unchanged, rebuilds
the 16 short-sentence passages from the held-out region with list-like windows rejected, writes
doc/0013_probe_batch_v2.json and the sidecar doc/0013_probe_batch_v2_spans.json (span start/end per passage,
the same for both tokenizers).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_data  # noqa: E402
import study0012_prep as prep  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

HEAD_RE = re.compile(r"^\s*(CHAPTER|Chapter|BOOK|PART|ACT|SCENE)\b|^\s*[IVXLCDM]+\.")
INITIAL_RE = re.compile(r"(?<![A-Za-z])[A-Z]\.")
QUOTE_CHARS = '"“”'


def sentences(text):
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]


def prose_like_not_list(text, start_at_line_start, end_at_line_end):
    """The 0013 rejection rules; returns (ok, reason)."""
    if len(INITIAL_RE.findall(text)) > 2:
        return False, 'initials'
    lines = text.split('\n')
    if any(HEAD_RE.match(l) for l in lines):
        return False, 'heading'
    if text.count(':') > 1:
        return False, 'colons'
    # short-line rule: judged on complete lines only (a window's first/last line is usually a partial one)
    full = list(lines)
    if not start_at_line_start:
        full = full[1:]
    if not end_at_line_end and full:
        full = full[:-1]
    full = [l for l in full if l.strip()]
    if full and sum(len(l) < 30 for l in full) / len(full) > 0.25:
        return False, 'short_lines'
    sents3 = [s for s in sentences(text) if len(s.split()) >= 3]
    if len(sents3) < 2:
        return False, 'few_sentences'
    if not (any(c in text for c in QUOTE_CHARS) or len(sents3) >= 8):
        return False, 'no_dialogue_or_8_sentences'
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', default='doc/0012_probe_batch.json')
    ap.add_argument('--corpus', default='txt_local/corpus_high_quality_uppercase_2026_05_08.txt')
    ap.add_argument('--bpe', default='pt/study0012/bpe8192.pkl')
    ap.add_argument('--out', default='doc/0013_probe_batch_v2.json')
    ap.add_argument('--bpe_ctx', type=int, default=256)
    ap.add_argument('--char_ctx', type=int, default=1003)
    args = ap.parse_args()

    v1 = json.load(open(args.v1, encoding='utf-8'))
    keep = [p for p in v1['passages'] if p['kind'] != 'short_sentence']
    assert len(keep) == 48
    cut = v1['meta']['heldout_cut_offset']
    with open(args.corpus, encoding='utf-8') as f:
        text = f.read()
    held = text[cut:]
    assert len(held) == len(text) - cut

    # candidate windows: the same non-overlapping tiles as v1, prose-like, not overlapping kept passages
    taken = [(p['held_offset'], p['held_offset'] + 1024) for p in keep if p['held_offset'] is not None]
    tiles = prep.tile_windows(held)
    rej = {}
    cands = []
    for s in tiles:
        t = held[s:s + 1024]
        if not prep.proselike(t) or any(s < b and a < s + 1024 for a, b in taken):
            continue
        ok, why = prose_like_not_list(t, held[s - 1] == '\n', (s + 1024 < len(held) and held[s + 1024] == '\n'))
        if not ok:
            rej[why] = rej.get(why, 0) + 1
            continue
        m, nseg = prep.mean_sentence_words(t)
        if nseg >= 12:
            cands.append((m, s, t))
    cands.sort()
    print(f"{len(tiles)} tiles; candidates passing: {len(cands)}; rejected by rule: {rej}")
    new_short = [dict(kind='short_sentence', held_offset=s, corpus_offset=cut + s, text=t,
                      morph_score=prep.morph_score(t), mean_sentence_words=round(m, 2))
                 for m, s, t in cands[:16]]
    assert len(new_short) == 16
    # keep v1's kind order: prose, morphology, short_sentence, repetition
    passages = ([p for p in keep if p['kind'] in ('prose', 'morphology')] + new_short +
                [p for p in keep if p['kind'] == 'repetition'])
    meta = dict(v1['meta'])
    meta['version'] = 2
    meta['derived_from'] = 'doc/0012_probe_batch.json (sha256 419289c10a0830ca52ceddcd9740e50c6d5d9fbe1da8e3cfac76fea97596d7ca)'
    meta['short_sentence_v2'] = ("lowest mean sentence length among windows with >=12 sentences, after rejecting windows with "
                                 ">2 'X.' initials, a CHAPTER/BOOK/PART/ACT/SCENE/roman-numeral line, >1 colon, >25% short (<30 char) "
                                 "complete lines, <2 sentences of >=3 words; then requiring a quotation mark or >=8 sentences of >=3 words")
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, passages=passages), f, indent=1, ensure_ascii=False)
    sha = study_data.sha256_file(args.out)
    print("probe v2 sha256:", sha)

    # ---- shared scored spans ----
    bpe = BPETokenizer(8192)
    bpe.load(args.bpe)
    spans = []
    for i, p in enumerate(passages):
        t = p['text']
        enc = bpe.tokenizer.encode(t)
        offs = enc.offsets
        assert all(offs[j][1] == offs[j + 1][0] for j in range(len(offs) - 1)), f"non-contiguous offsets, passage {i}"
        assert offs[0][0] == 0 and offs[-1][1] == len(t)
        m = min(len(offs) - 1, args.bpe_ctx)               # predicted tokens 1..m
        while offs[m][1] > args.char_ctx:                  # last predicted token must end inside the char window
            m -= 1
        s, e = offs[1][0], offs[m][1]
        spans.append(dict(index=i, kind=p['kind'], start=s, end=e, length=e - s, bpe_pred_tokens=m))
    side = dict(meta=dict(bpe_ctx=args.bpe_ctx, char_ctx=args.char_ctx, probe_sha256=sha,
                          rule="start = end of BPE token 0; end = end of last BPE token that is predicted inside the BPE window "
                               "(<= bpe_ctx predictions) and ends within char_ctx characters"),
                spans=spans)
    sp_path = args.out[:-len('.json')] + '_spans.json'
    with open(sp_path, 'w') as f:
        json.dump(side, f, indent=1)
    print("spans ->", sp_path)
    for k in ('prose', 'morphology', 'short_sentence', 'repetition'):
        L = [x['length'] for x in spans if x['kind'] == k]
        print(f"span length {k}: min {min(L)} mean {sum(L)/len(L):.1f} max {max(L)}")
    print("\nFIRST 150 CHARS OF EACH NEW SHORT-SENTENCE PASSAGE:")
    for j, p in enumerate(new_short):
        print(f"[{j}] held_offset={p['held_offset']} meanSent={p['mean_sentence_words']}: {p['text'][:150]!r}")


if __name__ == '__main__':
    main()
