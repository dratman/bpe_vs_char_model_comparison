"""
coupler-queue 0012, steps 1-2: corpus facts, tokenizers, held-out region, probe batch.

  python3 py/study0012_prep.py --corpus txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
      --out_dir pt/study0012 --probe_out doc/0012_probe_batch.json
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import BPETokenizer, CharTokenizer  # noqa: E402
import study_data  # noqa: E402

PASSAGE_CHARS = 1024
MORPH_RE = re.compile(r"[a-z]+")
SUFFIXES = ('ing', 'ed', 's', 'es', 'ly', 'ness', 'tion')
PREFIXES = ('un', 're', 'dis')


def morph_score(text):
    n = 0
    for w in MORPH_RE.findall(text.lower()):
        if len(w) < 4:
            continue
        if w.endswith(SUFFIXES) or w.startswith(PREFIXES):
            n += 1
    return n


def mean_sentence_words(text):
    segs = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(segs) < 1:
        return 1e9, 0
    return sum(len(s.split()) for s in segs) / len(segs), len(segs)


def proselike(text):
    letters = sum(c.isalpha() or c == ' ' for c in text)
    return letters / len(text) >= 0.88 and text.count('\n') <= 8


def tile_windows(held, stride=PASSAGE_CHARS):
    """Non-overlapping 1024-char windows, each starting right after a space/newline."""
    pos, out = 0, []
    while pos + PASSAGE_CHARS <= len(held):
        # advance to a word start
        s = pos
        while s < len(held) and held[s - 1] not in ' \n' and s > 0:
            s += 1
        if s + PASSAGE_CHARS > len(held):
            break
        out.append(s)
        pos = s + stride
    return out


def make_repetition(words, rng):
    """Exactly 1024 chars: period P of exactly 512 chars (' w1 w2 ... wk'), twice."""
    while True:
        ws, n = [], 0
        while True:
            w = rng.choice(words)
            if n + 1 + len(w) <= 512:
                ws.append(w)
                n += 1 + len(w)
            else:
                break
        r = 512 - n
        if r == 0:
            pass
        else:
            cands = [w for w in words if len(w) == r - 1]
            if not cands:
                continue
            ws.append(rng.choice(cands))
        p = ' ' + ' '.join(ws)
        if len(p) == 512:
            return p * 2, 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out_dir', default='pt/study0012')
    ap.add_argument('--probe_out', default='doc/0012_probe_batch.json')
    ap.add_argument('--heldout_frac', type=float, default=0.01)
    ap.add_argument('--vocab_size', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=20261107)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    with open(args.corpus, encoding='utf-8') as f:
        text = f.read()
    cut = study_data.heldout_cut(len(text), args.heldout_frac)
    train_text, held = text[:cut], text[cut:]
    print(f"corpus {len(text):,} chars; cut offset {cut:,}; train {len(train_text):,}; held-out {len(held):,}")

    # ---- tokenizers (trained on the TRAINING text only) ----
    bpe_path = os.path.join(args.out_dir, f'bpe{args.vocab_size}.pkl')
    char_path = os.path.join(args.out_dir, 'char.pkl')
    if os.path.exists(bpe_path):
        bpe = BPETokenizer(args.vocab_size)
        bpe.load(bpe_path)
    else:
        bpe = BPETokenizer(vocab_size=args.vocab_size)
        bpe.train(train_text)
        bpe.save(bpe_path)
    chartok = CharTokenizer()
    chartok.train(train_text)
    chartok.save(char_path)
    missing = set(held) - set(chartok.stoi)
    print(f"char vocab size (train text): {chartok.vocab_size}; held-out chars not in vocab: {sorted(missing)}")

    # ---- probe batch ----
    tiles = tile_windows(held)
    cand = [(s, held[s:s + PASSAGE_CHARS]) for s in tiles]
    cand = [(s, t) for s, t in cand if proselike(t)]
    print(f"{len(tiles)} tiled windows, {len(cand)} prose-like")
    used = set()
    chosen = []

    def take(kind, ranked, n=16):
        k = 0
        for s, t in ranked:
            if s in used:
                continue
            used.add(s)
            chosen.append(dict(kind=kind, held_offset=s, corpus_offset=cut + s, text=t))
            k += 1
            if k == n:
                break

    # ordinary prose: random positions (not necessarily tile-aligned), non-overlapping
    prose = []
    tries = 0
    while len(prose) < 16 and tries < 100000:
        tries += 1
        s = rng.randrange(1, len(held) - PASSAGE_CHARS)
        while held[s - 1] not in ' \n':
            s += 1
        t = held[s:s + PASSAGE_CHARS]
        if len(t) == PASSAGE_CHARS and proselike(t) and all(abs(s - p[0]) >= PASSAGE_CHARS for p in prose):
            prose.append((s, t))
    for s, t in prose:
        chosen.append(dict(kind='prose', held_offset=s, corpus_offset=cut + s, text=t))
    prose_spans = [(s, s + PASSAGE_CHARS) for s, _ in prose]

    def overlaps_prose(s):
        return any(s < b and a < s + PASSAGE_CHARS for a, b in prose_spans)

    cand2 = [(s, t) for s, t in cand if not overlaps_prose(s)]
    morph = sorted(cand2, key=lambda st: -morph_score(st[1]))
    take('morphology', morph)
    short = []
    for s, t in cand2:
        m, nseg = mean_sentence_words(t)
        if nseg >= 12:
            short.append((m, s, t))
    short.sort()
    take('short_sentence', [(s, t) for m, s, t in short])
    for c in chosen:
        t = c['text']
        c['morph_score'] = morph_score(t)
        c['mean_sentence_words'] = round(mean_sentence_words(t)[0], 2)

    # repetition passages: uniform draw from the 1000 most common corpus words (training text sample)
    cnt = collections.Counter(re.findall(r"[a-z]{2,12}", train_text[:50_000_000].lower()))
    common = [w for w, _ in cnt.most_common(1000)]
    for i in range(16):
        t, off = make_repetition(common, rng)
        assert len(t) == 1024 and t[:512] == t[512:]
        chosen.append(dict(kind='repetition', held_offset=None, corpus_offset=None,
                           repeat_offset=off, text=t))
    assert len(chosen) == 64 and all(len(c['text']) == PASSAGE_CHARS for c in chosen)

    # ---- round trip ----
    bad = []
    for i, c in enumerate(chosen):
        if bpe.decode(bpe.encode(c['text'])) != c['text']:
            bad.append(('bpe', i))
        if chartok.decode(chartok.encode(c['text'])) != c['text']:
            bad.append(('char', i))
    print("round-trip failures:", bad)

    # repetition: token-level alignment check
    for c in chosen:
        if c['kind'] != 'repetition':
            continue
        enc = bpe.tokenizer.encode(c['text'])
        offs = enc.offsets
        k = next(i for i, (a, b) in enumerate(offs) if a >= 512)
        n = len(enc.ids)
        c['bpe_repeat_token_index'] = k
        c['bpe_aligned'] = bool(n == 2 * k and enc.ids[:k] == enc.ids[k:])
    print("BPE repetition passages aligned:", sum(c.get('bpe_aligned', False) for c in chosen if c['kind'] == 'repetition'), "/ 16")

    meta = dict(corpus=os.path.abspath(args.corpus), corpus_chars=len(text), heldout_frac=args.heldout_frac,
                heldout_cut_offset=cut, seed=args.seed, passage_chars=PASSAGE_CHARS,
                bpe_vocab_size=bpe.vocab_size,
                kinds={k: sum(c['kind'] == k for c in chosen) for k in ('prose', 'morphology', 'short_sentence', 'repetition')},
                scoring=dict(
                    morphology="count of lowercase words (len>=4) ending in -ing/-ed/-s/-es/-ly/-ness/-tion or starting un-/re-/dis-, per 1024-char window; top 16 non-overlapping tiled windows",
                    short_sentence="lowest mean sentence length in words (split on . ! ?), among windows with >=12 sentences",
                    prose="16 random non-overlapping windows starting at a word start",
                    repetition="period of exactly 512 chars ' w1 w2 .. wk' (words uniform from the 1000 most common words of the training text), repeated twice"),
                filter="windows must be >=88% letters+spaces and <=8 newlines (except repetition)")
    os.makedirs(os.path.dirname(args.probe_out) or '.', exist_ok=True)
    with open(args.probe_out, 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, passages=chosen), f, indent=1, ensure_ascii=False)
    print("probe sha256:", study_data.sha256_file(args.probe_out))

    # ---- step 1 facts: chars per BPE token, on the full training text (via the cache) ----
    tr, ho, info_b = study_data.load_or_build_tokens(text, args.corpus, bpe, bpe_path, args.out_dir, args.heldout_frac)
    tr2, ho2, info_c = study_data.load_or_build_tokens(text, args.corpus, chartok, char_path, args.out_dir, args.heldout_frac)
    ratio = info_b['train_chars'] / info_b['train_tokens']
    ratio_h = info_b['heldout_chars'] / info_b['heldout_tokens']
    print(f"BPE train: {info_b['train_tokens']:,} tokens; chars/token {ratio:.4f}; held-out chars/token {ratio_h:.4f}")
    print(f"char context = 256*ratio = {256 * ratio:.1f} -> nearest multiple of 64: {round(256 * ratio / 64) * 64}")
    print(f"char train tokens {info_c['train_tokens']:,}")
    print("bpe sha256:", study_data.sha256_file(bpe_path.replace('.pkl', '.json')), " pkl:", study_data.sha256_file(bpe_path))
    print("char sha256 (pkl):", study_data.sha256_file(char_path))


if __name__ == '__main__':
    main()
