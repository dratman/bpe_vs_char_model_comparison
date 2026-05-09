# Diary 091 — Setting up a char vs BPE comparison: what is matchable across tokenizations

Date: 2026-05-08 / 2026-05-09

## What is now running

Two training runs are in flight on different machines, sharing the
same architecture (16 layers, 8 heads, n_embd=1280, block_size=4096),
the same corpus, the same case-preservation, the same precision, and
matched character-budget data exposure:

- **Studio** — char tokenizer (vocab=78), batch=4, lr=1.5e-4,
  500,000 iters, ~24 days projected. Launched 2026-05-09 00:38 EDT.
- **M3 laptop** — BPE tokenizer (vocab=32,000), batch=2, lr=1.06e-4,
  220,000 iters, ~17–18 days projected. Restarted at batch=2 after the
  first batch=4 attempt revealed memory pressure on the M3's 64 GB
  unified RAM.

Both share the same corpus:
`txt_local/corpus_high_quality_uppercase_2026_05_08.txt` — 1.27 GB,
3,979 books, 78-character vocabulary, document-shuffled with seed=42.

This is a more controlled char↔BPE comparison than diary 081's first
attempt, in two ways: (a) both runs use the *same* corpus and the same
case-preservation; (b) the two runs are matched on total characters
seen at completion (~8.2 B chars each), rather than on iters.

## How the corpus got rebuilt

The earlier corpus `corpus_high_quality_2026_04_26.txt` was lowercase
only. Rebuilding with case preserved required first untangling the
build pipeline. Found five real bugs in the corpus-build scripts:

1. `filter_corpus.py` had a 40-line `KEEP_SHELVES` set that was dead
   code — the function returned 'keep' verdicts that `main()` never
   acted on. The set was effectively documentation.
2. `clean_and_combine_corpus.py` accepted a different character set
   than `rebuild_corpus.py` (allowed `/`, the latter did not), so
   combining Gutenberg with Wikipedia would silently produce a different
   vocabulary than pure Gutenberg.
3. `rebuild_corpus.py`'s ASCII-folding stripped *all* non-decomposable
   Latin characters silently — `ß → ss` got dropped to `strae`,
   `Æsop → sop`, `Œuvre → uvre`. Fixed with an explicit transliteration
   map (ß, þ, ð, æ, œ, ø, ł and uppercase variants) applied before NFKD.
4. The English-detection function-word list omitted `i` (the pronoun).
   Fine for omniscient prose, but biased against first-person memoir.
5. `scan_corpus_quality.py` docstring said 10-second timeout; code
   uses 10-minute. Just a doc fix.

Then the scripts learned `--preserve_case`. Lower-case still produced
by default (preserves backward compatibility with existing corpora).
A `--manifest` flag now writes a TSV record of which book ended up at
which post-shuffle position.

## The Haiku-decisions detective work

The single non-obvious bit. The 4,430-book filter that produced
`corpus_high_quality_2026_04_26.txt` was done by submitting samples
to Claude Haiku for quality judging. Decisions were saved at
`doc/book_quality_decisions.json` — but the keys are integer indices
`'0'` through `'6197'` into `doc/all_book_samples.txt`, and the script
that did the indexing isn't in the repo. Without that mapping, the
keep list could not be rebuilt.

Recovered the mapping by fuzzy text matching. For each `=== BOOK N ===`
entry in `all_book_samples.txt`, take a 60-character signature from
the middle of the normalized text, build an Aho-Corasick automaton
over all 6,198 signatures, scan all 8,794 source `.txt` files. Strip
Project Gutenberg headers and footers from each source file before
matching, since PG license/donation/contact boilerplate produced
spurious matches (a single `Town_Geology.txt` initially absorbed 67
different book signatures, all from boilerplate). Demote any file that
absorbed multiple book indices — not trustworthy.

Recovered 5,486 clean 1-to-1 matches; from there, 3,981 of the 4,430
Haiku KEEP verdicts (89.9 %) became filenames. The other 449 were lost
to either no-source-file (250) or boilerplate-only-signature (199).

The new corpus has 3,979 books vs the original's 4,430. Slightly
smaller, fully case-preserved, and reproducible — manifest in
`corpus_manifest_2026_05_08.tsv` records the post-shuffle order with
char and paragraph counts per book.

## On Chinchilla for character models

Spent some thought on whether the 20:1 token-per-param ratio applies
to character-level training or whether it shifts. Conclusion: it
should apply *in character-tokens*, not in BPE-tokens — Chinchilla's
math is about FLOP/param, not about token semantics. So a 152 M char
model wants ~3.0 B character-tokens, and `char_high_quality` saw 4.1 B
(~27:1), basically right at the Chinchilla optimum.

Three caveats argue for treating 20:1 as a soft *lower* bound for
char models:

1. Hoffmann et al. measured on BPE web text. The optimum *constant*
   may shift for character data; not measured experimentally.
2. Per-token entropy differs by an order of magnitude between char
   and BPE. Char val loss ≈ 0.82 (P[correct] ≈ 0.44); BPE val loss ≈
   3.36 (P[correct] ≈ 0.035). Most char predictions are nearly free
   (within-word continuation). The *informative* gradient signal per
   token is much smaller for chars, suggesting an optimum closer to
   30–80:1 for char models.
3. `char_high_quality` showed train loss 0.79 vs val loss 0.82 at 3.2
   epochs (gap 0.03) — well below any overfitting threshold. The
   corpus had clearly more to give. So 27:1 was not the empirical
   ceiling; we don't know where that ceiling is.

The corollary: data is less of a hard ceiling than a strict 20:1
reading suggests. With 4–8 epochs of repetition (which empirical
work on similar literary corpora generally finds benign), the
practical model-size ceiling on this 1.27 B-char corpus sits around
400–500 M params before training quality really starts to suffer.
Bigger is possible — it just trades wall time for either marginal
gain or flat loss.

## Why we picked 16L / 8H / 1280

`char_high_quality` was 12L / 8H / 1024 = 152 M params. For the
case-preserved rebuild, scaling up was the obvious next move. Three
candidate scales were considered, with rough-projection wall times in
parens for an 11-day budget at batch=4 block=4096 bf16 on the Studio:

- 12L / 8H / 1280 = 240 M (~17 d) — bullseye on Chinchilla 22:1
- 16L / 8H / 1024 = 206 M (~15 d)
- 16L / 8H / 1280 = 320 M (~22 d) — slightly under-Chinchilla at 12.5:1
- 12L / 8H / 1536 = 343 M (~25 d)

Speed-tested two of these on the Studio at block=4096 bf16 with
batch=4, MPS:

- 12L / 8H / 1024 = 151 M: **1.92 sec/iter** (= same as `char_high_quality`
  at block=512 fp32, but 2× more tokens-per-iter; net 2× tokens/sec).
  MFU jumped 1.3 % → 4.0 %.
- 16L / 8H / 1280 = 320 M: **4.18 sec/iter**. MFU dipped slightly to
  3.5 % (extra layers cost some kernel-launch efficiency on MPS).

Once Ralph said wall time was not a concern, the data-bound argument
loosened: at 6 epochs the practical ceiling is ~400 M. We picked 16L /
8H / 1280 = 320 M, accepting the modest under-Chinchilla 12.5:1 ratio
on the strict reading because in adjusted (per-char-model) terms it
sits closer to optimal, and the existing depth comparison work in
diary 060 makes 16L the natural depth to study with continuity.

## A useful surprise: bf16 + larger block_size buys hardware utilization

The 12L / 8H / 1024 speed test showed the Studio's M2 Ultra MPS
running ~3× higher MFU at block=4096 bf16 than `char_high_quality` had
at block=512 fp32. PyTorch's `scaled_dot_product_attention` on MPS
uses a memory-efficient kernel that scales much better than the naive
O(T²) memory cost would suggest. The naive worry — that going from
block=512 to block=4096 would mean 64× more attention work per layer
— turned out to be moot, because larger sequences amortize per-kernel
overhead more efficiently and bf16 halves both compute and memory.

Net outcome for the comparison: 152M at block=4096 bf16 has *the same
tokens/sec* as 152M at block=512 fp32. We "spent" the savings on a
bigger model.

## On char↔BPE matching: batch_size is not directly meaningful

Realized partway through configuring the M3 BPE run that thinking of
"the same batch_size" as a fair-comparison knob across tokenizations
is incoherent. At the same `(batch_size=4, block_size=4096)`:

|  | char  | BPE  |
|---|---:|---:|
| length per sequence | 4,096 chars | 4,096 BPE-tokens (~18,432 chars) |
| chars per gradient update | 16,384 | ~73,728 |
| param updates per corpus epoch | 77,500 | 17,200 |

So at the same nominal `batch_size`, BPE is processing ~4.5× more
text per gradient step and consuming the corpus 4.5× faster per
update. There is no single "matched" notion of batch_size. The
question instead becomes: *what should be matched?*

For the "tokenization at fixed character budget" comparison, the
right matching is **total characters seen at completion**, not iters
or batch size or anything else. We have that:

- char run: 500K × 16,384 = **8.2 B chars** (= 6.4 epochs)
- BPE run (b4): 110K × 73,728 = **8.1 B chars** (= 6.4 epochs)
- BPE run (b2, currently in flight): 220K × 36,864 = **8.1 B chars**
  (= 6.4 epochs)

The choice of batch_size on each side then becomes a per-machine
stability decision, not a per-comparison decision. Once we decided
this, dropping BPE to batch=2 to reduce M3 memory pressure became a
non-event for the comparison's main metric.

## On char↔BPE matching: block_size also is not directly meaningful

A further realization, the same flavor: at the same `block_size=4096`,
the BPE model has roughly 4.5× longer effective context window in
characters. This is *intrinsic to BPE* — longer effective context per
fixed input length is one of the things the tokenization buys the
model. It's a feature of the comparison, not a confound.

If we wanted to *control out* the context-window difference, the
right move would be a *third* run with BPE block≈910 (= 4,096 / 4.5)
to match the char model's 4,096-char effective context. That's a
plausible follow-up. The current setup deliberately doesn't try — it
compares "char with 4 K-char context" vs "BPE with 18 K-char context"
at equal char budget, and accepts the context-length advantage as
part of what BPE is.

## Tooling improvements made along the way

- `py/train.py` startup banner now prints *every* CLI arg, the
  command line via `sys.argv`, and the hardcoded weight_decay / betas
  / grad_clip values. Logs from direct `python` invocations (no shell
  wrapper) are now self-describing. Earlier logs only printed a
  curated subset and silently omitted the architecture itself.
- `py/sample.py` auto-detects whether the loaded tokenizer is
  case-preserved (any uppercase letter in itos for char, any uppercase
  in vocab keys for BPE). When detected: skip the default
  prompt-lowercasing, skip the post-hoc `capitalize_sentences` pass.
  Both would silently degrade output for a case-preserved model.
- `py/match_book_samples.py` is new, recovers the
  index→filename mapping for `book_quality_decisions.json` so any
  future Haiku-quality re-application can produce a clean keep list.

## Open questions

1. Will the train/val gap on the Studio char run stay tight past
   epoch 4? `char_high_quality` showed 0.03 gap at 3.2 epochs; the new
   320 M model trains on a slightly smaller (3,979 vs 4,430 books)
   corpus. If the gap widens earlier than expected, that's the moment
   to consider stopping.
2. Will the BPE run on the M3 hold thermally for 17–18 days? Laptop
   chassis cooling is the unknown. The training-monitor LaunchAgent
   on the Studio will catch its case; the M3 doesn't have one
   installed and is only reachable by SSH. Plan to spot-check.
3. Eventually: a third run at BPE block≈910 to control out the
   context-length advantage. Not yet, but worth flagging.
4. The 449 books we lost in the matcher (PG-boilerplate signature
   collisions and missing source files). Multi-excerpt disambiguation
   could likely recover most of them. Small win. Bigger leverage:
   adding Wikipedia biographies via `clean_and_combine_corpus.py` to
   roughly double the corpus and unlock 800 M+ param models without
   data starvation.

## The bigger arc

This diary closes a loop opened in 081: the original BPE training was
done on a different (older, less-clean) corpus and got stopped
mid-flight. The new comparison uses the same case-preserved corpus
for both, makes the matching explicit ("same characters seen at
completion"), and runs both to convergence on machines we can monitor.

When both runs finish, we should be able to state — for the first
time, on a fair experimental footing — what tokenization buys (or
costs) at this scale on literary English, with everything else
controlled.
