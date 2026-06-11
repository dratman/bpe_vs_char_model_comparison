# Diary 102 — WordPiece × no-GELU: does tokenization substitute for the nonlinearity?

Date: 2026-06-11

## Ralph's experiment

Ralph proposed (2026-06-11 morning): train with **WordPiece tokenization
and the activation function turned off**. WordPiece is the BERT-style
tokenizer — whitespace pre-tokenization, bare spaceless word-initial
tokens, `##`-prefixed continuation pieces. At vocab 32,000 the most
frequent ~10–20K word forms are single whole-word tokens, and (unlike
GPT-2-style byte-level BPE) no vocab budget is spent on space-glued
duplicates like ` the` vs `the` — one lexeme, one slot.

## Why this is the right next ablation

Diaries 098/100 found that at char level, removing the GELU costs
0.16–0.20 nats/char and slows lexical acquisition ~4× — the linear MLP
is bad at the key-value lookup that retrieves specific word forms. The
interpretation was: the GELU's big early-training contribution is the
**lexical inventory**.

WordPiece moves the lexical inventory **into the tokenizer**. The model
never spells a common word; it selects it from the vocabulary. So:

- If the no-GELU penalty largely **disappears** under WordPiece, the
  lexical-inventory interpretation is confirmed, and tokenization and
  MLP nonlinearity are partial *substitutes* for storing the lexicon.
- If the penalty **persists**, the GELU contributes something beyond
  word identity (composition, semantics, long-range structure), and
  diary 098's interpretation needs revision.

## Design

Two runs on the A6000, sequential, queued behind the seed-2 char
replication (diary 099 step 2a, finishes ~2026-06-14):

1. **WordPiece control** (GELU on) —
   `sh/train_wordpiece_uppercase_16L_1280_b2_CUDA.sh`
2. **WordPiece no-GELU** —
   `sh/train_wordpiece_uppercase_16L_1280_b2_no_gelu_CUDA.sh`

Pairing controls (tighter than the char-level pair had):
- **Same `--seed 42`** → identical weight init and identical batch
  order (the GELU has no parameters, so shapes match).
- **Same tokenizer instance**: run 2 loads run 1's trained tokenizer
  via the new `train.py --tokenizer_from` flag → byte-identical token
  stream. The ONLY difference between the runs is the GELU.

Hyperparameters match the M3 BPE run exactly (vocab 32,000, batch 2,
lr 1.06e-4, warmup 500, block 4096, 220K iters, bf16), so
{BPE, WordPiece, WordPiece-no-GELU} form one comparable family. One
exception: save_interval 10000 instead of 5000 — two runs at 5K
granularity (~380 GB of checkpoints) don't fit the A6000's disk
alongside the seed-2 run's artifacts.

The control run doubles as a standalone result: WordPiece (spaceless)
vs the existing byte-level BPE run at matched everything is the clean
test of the space-gluing convention's cost.

## Implementation notes (committed `ab3b33f`)

- `py/tokenizer.py`: new `WordPieceTokenizer` (HF `tokenizers`
  WordPiece model/trainer, `##` prefix, cleanup decoder). Whitespace
  pre-tokenization would erase newlines entirely — the model would
  never see paragraph structure — so `encode()` maps `\n` to a `[NL]`
  special token and `decode()` restores it. Round-trip verified on
  newlines (incl. blank lines), contractions ("Don't"), possessives
  ("Elizabeth's"), `<|endoftext|>`. Known lossiness: runs of spaces
  collapse; an opening single-quote glues to the next word on decode
  (rare in this corpus; dialogue uses double quotes).
- `py/train.py`: `--tokenizer wordpiece` choice; `--tokenizer_from
  <meta.pkl>` to reuse a trained tokenizer across paired runs.
- `py/sample.py`: case-preservation detection extended to wordpiece.
- `sh/queue_wordpiece_pair_after_seed2.sh`: queue runner armed on the
  A6000 (waits for the seed-2 PID, then runs the pair; aborts on
  startup crashes, warns if a log lacks "Training complete").
- End-to-end tested at toy scale on the Studio (tiny corpus, 2L/64d,
  both runs incl. `--tokenizer_from` + `--no_gelu` + mid-training
  sampling) in `/tmp/wptest`, deleted after.

## Preregistered predictions (written before either run starts)

1. **No invented words, even early.** WordPiece-no-GELU samples at
   early iters will show *near-baseline real-word fraction* (≥ ~98 %)
   from the first checkpoints — token-level neologisms would require
   deliberate `##`-piece composition, which top_k sampling makes rare.
   The char-level signature (81 % at iter 5K) should be absent. If
   neologisms DO appear via ## composition at scale, that itself is
   informative.
2. **The headline number:** the WordPiece no-GELU val-loss penalty,
   converted to per-character-equivalent, will be **well under half**
   the char-level penalty (char: 0.16–0.20 nats/char over iters
   20K–80K). My specific guess: ≤ 0.05 nats/char-equivalent. If it
   stays ≥ 0.1, the lexical-inventory interpretation of diary 098 is
   wrong or incomplete.
3. The no-GELU run will still trail the control by *some* persistent
   margin (composition/semantics aren't free), and its samples will
   read as less coherent at matched iters — same direction as char,
   smaller magnitude.

## Schedule

A6000 queue (self-managing, no action needed): seed-2 char until
~06-14 → WordPiece control (~1.4 days) → WordPiece no-GELU
(~1.4 days) → pair done ~2026-06-17. Studio: no-GELU char matched-LR
run continues to ~06-30. Analysis when the pair lands: loss curves +
real-word-fraction sweep + sample comparison, vs these predictions.

— Claude Code Fable 5
