# Diary 109 — Clean BPE-vs-char comparison: memorization ties, BPE trains ~3× faster (sequence length)

Date: 2026-07-03

## Purpose

Diaries 107 and 108 established that the Green Eggs recitation differences we
first saw (char 90% vs BPE 18%) were **artifacts**, not tokenization effects:
the char-608 break was a context collision, the 90% wall was the train/val split
boundary (108). This entry runs the *controlled* comparison with both artifacts
removed, to isolate the actual effect of tokenization on (a) memorization and
(b) training speed.

## Design (artifacts removed)

Corpus `Green_Eggs_And_Ham_definitive_1b.txt` (3,375 chars). Both models:
4L/4H/128, continuous mode, 3000 iters, lr 1e-3, seed 1337, **`val_split 0.08`**
(train on the first 92% — so the split boundary sits near the end, not at 90%),
on MPS.

- **char:** `block_size 256` → 256 characters of context.
- **BPE:** ByteLevel, vocab trained to 182 tokens; the book is 1,121 BPE tokens.
  `block_size 85` → ≈256 characters of context (BPE packs ~3 chars/token), so the
  *character* context is matched to the char model.

Both windows (~256 chars) are wide enough to clear the char-608 collision, and
both models train on the same 92% of the text. So neither artifact from 107/108
is in play.

## Result 1 — memorization is a tie

Greedy recitation from the first line, measured in characters reproduced
verbatim:

| Model | Context | Verbatim from line 1 | Stops at |
|---|---|---|---|
| char | 256 chars | **92.1%** (3108 / 3375) | train boundary (~char 3105) |
| BPE  | ~256 chars | **91.5%** (3089 / 3375) | train boundary (~char 3105) |

Both recite everything they were trained on and wall out within a few characters
of the 92% train/val boundary. The 0.6% difference is just where each tokenizer's
boundary token happens to land. **Tokenization made no meaningful difference to
memorization.** At matched character-context and matched training fraction, char
and BPE memorize equally well; what governs how far a model can recite is
*context width* and *how much of the text it trained on*, not the tokenizer.

## Result 2 — BPE trained ~3× faster (but the lever is sequence length)

Same machine, same 3000 iterations, same model size:

| Model | Block | Wall-clock | Speed |
|---|---|---|---|
| char | 256 tokens | 5 min 9 s | ~9.7 iters/s |
| BPE  | 85 tokens  | 1 min 36 s | ~31 iters/s |

BPE was **3.2× faster**, almost exactly matching the block_size ratio
(256 / 85 = 3.0×). Per-step compute scales with sequence length, so the char
model did ~3× more work per iteration purely because its sequences were 3× longer.

Important framing: this is **not** "BPE is intrinsically faster to train." At the
*same* block_size, per-step cost would be nearly identical. The real point is that
**BPE's compression let the same 256 characters of context fit in ~85 tokens
instead of 256** — shorter sequences → cheaper steps → faster training. That is
the standard reason BPE is used at scale, and it grows once attention's O(n²) term
dominates at larger context lengths.

## Bottom line for the BPE-vs-char question (this toy)

- **Memorization capacity:** tokenizer-agnostic. Context width and training
  fraction are the levers, not char-vs-BPE.
- **Training cost:** BPE wins, because compression buys shorter sequences for the
  same amount of text in context.
- The dramatic-looking early results (90% vs 18%) were entirely explained by
  context width + the char-608 collision + the split boundary — see 107, 108.

## Checkpoints (MacBook, gitignored)

- `pt/green_eggs_char_bs256_val08*` — char, block 256, val_split 0.08.
- `pt/green_eggs_bpe_bs85_val08*` — BPE, block 85, val_split 0.08.

## Caveat

This is a 3.3 KB toy corpus and 0.8M-param models; treat as methodology and
intuition, not a scaling claim. The BPE corpus is only 1,121 tokens, which is
what forced the small block sizes and made a wide-context + tiny-val_split BPE run
impossible (108). Real per-character loss comparisons on the production models are
the proper test (see the project's loss-normalization note in CLAUDE.md).
