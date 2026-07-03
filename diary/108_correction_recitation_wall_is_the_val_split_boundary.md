# Diary 108 — Correction to 107: the recitation "wall" is the train/val split boundary, not a capacity limit

Date: 2026-07-03

## What this corrects

Diary 107 (Green Eggs memorization) concluded that the char/block-256 model's
free-running recitation stopped at char 3040 (90.1%) because of an
**under-memorization / capacity limit** — a transition the small model couldn't
carve out against high-frequency look-alikes. That conclusion is **wrong.**
Findings 1 and 2 of diary 107 stand; Findings 3 and 4 (the char-3040 story) are
superseded by this entry.

## How it surfaced

While setting up a BPE version of the run (below), Ralph asked a one-line
question — "Or change the split?" — which prompted checking how `train.py`
splits train/val in continuous mode. It is **contiguous**:

    n_val = int(n * args.val_split)
    train_data = all_tokens[:n_train]   # first (1 - val_split)
    val_data   = all_tokens[n_train:]   # LAST val_split fraction

So with the default `val_split 0.1`, the **last 10% of the book is held out** as
validation — the model never trains on it. For the 3,375-char corpus, training
data ends at char **3037**. The model's recitation broke at char **3040**. A
90.1% wall sitting next to a 90% split boundary should have been an immediate
red flag in 107; it wasn't caught.

## The decisive test — move the split, watch the wall move

Retrained char/block-256, identical except `val_split 0.30` (train on the first
70% only, 2,363 chars):

| val_split | Train data ends at | Recitation wall (observed) |
|---|---|---|
| 0.10 | char 3037 (90%) | char 3040 (**90.1%**) |
| 0.30 | char 2363 (70%) | char 2381 (**70.5%**) |

The wall tracks the train/val boundary. The model recites what it trained on and
diverges within ~20 chars of where its training data ends. This is causal, not
correlational: changing only `val_split` relocated the wall.

**Corrected interpretation.** The char/block-256 model did **not** hit a
capacity limit at char 3040. It was never shown the last 10% of the book, so it
cannot recite it and is (correctly) confident-wrong at the first held-out
transition. The teacher-forced probe in 107 (P(correct) ≈ 0) is fully explained
by "never trained on this target." And 107's "training longer didn't help"
(`bs256_long`) is the same story — you cannot memorize data you never see; more
iterations on the training 90% will never teach the held-out 10%.

## What still stands from 107

- **Greedy reveals memorization; temperature-0.8 hides it.** Unchanged.
- **The block-128 break at char 608 is a genuine context collision.** Unchanged
  — 608 is at 18%, far below any split boundary, and it was a true
  identical-128-char-window collision (the "…Sam-I-Am.\n\n" refrain preceding two
  different stanzas). Collisions are a real, separate ceiling.

**Unified picture:** free-running recitation from the first line is bounded by
`min(first context collision, train/val boundary)`. For block 256 there is no
collision before the boundary, so the boundary (90% at val_split 0.1) binds. For
block 128 the collision at char 608 binds first. Neither run was capacity-limited.

## The BPE rerun that started this (green_eggs_memorize_bs256_bpe)

Ralph asked to rerun the best char run with BPE. Notes:
- BPE (ByteLevel, requested vocab 512) trained only **182 tokens** on this 3.3 KB
  corpus and compressed the book to **1,121 tokens**.
- `block_size 256` was **infeasible**: the 10% val split is only 112 tokens <
  256, so `get_batch_continuous` crashed (`randint from=0 >= to=-144`). Had to
  drop to `block_size 64` tokens (~190 chars of context).
- Result: **18.0% (char 608), the same collision and same "A train!" derailment
  as the char/block-128 model.** Per-char loss ~0.018 (0.053/token ÷ ~3
  chars/token) ≈ the char model's 0.021 — BPE memorized locally just as well.
- Takeaway: at comparable *character* context, BPE and char memorize about
  equally and hit the same collision. Tokenization neither helped nor hurt here;
  **effective character-context width** drove the difference. (And a corpus this
  short in BPE tokens can't support a 256-token window at all.)

## Checkpoints from this session (MacBook, gitignored)

- `pt/green_eggs_memorize_bs256_bpe*` — BPE, block 64, 3000 iters (slug still
  says "bs256"; misleading — it is block 64).
- `pt/green_eggs_char_bs256_val30*` — char, block 256, val_split 0.30, the
  split-boundary test.

## Method note for future me

When a memorization/recitation run "walls out" at a round percentage, check the
train/val split boundary FIRST (contiguous, last `val_split`), before reaching
for capacity or collision explanations. `(1 - val_split) × length` is the prime
suspect.
