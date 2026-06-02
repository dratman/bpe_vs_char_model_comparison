# Diary 094 — Char surpasses BPE on per-character loss given the full training budget

Date: 2026-06-02

## The numbers

The Studio character model finished its planned 500,000 iterations
on 2026-06-02 at 08:54 EDT, 24 days 8 hours after launch. The M3
BPE model stopped at iter 145,100 on 2026-05-20 (its best was at
iter 132,000, and 13 evals past that found no improvement). Both
models share the same architecture (16 layers, 8 heads, 1280
embedding, block 4096) and the same case-preserved 1.27 GB
document-shuffled Gutenberg corpus. They differ only in tokenization
and the per-batch settings that fall out of that.

|                                    | char run                       | BPE run                        |
|------------------------------------|--------------------------------|--------------------------------|
| vocabulary                         | 78                             | 32,000                         |
| batch_size                         | 4                              | 2                              |
| learning_rate (peak)               | 1.5e-4                         | 1.06e-4 (sqrt-scaled)          |
| max_iters configured               | 500,000                        | 220,000                        |
| iters actually run                 | 500,000                        | 145,100 (stopped on val plateau) |
| iter at best-val                   | 482,000                        | 132,000                        |
| epoch at best-val                  | 6.93                           | 4.19                           |
| best val loss                      | **0.7152** (per char)          | **3.3657** (per BPE-token)     |
| best val converted to per-char     | 0.7152                         | ~0.748 (÷ 4.5 chars/token)     |
| P(next char correct) at best-val   | e^(−0.7152) ≈ **0.489**        | e^(−0.748) ≈ **0.473**         |

Per-character loss is the only loss the two models can be compared
on, because the units of their per-token losses are different
quantities. The char model predicts one of 78 characters; the BPE
model predicts one of 32,000 sub-word tokens.

## The reversal

Earlier in training, BPE was clearly ahead:

- At the 2026-05-17 sample point (char iter 154K, BPE iter 95K),
  diary 093 noted *"per-character loss ~0.77, slightly ahead of the
  Studio char run at the same approximate corpus exposure"* for BPE.
- At BPE's best (epoch 4.19, val 0.748 per char), char was still
  somewhere around 0.77.

But char kept training and BPE didn't. Going past epoch 4.6, BPE's
val loss began rising — the classic overfitting signature, with the
train/val gap widening from ~0.15 (early) to 0.25-0.40 (late). Char's
val kept dropping. By the time BPE was stopped at epoch 4.6, char
was at iter ~325K (epoch 4.65) with val ~0.7342, already ahead. Char
then ran for another 2.5+ epochs and found two more bests:

- iter 390K (epoch 5.61, 2026-05-28): val 0.7186
- iter 482K (epoch 6.93, 2026-06-01): val 0.7152

Final char val (0.7152) beats final BPE val (~0.748) by 0.033 in
per-character loss — about a 4 % absolute and 6 % relative
improvement.

## Why the difference

The most parsimonious reading is that BPE's per-token entropy
shrinks much faster than char's per-char entropy. The BPE model
sees ~4.5 characters of context per token slot, so each training
step covers more linguistic surface area. Convergence on the
high-frequency token distributions happens early. Once that
saturates, BPE has no obvious next thing to learn from the same
data — and the model has enough capacity to start memorizing book
fragments instead. That's what late-training overfitting looks like.

The char model has a much harder job per slot: predict the next
letter, given the last 4096 letters. The entropy per slot is
genuinely lower (this is what makes per-character loss
comparable across vocab sizes), but the task keeps yielding
incremental gains for many more epochs. The model can keep
sharpening word-boundary detection, refining rare-word spellings,
and improving long-range coherence in ways that BPE has already
"discretized away" by tokenizing.

This connects directly to diary 093's framing: **tokenization fixes
the level at which the model improvises and at which it can keep
improving**. Char models improvise at the word level — they can
keep refining word-level prediction indefinitely as they see more
text. BPE models improvise at the sentence level — once their
sub-word patterns saturate, there is no finer scale at which more
text would help. The model has nowhere to put the extra capacity
except memorization.

## What this is not

- It is **not** a claim that char tokenization is universally
  better. The cost is real: char training needed 2× the wall time
  per character of corpus (16,384 chars per iter vs 36,864 chars
  per iter), and 50 % more total wall time than BPE would have
  needed at its full 220K-iter schedule.
- It is **not** a statistically powered comparison. One run each.
  Run-to-run noise on val loss at this regime is at least ±0.01-0.02
  in BPE-token units (eval is over 20 batches). That said, the gap
  is ~3× larger than the noise floor.
- It is **not** evidence that BPE would have caught up if trained
  longer. By epoch 4.6 BPE was already overfitting; more iters
  would have made val worse, not better, on this corpus. The right
  followup is BPE with regularization or more data, not BPE for
  more iters.
- It does **not** apply to the M3 BPE *resumed* run (started
  2026-05-26 to complete the originally-planned 220K cosine
  schedule as a deliberate overtraining experiment). That run's
  point is to *measure* the overtraining curve, not to compete
  with char.

## Open questions

1. At what epoch would the char model also begin overfitting?
   Epoch 6.93 (iter 482K) was still finding new bests. The cosine
   LR was down to 1.54e-5 by that point, so the schedule was near
   exhausted. Restarting with a fresh warm LR might find more
   improvement; restarting without might just confirm the plateau.

2. Does the *per-character* improvement reflect the *per-word* or
   *per-sentence* improvement at all, or are they decoupled? Char
   keeps improving on per-char prediction, but does that translate
   into samples that read better at the sentence and paragraph
   level? The final-sample text in the training log is qualitatively
   coherent, but I have not done a side-by-side comparison of
   samples from char-at-iter-342K (val 0.7284) vs char-at-iter-482K
   (val 0.7152) to see what the additional 0.013 of per-char loss
   buys at the prose level. Future diary if it does.

3. With a corpus 10× larger, would BPE keep improving for many
   more epochs (because it would see less repetition) and overtake
   char? Plausibly yes — the overfitting signature here might be
   driven by epoch count more than by token count. The 1.27 GB
   corpus is small enough that 4-5 epochs lets BPE start
   memorizing fragments.

## Related diaries

- **093**: Tokenization fixes the level at which the model
  improvises (char at word level, BPE at sentence level).
- **081**: BPE experiment setup, results, and open questions
  (the precursor to this run).
- **074**: Hypothesis that tokenization is a topological operation.
- **055**: Dark subspace exists in char models, absent in BPE
  models — possibly related: char models have somewhere to put
  late-training fine structure, BPE models don't.
- **049**: Synthesis of character model findings.
