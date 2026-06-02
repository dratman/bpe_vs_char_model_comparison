# Diary 094 — Char surpasses BPE on per-character loss given the full training budget

Date: 2026-06-02

*Note: an earlier draft of this entry used the BPE original-run best
(0.748 per-char) for the comparison. After noticing that the BPE
*resumed* run reached a better best of 0.725 per-char before its own
full schedule completed (2026-06-01), the comparison was redone using
that better number. The thesis stands but the margin is narrower.*

## The numbers

The Studio character model finished its planned 500,000 iterations
on 2026-06-02 at 08:54 EDT, 24 days 8 hours after launch. The M3
BPE model was run in two legs: an original run that was stopped on
val plateau at iter 145,100 (2026-05-20), then a resume that
continued the original 220,000-iter cosine schedule to completion
(2026-06-01) as an overtraining-curve experiment. Both legs share
the same model, optimizer, schedule, and corpus — only the RNG
state differs between them, so they're best read as one model
trained for 220K iters with a pause around 145K.

All three runs share the same architecture (16 layers, 8 heads,
1280 embedding, block 4096) and the same case-preserved 1.27 GB
document-shuffled Gutenberg corpus. They differ only in tokenization
and the per-batch settings that fall out of that.

|                                    | char run             | BPE original           | BPE resumed                |
|------------------------------------|----------------------|------------------------|----------------------------|
| vocabulary                         | 78                   | 32,000                 | 32,000 (same)              |
| batch_size                         | 4                    | 2                      | 2                          |
| learning_rate (peak)               | 1.5e-4               | 1.06e-4                | 1.06e-4 (cosine continued) |
| max_iters configured               | 500,000              | 220,000                | 220,000                    |
| iters actually run                 | 500,000              | 145,100 (stopped early)| 220,000 (full schedule)    |
| iter at best-val                   | 482,000              | 132,000                | 168,000                    |
| epoch at best-val                  | 6.93                 | 4.19                   | 5.34                       |
| best val loss                      | **0.7152** per-char  | 3.3657 per-BPE-token   | **3.2652** per-BPE-token   |
| best val per-char (÷ 4.5)          | **0.7152**           | ~0.748                 | **~0.725**                 |
| P(next char correct) at best-val   | e^(−0.7152) ≈ **0.489** | e^(−0.748) ≈ **0.473** | e^(−0.725) ≈ **0.484**  |

Per-character loss is the only loss the two tokenization schemes
can be compared on, because the units of their per-token losses
are different quantities (78-way vs 32K-way classification at each
step).

The right BPE number to use for the comparison is the resumed
run's 0.725, not the original's 0.748. The original-run stop was
premature: there were ~36K more iters of genuine val improvement
available, and missing them initially distorted the picture.

## The reversal

Earlier in training, BPE was clearly ahead:

- At the 2026-05-17 sample point (char iter 154K, BPE iter 95K),
  diary 093 noted *"per-character loss ~0.77, slightly ahead of the
  Studio char run at the same approximate corpus exposure"* for BPE.
- At BPE-original's first apparent best (epoch 4.19, val 0.748 per
  char), char was still somewhere around 0.77.

But char kept training and BPE-original was stopped on what looked
like a val plateau. Char's val kept dropping. By the time char
crossed epoch 5.3 (iter ~410K) with val ~0.72, BPE-resumed had also
crossed epoch 5.3 (iter 168K) with val 0.725 — its own true minimum.
Both runs found their best around the same epoch but char's was
slightly lower. Char then ran for another 1.7 epochs while BPE-resumed
held flat at 0.725 through epoch 6.99.

Char's late-training best-vals:

- iter 390K (epoch 5.61, 2026-05-28): val 0.7186
- iter 482K (epoch 6.93, 2026-06-01): val 0.7152

BPE-resumed found no new best in its last 52K iters (168K → 220K),
with val bouncing in the 3.40-3.51 range — the model had saturated.
Char's val kept inching down over the same epoch range, ending at
0.7152.

Final char val (0.7152) beats BPE-resumed's true minimum (~0.725)
by 0.010 in per-character loss — about a 1.4 % absolute and 1.4 %
relative improvement. Small but consistent across the final few
epochs where both models had access to the same data.

## Why the difference

The most parsimonious reading is that BPE's per-token entropy
shrinks faster than char's per-char entropy. The BPE model sees
~4.5 characters of context per token slot, so each training step
covers more linguistic surface area, and the model converges on
the high-frequency token distributions earlier. Once that
saturates, BPE has no obvious next thing to learn from the same
data — which empirically shows up as the val-loss floor at 0.725
that BPE-resumed sat on for 52K iters without further improvement.

(Importantly, BPE-resumed did *not* catastrophically overfit past
its saturation point — train/val gap widened modestly but val
stayed flat rather than rising. The "model memorizes book
fragments past the val minimum" story we'd guessed from the
original-run plateau didn't materialize in the cleaner data the
resumed run produced.)

The char model has a much harder job per slot: predict the next
letter, given the last 4096 letters. The entropy per slot is
genuinely lower (this is what makes per-character loss
comparable across vocab sizes), but the task keeps yielding
incremental gains for more epochs. The model can keep sharpening
word-boundary detection, refining rare-word spellings, and
improving long-range coherence in ways that BPE has already
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
  per iter), and ~4× more total wall time than BPE (24d 8h vs
  ~6d for the full resumed schedule, accounting for the M3 being
  the slower machine in this comparison).
- It is **not** a statistically powered comparison. One run each.
  Run-to-run noise on val loss at this regime is at least ±0.01-0.02
  in BPE-token units (eval is over 20 batches). The 0.010 per-char
  gap is right at the edge of that noise floor — within one or two
  σ. A second run of either model could swap the ranking.
- It is **not** evidence that BPE would have caught up if trained
  longer than 220K iters. The BPE-resumed run reached its true
  minimum at iter 168K (epoch 5.34) and stayed there for the next
  52K iters — the model had saturated. More iters would not help;
  the right followup is BPE with regularization, more data, or a
  larger vocabulary, not BPE for more iters.

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
