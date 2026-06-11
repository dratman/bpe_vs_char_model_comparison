# Diary 101 — BPE memorization probe: a clean null result

Date: 2026-06-11

## The question being closed

The BPE run was resumed past its early stop (diary 094, HANDOFF
BPE-resumed section) to test the overtraining hypothesis: *samples
grow more memorized as training proceeds past the val-loss minimum.*
`py/memorization_probe.py` was written 2026-06-06 to quantify this
and has been a pending TODO since. It ran overnight 2026-06-10 →
06-11 as job 3 of the step-1 sweep, against the three BPE contrast
points — with the freshly re-staged checkpoints (see below).

Probe design (details in the script header): 200 corpus passages,
512-char prompts, 256-char continuations, seed 42, same offsets for
every model. Metric 1, extractable memorization: greedy free
continuation, exact-match prefix length in characters vs the true
continuation (Carlini-style). Metric 2, teacher-forced greedy
accuracy: argmax==truth per continuation token, one forward pass.

## The result

```
   iter   tok   best_val   mean_chr   med_chr   p95_chr   extract%   tf_acc%
 132000   bpe     3.3657        0.9         0         5       0.0%     34.5%
 168000   bpe     3.2652        0.9         0         5       0.0%     35.6%
 220000   bpe     3.2652        1.2         0         9       0.0%     36.7%
```

(132K = original-run best; 168K = resumed-run true val minimum;
220K = resumed-run final, 52K iters past the minimum.)

**There is no extractable memorization at any of the three points.**
Mean matched prefix is about one character; the median is zero —
on a typical corpus passage the greedy continuation diverges from
the original text immediately. The 95th percentile is 5-9
characters (roughly one word). Not a single prompt out of 200, at
any checkpoint, reproduced even 50 characters verbatim. The rise
from 0.9 to 1.2 mean chars and 34.5 % to 36.7 % teacher-forced
accuracy across 88K iters is real but tiny — the model keeps getting
slightly better at *predicting* the corpus while showing no sign of
*regurgitating* it.

## What this means

The overtraining hypothesis, in its memorization form, is
**disconfirmed at this scale**: training a 360M-param BPE model 52K
iters (1.65 epochs) past its val minimum on a 1.27 GB corpus does
not produce verbatim-recall behavior. This agrees with what diary
094 noted from the loss side — past the minimum, val stayed *flat*
rather than rising; the train/val gap widened only modestly. The
"model memorizes book fragments past the val minimum" story we
built on the original-run plateau is dead in both the loss data and
now the behavioral data.

A capacity argument says we maybe should have expected this: the
corpus is ~283M BPE tokens against ~360M parameters, and the model
sat at val ≈ 3.27 nats/token — it never got within sight of the
near-zero-loss regime where verbatim recall becomes the
loss-minimizing behavior. Memorization-on-overtraining findings in
the literature typically involve many more epochs, much smaller
corpora relative to capacity, or both.

## Caveats

- Greedy decoding from 512-char prompts is one operating point.
  Longer prompts (more specific context) extract more in the
  Carlini-style literature; a 1024- or 2048-char-prompt rerun is
  cheap if we want a stronger version of the null.
- Prompt offsets are uniform over the whole corpus, so ~90 % of
  passages come from the training split (val_split = 0.1). That is
  the right region for a memorization probe, but the metric mixes
  in some val passages; with extract% pinned at zero everywhere,
  the mixture doesn't matter here.
- The char model's late checkpoints were not probed (the probe is
  char-comparable by design; the sweep only covered the standing
  BPE TODO). Worth one run someday for the cross-tokenizer
  comparison: does char's per-character improvisation show even
  less extraction?

## Operational note: stale checkpoint caught and fixed

Before the run, the Studio's copy of
`pt/bpe_uppercase_16L_1280_b2_resumed.pt` turned out to be dated
May 27 — one day BEFORE the true iter-168K best was saved on the M3
(May 28 09:08). It was the iter-157K-era best (val 3.3556), not the
3.2652 true minimum. Both it and the missing
`..._resumed_final.pt` were re-staged fresh from the M3 on
2026-06-10. Any future analysis that used the Studio's pre-June-10
copy of that file should be checked. The probe table above confirms
the fix: the 168K row reports best_val 3.2652.

## Related diaries

- **094**: char surpasses BPE given full budget; noted val stayed
  flat (no catastrophic overfit) past the BPE minimum.
- **099**: the plan this run executes (step 1).
- **100**: the real-word-fraction result from the same sweep.

— Claude Code Fable 5
