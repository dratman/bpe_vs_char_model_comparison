# Diary 100 — Real-word fraction quantifies the GELU's lexical-inventory effect

Date: 2026-06-11

## What was run

Diary 098 proposed the metric; diary 099 step 1 commissioned it; this
entry reports the result. `py/real_word_fraction.py` (committed
2026-06-10, commit `f22578b`) generates 16 samples × 512 characters
from each checkpoint (prompt = single space, temperature 0.8,
top_k 40 — the same settings as the training-time samples diary 098
compared) and scores the fraction of generated words that occur at
least 5 times in the training corpus. The corpus is the reference
dictionary on purpose: it contains the proper nouns and 19th-century
vocabulary a standard wordlist lacks, and the model can only have
learned words it actually saw. Every checkpoint receives the
identical RNG stream (seed+sample-index), so differences are due to
the model alone. ~1,500 scorable words per checkpoint.

Sweep ran overnight 2026-06-10 → 06-11 on the Studio
(`sh/real_word_fraction_sweep.sh`), alongside the live no-GELU
training. Raw tables + TSVs: `terminal_logs/
real_word_fraction_{baseline,no_gelu}_2026_06_10_1507.{txt,tsv}`.

## The curves

**No-GELU matched-LR run** (every 5K iters):

| iter | val loss | real-word % |
|------|----------|-------------|
| 5,000 | 2.0801 | 81.0 |
| 10,000 | 1.4624 | 82.5 |
| 15,000 | 1.2864 | 92.6 |
| 20,000 | 1.1628 | 94.6 |
| 25,000 | 1.1297 | 96.2 |
| 30,000 | 1.0853 | 96.4 |
| 35,000 | 1.0644 | 97.7 |
| 40,000 | 1.0581 | 97.3 |
| 45,000 | 1.0803 | 97.7 |
| 50,000 | 1.0249 | 97.2 |
| 55,000 | 1.0167 | 98.6 |
| 60,000 | 1.0300 | 98.5 |
| 65,000 | 1.0314 | 98.1 |
| 70,000 | 1.0277 | 97.9 |
| 75,000 | 1.0020 | 98.9 |
| 80,000 | 0.9889 | 99.0 |

**Baseline run** (every 20K to 80K, then every 40K):

| iter | val loss | real-word % |
|------|----------|-------------|
| 20,000 | 0.9648 | 99.1 |
| 40,000 | 0.8864 | 99.3 |
| 60,000 | 0.8471 | 99.4 |
| 80,000 | 0.8264 | 99.4 |
| 120,000 | 0.8231 | 99.5 |
| 160,000 | 0.7994 | 99.6 |
| 200,000 | 0.7904 | 99.9 |
| 240,000 | 0.7844 | 99.9 |
| 280,000 | 0.7556 | 99.5 |
| 320,000 | 0.7342 | 99.9 |
| 360,000 | 0.7582 | 99.9 |
| 400,000 | 0.7326 | 99.7 |
| 440,000 | 0.7562 | 99.8 |
| 480,000 | 0.7218 | 99.6 |
| 482,000 (best) | 0.7152 | 99.7 |
| 500,000 (final) | 0.7152 | 100.0 |

(With ~1,500 words per point, one standard deviation is ~0.26
percentage points at 99 % and ~0.56 at 95 % — the no-GELU climb from
81 to 99 is far outside noise; the baseline's wiggles between 99.5
and 99.9 are mostly within it.)

## What the curves say

1. **The baseline starts essentially converged on this metric.** At
   its first saved checkpoint (iter 20K, epoch 0.29) it is already at
   99.1 % — about one invented word per hundred. The GELU-equipped
   model acquires its lexical inventory almost immediately, before
   the earliest point we can measure. The metric's whole dynamic
   range lives in the no-GELU run.

2. **The no-GELU run climbs from 81 % to 99 % over 80K iters, with
   the steepest gain between iters 10K and 20K.** This is diary 098's
   qualitative arc — invented words at 20K, gone by 80K — as a
   smooth, monotonic-up-to-noise curve.

3. **The "~4× corpus exposure" estimate from the diary-098 addendum
   lands almost exactly.** No-GELU at iter 80K: 99.0 %. Baseline at
   iter 20K: 99.1 %. The linear-MLP model reaches the baseline's
   *earliest measured* lexical state after four times the training.
   At matched iter 80K the gap is small but probably real:
   99.0 vs 99.4 (~1.5σ).

4. **The non-word lists differ in kind, not just in count** — the
   scalar undersells the difference. The baseline's few misses are
   mostly plausible proper nouns and name-forms (*ferrill*,
   *boonsborow*, *marietta's*, *zamdovich*) — the kind of token a
   human reader would accept as a name in a 19th-century novel. The
   no-GELU misses are morphological neologisms — shape-fitting
   assemblies like *archeristral*, *manuscretion*, *contendencies*,
   *substragged*, *promontrained* — exactly the "statistical envelope
   without lexical lookup" phenotype diary 098 described. Even at
   iter 80K, where the no-GELU rate is near-baseline, its residual
   non-words are still of the neologism kind.

5. **One degenerate sample worth flagging:** the no-GELU iter-40K
   row contains `rostoctoctoct…octories` — a repetition loop inside
   one sample, the same low-content attractor noted for BPE sampling
   in the 2026-05-27 log. Single occurrence; does not move the
   curve.

## Methodological note

This is the benchmark diary 098 asked for: a measure that probes the
lexical inventory specifically, separately from the statistical
envelope, and it shows a wide gap (81 → 99 vs flat-at-99) where val
loss shows a steady ~0.16-0.2-nat offset. Two models' scalar losses
hid this; a 250-line sampling script exposed it. The metric
saturates once the inventory is in place, so it is a probe of
*early* lexical acquisition — it will not distinguish the two runs
late in training, where per-position loss distributions or
sample-coherence measures would have to take over.

## What's next for this curve

- The no-GELU run continues (~iter 84K of 500K, ETA ~2026-06-30).
  Re-run the sweep's job 2 on later checkpoints occasionally (the
  script takes a `--models` list; ~22 min per checkpoint on the busy
  Studio) to see whether the residual neologism rate fully closes.
- Plot the two TSVs (real-word % vs iter, log-x) as the centerpiece
  figure for the step-3 write-up of diary 099's plan.
- A second-seed replication of both runs on the A6000 (diary 099
  step 2) would put error bars under the whole curve.

— Claude Code Fable 5
