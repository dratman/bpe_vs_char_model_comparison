# Diary 099 — Path forward: consolidate, replicate, publish

Date: 2026-06-10

## The question

Ralph asked, in light of the full diary record, the hardware now
available, and his own intermediate level of knowledge: what path
forward would help him understand more and potentially be useful to
others?

This entry records the recommendation I gave, and the reasoning
behind it, so that future sessions can pick the plan up (or argue
with it) without reconstructing the conversation.

## The lab's position as of today

Assets on hand:

- **A completed matched pair of models** (diary 094): the 320M-param
  char model (best val 0.7152 per character) and the 360M-param BPE
  model (best val 3.2652 per BPE token ≈ 0.725 per character), same
  architecture, same 1.27 GB case-preserved corpus. Char wins by
  ~1.4 % — but with one run per condition, that margin sits at the
  noise floor, as diary 094 itself admits.
- **A live ablation with a story** (diary 098): the no-GELU
  matched-LR run on the Studio, now past iter 80K, ETA ~2026-06-30.
  At iter 20K it invented words (*plake*, *culty*, *capacific*); by
  iter 80K the invented words had faded and the failure mode shifted
  to ordinary semantic incoherence — the same failure mode the
  baseline showed at iter 10K. The finding matured from "GELU is
  necessary for lexical inventory" to "GELU is necessary for lexical
  inventory to develop *efficiently*."
- **Tools ready but underused**: `py/markup_predictions.py` (per-token
  rank and confidence markup, verified working 2026-06-10) and
  `py/memorization_probe.py` (committed 2026-06-06, never yet run
  against real weights).
- **New hardware**: the Linux A6000 box, benchmarked at ~0.52
  sec/iter vs the Studio's 4.18 (diary 096) — roughly 8× faster —
  and currently sitting idle.

## The recommendation, in one sentence

Turn the GELU/lexical-inventory finding into a quantitative,
replicated, written-up result, using the idle A6000 to make
replication affordable for the first time — that is the route that
both deepens understanding and produces something genuinely useful
to others.

## Why the GELU thread and not the others

Three candidate "flagship" findings were weighed:

1. **Char-vs-BPE comparison (093/094).** Weaker as-is: n=1 per
   condition, and the 0.010 per-char margin is within run-to-run
   noise. It needs a second seed before it can carry weight — which
   the plan below provides as a side effect.
2. **GELU/lexical-inventory (098).** The strongest candidate, for
   three reasons. First, it is *observable by direct inspection*:
   invented words appearing and then fading is something any reader
   can see in raw samples without trusting anyone's instruments —
   which is exactly the advantage Ralph's small-model research
   program is built on. Second, it already contains a methodological
   lesson the field needs: the iter-20K conclusion turned out to be
   a stage-of-training effect, not an architecture effect.
   "Single-timepoint architecture comparisons measure learning
   speed, not capability" is a publishable point on its own, and it
   was caught honestly in our own data. Third, the experiment
   finishes itself — the matched-LR run completes around June 30
   with no further effort.
3. **Dark subspace (055) and the diary-095 audio/prosody program.**
   Both interesting, both deferred. The dark-subspace thread is less
   mature as a standalone result; the prosody program is a much
   larger lift (aligned speech corpora, new pipelines) and should
   wait.

## The plan, in order

**Step 1 — now, no training required: the real-word-fraction
metric.** Diary 098 already proposed it: generate samples from each
saved checkpoint of the baseline and no-GELU runs, and score the
fraction of generated words that appear in an English wordlist.
This turns "invented words faded by iter 80K" into a curve over
training time — the centerpiece figure of any write-up. It is a
small Python script; sampling is cheap; the checkpoints already
exist. While at it, finally run `py/memorization_probe.py` on the
three BPE contrast points {132K, 168K, 220K-final} — it closes the
oldest open TODO and adds the overtraining data point.

**Step 2 — put the idle A6000 to work on replication.** Nearly
every result in this diary is n=1. At 0.52 sec/iter, a full
500K-iteration run takes ~3 days on the A6000 versus 24 on the
Studio. Suggested queue: (a) a second-seed char baseline, which
gives error bars and directly strengthens the diary-094 char-vs-BPE
claim; (b) the no-bias trial, already scripted as
`sh/train_char_uppercase_16L_1280_no_gelu_no_bias_trial.sh`. The
deeper point: the A6000 changes what the lab can afford
scientifically. Seeds and ablations stop being month-long
commitments, so replication should become the default standard from
here on.

**Step 3 — when the matched-LR run finishes (~June 30), write it up
for an outside audience.** By then the materials will be: matched-LR
loss trajectories, matched-loss sample comparisons, the
real-word-fraction curves from step 1, and the methodological
lesson about single-timepoint comparisons. That is a tight, fully
reproducible piece at a scale anyone with one GPU can rerun.
Candidate venues: LessWrong / Alignment Forum (where small-model
interpretability work is actively read) or an arXiv note. Releasing
the checkpoints plus `py/markup_predictions.py` on HuggingFace would
let readers poke at the models directly — and the markup tool's
rank-colored output (which already implements idea #1 from diary
095's experiment ladder) makes vivid figures.

## What this plan is built around

Three principles, made explicit:

- **Use what already exists.** Steps 1 and most of 3 require no new
  training — the checkpoints, tools, and half the analysis are
  already on disk.
- **Fix the one real weakness.** The lab's findings are careful but
  unreplicated. The new hardware removes the excuse.
- **End in an artifact for others.** Eighty-plus diary entries are a
  research record, not a publication. One well-chosen finding,
  written up with figures and reproducible code, is how the work
  starts being useful beyond this lab.

## Status

Recommendation delivered 2026-06-10; Ralph has not yet said whether
to proceed. No code written. Step 1 (the real-word-fraction script)
is the natural starting point whenever he gives the word.

— Claude Code Fable 5
