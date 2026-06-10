# Diary 096 — No-GELU invents words: same loss, different behavior

Date: 2026-06-07

## The setup

After the Studio char-level baseline training finished on 2026-06-02
(16 layers, 8 heads, n_embd=1280, block=4096, final val loss 0.7152
per character), the next question was what role the MLP's GELU
nonlinearity actually plays. Each transformer layer has two
sub-blocks: an attention sub-block and an MLP sub-block. The MLP
internally widens the residual stream from 1280 to 5120 dimensions,
applies a GELU element-wise on those 5120 hidden dimensions, then
projects back down to 1280. Setting `--no_gelu` in `py/train.py`
bypasses the GELU. Without the GELU, the two linear projections
inside the MLP compose into a single linear map of rank at most
1280, so the 4× widening becomes mathematically wasted — the MLP
block reduces to a single rank-1280 linear transformation.

Two trials were launched to probe this:

1. **First trial (2026-06-05, completed 2026-06-06):** `--no_gelu`
   added, `max_iters=10000`, otherwise identical to the baseline.
2. **Matched-LR trial (launched 2026-06-06, ongoing):** `--no_gelu`
   plus `max_iters=500000` (same as baseline). The matched
   `max_iters` keeps the cosine learning-rate schedule identical to
   the baseline over the comparison window, so the only difference
   from the baseline is the missing GELU. Initially planned to be
   stopped manually around iter 10K, but Ralph decided to let it run
   to completion.

The first trial established that no-GELU does train coherently — the
loss descends, no NaN, no instability — but its iter-10K val loss
of 1.84 looked much worse than the baseline's iter-10K val of 1.04.
Most of that gap turned out to be a confound: the first trial's
cosine schedule collapsed LR to its floor by iter 10K, while the
baseline still had near-peak LR there. The matched-LR trial fixes
this and lets the no-GELU descent develop under the same conditions
as the baseline.

## The data

At iter 20,000 the matched-LR no-GELU trial had train loss 1.2039,
val loss 1.1628. The baseline at iter 20,000 had train loss 0.9539,
val loss 0.9648 — about 0.20 nats per character better. So at
equal iter and equal LR schedule, the no-GELU model is genuinely
behind, by a clear and stable margin.

Three samples at iter 20K from each run, both at temperature 0.8 and
top_k 40, both prompted with a single space:

**Baseline (with GELU), val 0.9648:**

```
1. " I have done me the work to do the little intellect and may have
    me tell you--I am going to keep my mouth and look out for my yo[ung]"

2. ""The young men were attentive, and health and stationary. As to
    the address of the whole body of the same name, from the day of"

3. ""There was a scratching confusion in one of the directions before
    the old man visited the old man who visited the service and fo[r]"
```

**No-GELU (matched-LR), val 1.1628:**

```
1. " the plake in give the bay. The culty with two fights green
    intivitiate to me that Jefferson's heart may be the capacific
    market"

2. " miser, and the point of his eye down the seaward; and the child,
    that he had sensed the army; and the interruptions to the elec[tion?]"

3. "s weile the fire wish of witnesside the mingle armies, and that
    Tom named Patron with the word and maps in the midscenes of the"
```

## The observation

The baseline at val 0.96 produces almost entirely real English words.
Where it fails is at the semantic level — "scratching confusion in
one of the directions", "health and stationary" — but every word is
recognizable. The no-GELU at val 1.16 produces a noticeable number
of *invented* words, all with plausible English morphology but no
actual lexical existence:

- *plake* — looks like a noun
- *culty* — looks like an adjective
- *intivitiate* — looks like an `-iate` verb, perhaps a blend of
  *initiate* and *invite*
- *capacific* — looks like an `-ific` adjective, perhaps a blend of
  *capable* and *Pacific*
- *weile* — looks like an archaism or a dialect form
- *witnesside* — looks like a compound noun
- *midscenes* — looks like a compound noun

That is seven invented tokens across three samples of ~128 characters
each. The baseline at lower loss produces at most one or two.

## Matched-loss comparison

The natural follow-up question Ralph asked: is this just a loss-level
effect? At lower loss the no-GELU model might also produce real
words. To test this, the two runs' samples should be compared at
*matched val loss* rather than at matched iter.

This is constrained by what's already saved. The baseline run sampled
every 10K iters (10K, 20K, 30K, ...) and saved checkpoints every 20K
iters (20K, 40K, ...). The baseline crossed val 1.16 around iter
5,500-6,500 — well before its first sample (iter 10K) and well before
its first checkpoint (iter 20K). So no baseline sample exists at
exactly val 1.16, and no baseline checkpoint is available to generate
one from.

The closest matched-loss comparison from existing artifacts:

- Baseline iter 10K, val 1.04 — 0.12 nats *lower* than the no-GELU
  point we want to match against.
- No-GELU iter 20K, val 1.16.

Even though the baseline at this comparison is slightly better in
loss, the qualitative gap is striking. The baseline iter 10K sample:

```
1. " Here after a preparative mark of assassination, which implemented
    the entire place in those of this discovery. As he fell near"

2. " Henry, and you there were the only time which would only be no
    chances to emity to cause him. He was enough to be the articles"

3. " Thereiches. ale, buried its destinies of that creature. Lake
    seems too hard attentive to him as of as a leader, its creatures a"
```

Two genuinely invented tokens across all three samples (*emity*,
*Thereiches*), one of which is clearly an attempt at *enmity*. The
rest are real English words assembled into semantically incoherent
sentences.

So the pattern holds even at matched loss: at val ~1.0-1.2, the
baseline produces real words with incoherent semantics; the no-GELU
produces fewer real words and more plausibly-shaped neologisms.
The architectural difference creates a qualitatively distinct
failure mode, not just a quantitative one.

To do an exactly matched comparison (rather than 0.12 nats off), a
baseline-architecture run with `save_interval=1000` from a fresh
start through iter ~7000 would be needed. That would take ~8 hours
on Studio and give a checkpoint at val 1.16 for a sample at the
exact matched point. Worth doing if the qualitative pattern at
exact match becomes important enough to nail down.

## What this suggests about the GELU's role

Diary 088 traced the "appalpittidax" copying mechanism in the
152M-param char model: layer 9 head 3 sent strong attention back to
the first occurrence of the invented word, and the logit lens showed
the character probabilities sharpening at layer 9. That mechanism
required the upper-layer MLPs to act as a content-addressable
key-value memory — the up-projection generates keys, the GELU
soft-selects which keys to activate, the down-projection writes the
associated values back into the residual stream. Geva et al. (2021,
"Transformer Feed-Forward Layers Are Key-Value Memories") established
this functional reading at the level of large transformers.

The no-GELU samples are consistent with that reading. Without the
GELU's nonlinear key-selection, the MLP can still pass through and
recombine features linearly, which is enough to capture:

- Letter co-occurrence statistics (the invented words have
  letter-bigram distributions close to English)
- Word-length distribution (the invented words are plausibly sized)
- Syntactic position-fillers (the invented words land in
  grammatically appropriate slots — *culty* sits where an adjective
  belongs)
- Capitalization, punctuation, dialogue formatting, register

What the linear MLP cannot do is the nonlinear key-value lookup that
distinguishes between candidate real words for a given
syntactic-and-statistical slot. So when the no-GELU model arrives at
"the [adjective] with two fights green", it has the right
statistical shape — but instead of retrieving *casty* or *crusty* or
*lusty* from its lexical inventory, it produces *culty* — a token
that fits the shape but isn't actually in the inventory because the
inventory isn't being indexed.

The model has learned the *statistical envelope* of English without
the *lexical inventory*. Both are kinds of structure a model can
acquire from a corpus, and they trade off in ways the scalar loss
can't see directly.

## Methodological note: loss as a scalar hides distributional differences

Two models can reach similar average loss by very different
distributional means. One model can be excellent on common tokens
and mediocre on rare ones; another can be decent across the board.
Both end up with the same average. The average doesn't say which
strategy is in play.

In this case, the best guess about how the per-position loss
distributions differ:

- The baseline pays very low loss on common closed-class words
  (*the*, *of*, *and*, articles, prepositions) and on within-word
  character predictions, with higher loss on rare-word identity.
  Common words dominate any text, so getting those near-perfect
  pulls the average down.
- The no-GELU model also does well on within-word character
  predictions (its linear MLP can learn letter-bigram statistics)
  but pays higher loss on word starts, where the key-value lookup
  for word identity would normally fire. The result is a higher
  average and a different shape to the residual error.

The averages are 0.20 nats apart at matched iter and ~0.12 nats
apart at the nearest matched-loss comparison — close numbers that
mask qualitatively different behavior. The lesson is methodological:
loss is a useful number for tracking training progress within a
single architecture, but it is not sufficient for comparing
architectures. Architecture comparisons need at least one
qualitative dimension reported alongside the loss — sample
inspection, downstream-task performance, per-position loss
distribution, or something similar — to catch the kind of gap that
appeared here.

If one wanted to design a benchmark that *did* distinguish the two
models cleanly, the natural metric is "fraction of generated tokens
that are real English words" — a measure that probes the lexical
inventory specifically, separately from the statistical envelope.
That measure would show a clear gap where loss shows only a small
one.

## Caveats

- The matched-loss comparison is 0.12 nats off the exact match.
  An exact comparison would require a baseline rerun with finer
  save intervals.
- 20K iters is still very early in training (epoch 0.29 in this
  corpus). The no-GELU model may eventually build up partial
  lexical inventory at later iters, or it may plateau in the
  "plausible English gibberish" regime indefinitely. The matched-LR
  trial is running to completion (24 days) so this will resolve.
- Sampling parameters (temperature 0.8, top_k 40) affect what we
  see. At lower temperature the no-GELU model might produce more
  recognizable forms; at higher temperature the baseline might
  produce more odd words. The architectural gap should survive
  parameter changes but its precise size will not.
- The "fraction of real English words" measure proposed above is
  itself imperfect — it would call *Jefferson* and *Patron* real
  words (they are), but would miss the deeper question of whether
  the model has learned anything about *what those words refer to*.
  That, as discussed in earlier conversations, may not be a
  well-defined question for these models at all.

## Connection to broader interpretability questions

Three threads connect this finding to ongoing work:

1. **Wittgensteinian skepticism about "loss" as a name.** The same
   move that calls for skepticism about "semantic content" applies
   to "loss." It is one word for what, on inspection, decomposes
   into many separable behaviors. When two models with the same
   loss behave differently, what we are discovering is that the
   word "loss" was an inadequate handle for what we wanted to
   track. We wanted "captures English"; we used "low loss" as a
   proxy; the proxy can break.

2. **Feature inventories vs statistical envelopes.** What the model
   has learned is not a single thing. It is at least two separable
   things — the statistical envelope of the language (what kinds
   of letter sequences are likely) and the lexical inventory
   (which specific words exist and how to retrieve them). The
   GELU is necessary for the second but not the first. Future
   ablations might find other separable kinds of structure —
   syntactic structure, long-range coherence, register-tracking —
   each with its own architectural prerequisites.

3. **Small models as interpretability targets.** The 314M-param
   no-GELU model is small enough that this kind of pattern is
   visible to direct sample inspection. In a 70B model the same
   architectural ablation would produce subtler symptoms (the
   model would have more "ways around" the missing capability)
   and would be much harder to characterize. Ralph's research
   program of working at small scale lets findings like this
   surface in a form humans can directly observe.

## Next steps

- Continue the matched-LR no-GELU run. It will reveal whether the
  invented-word behavior persists or eventually closes as training
  proceeds. Current val (1.16 at iter 20K) is still well above the
  baseline's final 0.7152, with about 480K iters remaining.
- When the trial is far enough along to plot the loss trajectory
  cleanly, update `plots/` with a baseline-vs-no-GELU comparison.
- After this trial, the planned next ablation is no-GELU + no-bias
  (`sh/train_char_uppercase_16L_1280_no_gelu_no_bias_trial.sh`,
  already committed). That will isolate the bias-term contribution
  on top of the no-GELU result.
- If the invented-word pattern persists into later iters, an
  interesting follow-up would be to train a small probe (real-word
  classifier) on the model's residual stream activations to find
  where the lexical-inventory information lives in the baseline
  and *fails to live* in the no-GELU model. That would localize
  the GELU's contribution to specific layers.

— Claude Code Opus 4.7 (1M context)

---

## Addendum 2026-06-10: invented words have faded by iter 80K

Three days into the matched-LR run, the picture has shifted enough
to warrant an update. The "Next steps" section above asked whether
the invented-word behavior would persist or close as training
proceeded. The data now says it closes — much earlier than I would
have guessed.

**Loss trajectory since the original entry:**

| iter | No-GELU val | Baseline val | Gap (nats) |
|------|-------------|--------------|------------|
| 20,000 | 1.1628 | 0.9648 | 0.198 |
| 56,000 | 1.0040 | ~0.85 | ~0.15 |
| 80,000 | 0.9889 | 0.8264 | 0.163 |

The no-GELU val has descended from 1.16 to 0.99 over 60K additional
iters. The gap to baseline has narrowed slightly — from 0.20 nats
at iter 20K to 0.16 nats at iter 80K — but is still substantial.
In probability terms, the baseline is at ~43.6 % per-character
correctness and the no-GELU model is at ~37.2 %. Both runs are at
the same LR (1.42e-04, cosine schedule barely decayed since endpoint
is at iter 500K), same speed (~4.16 sec/iter), no incidents.

**The qualitative finding: invented words are gone.** The three
samples at iter 80K from the no-GELU run:

```
1. " and a their gossip on the other should be the merman of a
    national permanent, which continued that modern station a
    considerabl[e]"

2. " woman skinny let it be known where the lady had been paid for
    a long time, for the more the star belief gathered and ashore in"

3. "ence to do this altogether at the farther existing mind. However,
    he had a great interest, favored at first, and there was no
    re[ason]"
```

Compare this to iter 20K, where three samples contained seven
neologisms (*plake, culty, intivitiate, capacific, weile, witnesside,
midscenes*). At iter 80K I find essentially no invented words.
"Merman" is real, "skinny" is real, "gossip" is real. The few odd
phrasings — "national permanent" used as a noun, "star belief", "the
farther existing mind" — are real-word combinations rather than
shape-fitting neologisms.

The failure mode has shifted from *lexical retrieval failure* to
*semantic incoherence with intact vocabulary* — the same failure
mode the baseline showed at iter 10K. The no-GELU model is now
about 3-4 epochs behind the baseline at producing this kind of
output, but it is producing it.

**Updated reading of what the GELU contributes.** The original
entry interpreted the iter-20K invented words as evidence that the
GELU is necessary for the MLP's key-value memory function, which
is how the model retrieves specific lexical items. The new data
softens that to a quantitative claim: the GELU is necessary for
the MLP's key-value memory function *to develop efficiently*. The
linear MLP can also build up a usable lexical inventory, but it
takes substantially more training time to do so. By iter 80K,
the no-GELU MLP has apparently accumulated enough rank-1280
representational capacity through composition across 16 layers
to act as an effective lexical index, even without the elementwise
nonlinearity each MLP would normally use to select keys.

This is the milder, more accurate interpretation. The GELU is
not strictly necessary for vocabulary acquisition; it is necessary
for vocabulary acquisition to happen on the corpus-exposure budget
the baseline uses. With ~4× the corpus exposure, the no-GELU model
catches up on this specific dimension of behavior.

**What this implies architecturally.** The 4× MLP widening
(1280→5120→1280) provides extra effective rank during training
*via the nonlinear feature decomposition* the GELU enables. Without
the GELU, the same MLP collapses to a rank-1280 linear map, but
the residual stream and the composition across 16 layers still
provide a large total representational capacity. Lexical inventory
information has to be stored *somewhere* — and apparently it can
flow into the attention weights, the per-layer Linear projections,
and the multi-layer composition pattern rather than living
exclusively in the GELU-gated MLP keys. The information has more
than one place to land; it just lands there slower.

**The gap that remains.** Even at iter 80K, the no-GELU loss is
0.16 nats per character above baseline, and the samples are
qualitatively rougher (more semantic incoherence per sentence).
Some structural difference is still being measured. The questions
the run will answer over the remaining ~420K iters:

- Does the gap continue narrowing toward zero, suggesting the
  GELU buys only training speed and no asymptotic capability?
- Does the gap stabilize at some non-zero value, suggesting the
  GELU buys a real asymptotic capability the no-GELU architecture
  cannot match?
- Does the no-GELU model's qualitative output, once stable, ever
  reach the baseline's final-iter coherence, or does it plateau
  at a recognizably less-fluent level?

The strong prior from this update is that the gap will keep
narrowing but probably not close completely — the GELU likely
buys both training speed AND some asymptotic capability — but
the data will decide.

**Methodological lesson.** The original diary entry overweighted
a single time-point observation. The invented words at iter 20K
read as a structural feature of the architecture; in retrospect
they were a stage-of-training feature. This is a recurring trap
in language-model interpretability: any cross-architectural
finding made at a single training point may be measuring "how
fast does X learn" rather than "what is X capable of." Matched-
epoch and matched-budget comparisons can mislead in opposite
directions for the same reason. The diligent approach is to
hold the comparison open across training time and let the
trajectory speak.

— Claude Code Opus 4.7 (1M context)
