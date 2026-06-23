# Diary 105 — A character model assembles a category region mid-network, and semantics beats spelling

Date: 2026-06-23

## Where this came from

Ralph had a conversation on Claude.ai (2026-06-21) about how a language
model detects categories and analogies, which moved — through his own
pushing — to the real question behind generalization: *why does
"nearby in the model's space" track "actually similar in the ways that
matter," rather than just "spelled alike"?* The browser conversation
told the standard word2vec story ("the cat-vector sits near the
dog-vector"). But that story assumes the token **is** the word. In our
character models there is no cat-vector — the embedding table holds only
single letters. So a category like "animal" cannot be looked up; it has
to be **assembled in the residual stream**, across layers, after the
network has recognized c-a-t as a unit. This entry tests whether that
assembled region exists, where in depth it appears, and — the part that
makes it mean something for a char model — whether it tracks **meaning**
or merely **spelling**.

## The design (the minimal-pair control is the whole point)

`py/category_geometry_probe.py`. Two word groups of 9, length-matched,
arranged as near **minimal pairs** so spelling is pitted directly
against meaning:

    cat/hat  dog/log  fox/box  owl/bowl  hare/hair
    goat/coat  mouse/mouth  wolf/wool  horse/house

Every animal shares letters (usually the final letter) with its paired
object. If "fox" sits with cat/dog/owl and **away** from "box" despite
the spelling, that is semantics beating surface form.

Each word is read inside 4 carrier sentences whose slot sits at varied
positions (so the positional signal cancels under frame-averaging). At
every layer we capture the residual stream at two readout positions:
(a) the word's **final letter**, and (b) the **position just after** it
(the model's running summary of the word before it predicts onward).
Per layer the 18 word vectors are **mean-centered** (this removes the
dominant shared residual direction that otherwise makes every cosine
~0.9), L2-normalized, and compared by cosine.

`separation` = mean within-category cosine − mean between-category
cosine. A **permutation null** (shuffle the animal/object labels 2000×)
gives the separation a scale (z, p). The **minimal-pair test** asks, for
each animal, whether it is closer to the centroid of the *other* animals
than to its surface-twin object (e.g. "9 of 9").

Run on the best Studio char checkpoint
`pt/char_uppercase_16L_1280.pt` (16 layers, n_embd 1280, iter 482K,
val 0.7152), read-only on CPU so it did not touch the live no-GELU run.
Raw table: `terminal_logs/category_geometry_2026_06_23.tsv`; heatmap:
`plots/category_geometry_2026_06_23.png`.

## Result

The **position-after-word** readout is the clean one:

| layer | separation | z | minpair |
|------|-----------:|----:|:-------:|
| embed | −0.101 | −0.8 | 1/9 |
| L00 | −0.105 | −1.1 | 0/9 |
| L03 | −0.054 | −1.7 | 0/9 |
| L04 | −0.034 | −1.2 | 0/9 |
| **L05** | **+0.051** | **2.5** | 0/9 |
| **L06** | **+0.193** | **7.6** | 5/9 |
| **L07** | **+0.281** | **9.5** | 9/9 |
| L08 | +0.347 | 9.6 | 9/9 |
| L09–L15 | +0.37 plateau | ~10 | 9/9 |

Three things, in order of importance. Note that two *different* depths
are doing two *different* jobs — the minimal-pair count and the
separation magnitude saturate at different layers, and that distinction
is the entry's whole point:

1. **The category region is real and strongly significant.** Separation
   climbs L05→L09 (0.051 → 0.193 → 0.281 → 0.347 → 0.373) and then
   **plateaus around 0.37 from L09 through L15** (peak L13, 0.383), all
   at z ≈ 10, p < 0.0005. Animals cluster; objects sit apart.

2. **Semantics beats spelling, cleanly — and it locks in earlier than
   the magnitude plateaus.** The minimal-pair count hits **9/9 at L07**
   (5/9 already at L06) and holds 9/9 thereafter: every animal is nearer
   to the other animals than to its look-alike object, despite shared
   letters. fox is animal-like, not box-like; mouse is animal-like, not
   mouth-like. Even the hardest pair, horse/house (the model's weakest
   animal representation here), lands animal-like. So the *decision*
   "this is an animal, not its twin" is settled by L07, while the
   region keeps *tightening* for two more layers.

3. **The region is assembled in a sharp band, layers 5–9, and the
   early-negative separation is positive evidence, not a curiosity.**
   Below L05 separation is **negative** — and that sign is exactly what
   the minimal pairs were built to produce: each animal shares letters
   with its twin, so while spelling dominates (the lower layers) every
   animal is pulled *toward* its object and *away* from the other
   animals. The separation only goes positive once meaning overrides
   surface form. The mid-network sign-flip is therefore what proves the
   probe is measuring spelling-early / meaning-late, not some artifact
   of word length or frequency. The unit gets built, then placed.

The **final-letter** readout tells the complementary half of the story:
it never separates well (peak 0.078 at L09; minpair only 0–3/9). The
reason is built into the minimal pairs — at the final letter the
strongest signal is the identity of that letter, and the pairs were
chosen to share it (cat/hat both end "t", fox/box both "x", dog/log both
"g"). One character later, at the space, the model has *closed* the word
into a semantic summary and the category emerges. So the assembly is
visible as a single step: letters at position *i*, category at *i+1*.

## What it says about the question Ralph was chasing

The browser conversation's deep point was that smooth interpolation
explains gap-filling but not why the joints fall in useful places. This
probe is a small, direct look at the joints in a char model, and it adds
a fact the word2vec framing cannot: **the useful joint is not in the
embedding — it is manufactured mid-network, and at the moment it is
manufactured it overrides the surface form that the lower layers were
locked onto.** The model spends its first ~5 layers reading letters and
its middle layers deciding what kind of thing the letters spelled.

This lines up with the older char-model findings: the layer machinery a
char model spends building word recognition (diaries 014/015/035), the
topological framing of tokenization (074), and the L9-region copying
machinery (088). The 5–7 transition band here is plausibly the same
real-estate where word-form becomes word-meaning. It is also the
geometric counterpart of the real-word-fraction work (098/100):
"invents plausible non-words" is interpolation that has not yet been
pinned to the lexical region; this entry shows the region the full model
pins to.

## Honest limits

- Two categories, 9 words each. "Animals" is a tight natural category;
  "objects" (hat, log, box, hair, coat, mouth, wool, house) is a
  grab-bag, which is why the object block in the heatmap is looser than
  the animal block. The separation is driven mostly by the animals
  cohering, which is the intended test, but a second *tight* contrast
  category (e.g. body parts, or colors) would sharpen the between-bucket
  claim.
- One checkpoint. The natural next step is to run the identical probe
  across the saved intermediate checkpoints and watch the 5–7 band
  *form over training* — that turns "where in depth" into "when in
  training," and is the direct test of the browser conversation's
  "generalization sometimes appears suddenly" thread.
- Read-only correlational geometry, not a causal claim. Whether the
  category direction is *used* (ablate it, watch predictions move) is a
  separate experiment.

## Across training: the region forms gradually, the fold stays put (2026-06-23)

Ran the identical probe on all 25 saved checkpoints of this run
(iter 20K → 500K, every 20K plus the 500K final) with
`py/category_geometry_sweep.py` — it reuses the probe internals
unchanged, so the numbers are directly comparable. Long-form data:
`terminal_logs/category_geometry_sweep_2026_06_23.tsv`; iteration × layer
heatmaps (separation and minimal-pair count, after-word readout):
`plots/category_geometry_sweep_2026_06_23.png`.

After-word peak separation and the earliest layer reaching the full 9/9
minimal-pair decision, by iteration:

| iter | peak separation | first 9/9 layer |
|-----:|----------------:|:---------------:|
| 20K  | 0.009 | none |
| 40K  | 0.198 | none |
| 60K  | 0.323 | L13 |
| 80K  | 0.429 | none (best 8/9) |
| 100K | 0.414 | L09 |
| 200K | 0.409 | L08 |
| 360K | 0.431 | L07 |
| 500K | 0.384 | L08 |

Three findings:

1. **It forms gradually — a ramp, not a snap.** Separation rises
   smoothly 0.009 → 0.198 → 0.323 → 0.429 over 20K–80K and then
   plateaus ~0.40. There is no grokking-style phase transition in this
   measure; the category geometry eases in over the first ~100K iters.
   This is the concrete answer to the browser conversation's "sometimes
   generalization appears suddenly" thread: for category geometry, at
   this scale on natural language, it does **not**. (Sudden grokking is
   mostly a small-algorithmic-task phenomenon.)

2. **The spelling→meaning fold sits at a fixed depth (~L5–L6) the
   entire time.** In the separation heatmap the cool→warm boundary
   barely moves left–right across all 25 rows; embed–L04 stay
   spelling-locked at *every* training stage. The division of labor
   (early layers = word-form, mid layers = category) is architectural
   and present from early training. What training changes is the
   *content* behind the fold, not the fold's location.

3. **The meaning-decision migrates to earlier layers over training.**
   The first layer that reaches 9/9 moves from deep (L13 at 60K) to mid
   (L09 at 100K, L07–L08 by 360K+). The model learns to settle "animal,
   not its look-alike twin" earlier in its own depth as it trains —
   resolving the category sooner and freeing later layers.

Honest wrinkle: the 9/9 minimal-pair count is a strict binary tally over
9 outcomes and is noisy at threshold (80K shows high separation 0.429 but
only 8/9 at its best layer, while 60K caught one layer at 9/9). The
continuous separation is the reliable signal and is cleanly monotonic;
the 9/9-layer column should be read as a trend, not a per-checkpoint
exact.

## Char vs BPE on the same 18 words: assembled vs looked-up (2026-06-23)

Ran the identical probe (same words, frames, mean-centering, null,
minimal-pair test) on the best same-corpus same-architecture BPE model
`pt/bpe_uppercase_16L_1280_b2_resumed.pt` (16L, n_embd 1280, vocab 32K,
iter 168K, val 3.2652 ≈ 0.725 per-char) and compared layer-for-layer to
the char best. `py/category_geometry_compare.py` (reuses the probe
internals unchanged; the char column reproduces the numbers above
exactly). All 18 words are **single tokens** in this BPE vocab (`Ġcat`,
`Ġhat`, … — verified), so there is no letter-sharing between cat and hat
at the token level. Data: `terminal_logs/category_geometry_compare_2026_06_23.tsv`;
plot: `plots/category_geometry_compare_2026_06_23.png`.

| readout / layer | char | bpe |
|---|---:|---:|
| word's-own-token @ embed | **−0.069** | **+0.065** |
| word's-own-token @ L02 | −0.066 (0/9) | 0.107 (8/9) |
| after-word @ embed | −0.101 | 0.000 |
| after-word @ L04 | −0.034 (0/9) | **0.406 (8/9)** |
| after-word @ L07 | 0.281 (9/9) | 0.443 (9/9) |
| after-word @ L15 | 0.366 | **0.773** |

Four contrasts, all pointing the same way:

1. **In BPE the category is partly *looked up* — it is already present
   at the embedding.** Word-token separation is **+0.065 at embed** and
   reaches 8/9 minimal pairs by L02. In the char model the same readout
   is **−0.069 at embed** (there is no "cat" there, only the letter "t")
   and does not go positive until L06. This is the cleanest statement of
   "assembled vs looked-up": a BPE token *is* the word unit, so the
   king−man+woman-style category geometry can live in the embedding
   table; a char model has to build the unit first.

2. **BPE resolves the category ~3 layers earlier in depth.** At the
   after-word readout BPE is already strongly separated (0.41, 8/9) by
   **L04**, where the char model is still negative; char does not reach
   9/9 until **L07**. The char model spends its first ~5 layers building
   the word-unit that the BPE tokenizer supplies for free.

3. **BPE never dips negative — confirming the char early-negative was
   surface form.** The char model's sub-zero early layers came from the
   minimal pairs sharing letters (spelling pulls each animal toward its
   twin). In BPE, `Ġcat` and `Ġhat` are unrelated rows, so that pressure
   is absent and the curve starts at/above zero at every layer. The
   single design choice (minimal pairs) explains both models at once.

4. **BPE keeps sharpening through the top layers; char plateaus.** BPE
   separation climbs 0.27 → 0.66 → 0.77 across L12–L15 while char sits
   flat at ~0.37. (Caveat: absolute magnitudes are not strictly
   comparable across two different embedding geometries — the load-bearing
   facts are the *sign* and the *onset depth*, which are unambiguous.)

This is the geometric counterpart of diary 093's claim that tokenization
chooses the level at which the model improvises: the BPE model is handed
word-units and spends its depth refining their relations; the char model
must manufacture the unit before it can place it, and we can watch
exactly where (L5–L7) it does so.

## Causal check: the model USES the direction, it does not merely contain it (2026-06-23)

Everything above is correlational geometry. `py/category_geometry_causal.py`
asks whether the animal direction is causal, with the confounds of the naive
version fixed:
- The direction is fit on a **broad, disjoint** word list (16 animals / 16
  objects: lion, tiger, …, chair, table, …) and the causal effect is measured
  on the **held-out** minimal-pair words. (Projecting out the axis that
  separates the very words you then measure is mechanical, not behavioural.)
- The readout is the model's **output** — its next-token distribution at the
  end of an **open** context ("The cat " with nothing after, so it predicts
  freely). Premise verified: animal vs object is **0.889 LOO-decodable** from
  that output (chance 0.5), so the test is powered. Behaviour, not hidden state.
- The control is **structure-matched**: the difference-of-means of random
  balanced 9/9 partitions of the same fit-words (a random unit vector would be
  a far weaker perturbation and make the axis look special for free).
- Intervention hooks **return** the modified tensor; the ablation is verified
  to bite (|residual·d̂| at L15: 13.97 → 0.000).

The per-layer category directions are only loosely aligned (mean |cos| 0.43),
so each layer's own axis is used (one-direction-everywhere is not justified
here).

**Ablation** (project the category axis out of every layer, all positions):

| metric | baseline | ablate category-d | ablate matched-control |
|---|---:|---:|---:|
| LOO animal/object acc | 0.889 | **0.722** | 0.856 ± 0.044 |
| output separation | 0.319 | **0.201** | 0.297 ± 0.051 |

Removing the category axis hurts category-prediction ~5× more than the
structure-matched control. Real, but **partial** — the model does not collapse
to chance, because it encodes category redundantly (consistent with the loosely
aligned per-layer directions: there is no single axis to delete).

**Patching** (bounded counterfactual — set the residual's component *along* the
category axis to the other category's typical value, at every layer; cannot blow
the residual out of distribution the way additive steering does):

Output animal-score (>0 = animal-ward; axis fit on the disjoint fit-words):

| | animal-score |
|---|---:|
| animals (reference) | +1.44 |
| objects (baseline) | −1.14 |
| **objects, axis → animal (category)** | **+2.09** |
| objects, axis → group-0 (matched control) | −0.97 ± 0.08 |

Setting an object's category coordinate to the animal-typical value moves its
prediction the *whole way* from object to animal (and slightly past); the
structure-matched control barely moves. And the change is coherent and
bidirectional in the actual generated text:

```
box   base         : The box was opened and the conte...
box   axis->animal : The box was a strange and wild c...
coat  axis->animal : The coat was a strange and wild c...
bowl  axis->animal : The bowl was a striking and well-...
cat   base         : The cat was so angry that she co...
cat   axis->object : The cat was on the table and the...
dog   axis->object : The dog was still on the stairs,...
```

Push the axis toward "animal" and an object becomes "strange and wild";
push an animal toward "object" and it gets placed "on the table" / "on the
stairs". The direction the probe found is the one the model **reads** to decide
whether to continue with animate or inanimate language.

Honest limits: (1) the ablation effect is partial — the category is redundantly
encoded, so deleting one axis degrades but does not erase it. (2) Additive
steering (add α·Δ at every layer) blew the char model out of distribution
("box skkks ks ks…") even at moderate α; the bounded project-and-replace patch
was the stable tool. (3) The "animal-score" axis is a linear readout fit on the
fit-words; the worked-text examples are the model-independent check that it is
not an artifact of that axis.

## Next

1. ~~Same probe across training checkpoints~~ — DONE 2026-06-23 (above):
   gradual ramp, fixed-depth fold, decision migrates earlier.
2. Add a tight third category (body parts, colors); redo the
   between-bucket separation so it does not lean on one natural category.
3. Compare the char model to the BPE model on the identical words — in
   BPE many of these are single tokens, so the "assembled vs looked-up"
   contrast should show as the category being present *earlier* in depth.
4. ~~Causal check: ablate the category direction~~ — DONE 2026-06-23 (above):
   ablation degrades category-prediction ~5× more than a matched control, and
   bounded patching flips object↔animal continuations. The direction is used.
