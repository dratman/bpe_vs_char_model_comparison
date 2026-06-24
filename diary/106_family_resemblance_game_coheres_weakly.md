# Diary 106 — A no-common-attribute category ("game") coheres, but weakly: family resemblance in the geometry

Date: 2026-06-23

## Where this came from

Diary 105 showed "animal" — a category whose members share many correlated
attributes (alive, moves, breathes) — is a single coherent, *causal* direction
in the char model. In conversation Ralph made the sharpening point that a
category need not be a single attribute, and that Wittgenstein's real subject
was the opposite case: **"game"** (board games, ball games, card games, word
puzzles) has *no* single common attribute — only overlapping family
resemblances. Chess, football, whist, and a riddle share no concrete thread.
The question: does the model nonetheless bind them into a region, or does
"game" fragment into disconnected sub-clusters the way the *concept* does?

## Design (and the confounds it has to dodge)

`py/category_family_resemblance.py`. Three sets of 12 words each, read at the
after-word position in 4 open contexts (same readout as the 105 causal test),
per layer, mean-centered cosine:

- **animal** (unified category, calibration): wild {lion, tiger, wolf, fox},
  farm {horse, cow, sheep, goat}, bird {owl, hen, goose, hawk}
- **game** (family resemblance, test): board {chess, draughts, billiards,
  dominoes}, field {football, tennis, golf, croquet}, parlour {cards, whist,
  riddle, puzzle}
- **contrast** (neutral baseline): river, mountain, letter, window, candle,
  church, valley, silver, kitchen, garden, bridge, cottage

Headline metric = **coherence above parts** with a **permutation null**: pool a
category's 12 words + the 12 contrast words, and ask whether the true grouping's
(within-category − to-contrast) cosine beats a null that randomly relabels 12 of
the 24 as "category". This asks whether the category is a *privileged* grouping
in the geometry. Animal calibrates the scale.

Two confounds shaped the design (both flagged in review):
- **Fragmentation (within-subtype − within-category) was demoted**, because the
  game subtypes were chosen to be maximally disparate, so it partly re-measures
  the word picks rather than the model.
- **Frequency is the load-bearing confound, and it is asymmetric.** Game words
  are 10–300× rarer than the animals (dominoes 176, croquet 263, whist 404 vs
  horse 55k, lion 9k). Rare words have noisier representations that *inflate*
  scatter — so a "game fails to cohere" result would be uninterpretable (could be
  rare-word noise), but a "game coheres *anyway*" result is robust (the noise
  fights against it). That is exactly why coherence, not fragmentation, is the
  headline: the confound makes the test conservative.

No classification accuracy (24 points in 1280-d are always separable). The 2D
PCA picture is illustration; the nulls carry the claim. n is small (12/category).

## Result — graded, and it lands where Wittgenstein predicts

| | animal | game |
|---|---:|---:|
| coherence above parts (L09) | **0.425** (z 14.8) | **0.135** (z 8.1, p<0.001) |
| cross-subtype cosine (L09) | +0.180 | −0.003 |
| to-contrast cosine (L09) | −0.222 | −0.109 |
| binds across subtypes by | **+0.402** | **+0.106** |

1. **"Game" *does* cohere above chance — there is a real thread.** Game
   coherence is significant from L04 on and sits at z ≈ 7–8 (p < 0.001) through
   the mid-late layers. The model binds chess/football/whist/riddle together
   more than a random 12-word grouping — despite their sharing no concrete
   attribute and despite being rarer words (so this is conservative). I would
   not have bet on a clean positive here.

2. **But it is weak — about a third of animal's.** Game coherence ~0.14 vs
   animal ~0.42; game sits at **~32%** of animal's coherence at L09.

3. **The thread is genuinely family-resemblance-shaped: no common core.** The
   load-bearing number is cross-subtype binding. Animal members bind across
   subtypes strongly (+0.40): a cow is genuinely similar to an owl. Game members
   have **cross-subtype cosine ≈ 0** (chess and football are *not* alike) — yet
   still sit above the contrast baseline (−0.003 vs −0.109). So there is a faint
   family thread but **no shared centre**: members overlap pairwise without a
   common attribute pulling them together. That is Wittgenstein's picture
   rendered in geometry.

4. **The binding is constructed across layers.** Game coherence is negative at
   the embedding and climbs to ~0.14 by L08 (animal is high throughout). So the
   network *builds* the thin game-thread; it is not a word-form artifact. (The
   high animal coherence at the embedding *is* partly a word-length artifact —
   animals are short words, so they cluster by trailing-space position before
   any content mixing — which is another reason to read the mid-late layers, not
   the input.)

5. **The 2D picture (illustration) shows it directly.** All 12 animals collapse
   into one tight clump; the 12 games scatter across a whole region — field
   games (tennis, golf, football) in one corner, board games (chess, billiards,
   draughts) in another, parlour games (cards, riddle, puzzle) in a third, with
   dominoes (rarest, 176 occurrences) stranded as an outlier. Animal is a blob;
   game is a constellation.

## Dimensionality: inconclusive (honest)

The plan was to report intrinsic dimensionality (participation ratio) as a clean
operationalization of Ralph's "one attribute = one dimension" — predicting game
spreads across more dimensions than animal. **It did not give a clean answer.**
In the concept-bearing layers (L08–L10) game and animal have *similar*
participation ratios (~8.9 vs ~9.4). At the very top layer game's PR collapses
(4.2 vs animal 8.3) — but that is almost certainly a rare-word artifact (rarer
words converge to generic next-character predictions near the output), the
*opposite* of the predicted direction, so it is not evidence for "game = many
dimensions." Dimensionality is too entangled with frequency here to read. The
coherence result stands; the dimensionality one is reported and set aside.

## What it means

The clean diary-105 story — a category is a single coherent direction the model
can look up and we can dial — is the *easy* case. It holds for "animal," whose
members share correlated attributes that collapse onto one axis. Pushed to the
case Wittgenstein actually cared about, the story **degrades exactly as
predicted but does not vanish**: "game" is a real-but-faint region with no
common centre, its members bound by overlapping resemblances rather than a
shared attribute. The model manufactured *some* "game"-ness from how the words
are used in text — more than a no-common-attribute concept might lead you to
expect — but nothing like the tight, dialable direction a shared-attribute
category gets. Family resemblance is visible in the geometry as a thin thread
over a scattered constellation.

So: where does "game" sit on the random↔animal spectrum? About one-third of the
way toward animal. A real grouping, a weak one, and shaped like family
resemblance rather than a definition. That is the honest, graded answer.

## Limits and next

- Frequency and length confounds are real; the coherence test was built to be
  conservative against frequency, but a frequency-matched replication (rare
  animals vs rare games, or BPE single-tokens) would harden it.
- n = 12/category, 4/subtype — lean on the nulls and the cross-layer profile,
  not point estimates.
- Natural sequels: (a) the same probe on the BPE model (games as single tokens,
  no rare-word letter-assembly noise); (b) a third intermediate category (one
  with a *loose* common attribute, e.g. "tool" or "weapon") to fill in the
  spectrum between animal and game; (c) is the thin game-thread *causal* the way
  animal's was — does patching it change predictions, or is it too weak to be a
  lever?
- Data: `terminal_logs/family_resemblance_2026_06_23.tsv`; figure
  `plots/family_resemblance_2026_06_23.png` (local). See diary 105 for the
  animal baseline this is measured against.
