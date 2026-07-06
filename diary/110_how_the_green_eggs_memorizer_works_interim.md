# Diary 110 — How the Green Eggs memorizer works (interim: attention understood, MLP still opaque)

Date: 2026-07-06

## Status of this entry

**Interim.** This records the mechanistic investigation into *how* the tiny
Green Eggs models memorize the text (following 107–109). It is deliberately
honest about where the story broke down: we localized the mystery to the MLP but
did **not** crack it. A follow-up entry (111) is expected once the MLP is
actually understood. Several of my tidy hypotheses along the way were wrong and
corrected by direct tests — those corrections are the most useful content here.

Models (all on the MacBook, gitignored): `green_eggs_char_bs256_val08` (4L,
0.83M), `green_eggs_bpe_bs85_val08` (BPE, 4L, 0.82M), a char depth sweep
`green_eggs_char_L{1,2,3}_val08` (0.24/0.43/0.63M), all char 4H/128, block 256,
val_split 0.08, 3000 iters, seed 1337.

## What we established about the mechanism

1. **Parametric, content-addressed.** From a fresh mid-book prompt ("Would you
   like them in a house?") the model continues verbatim with no preceding book in
   context — so the continuation is stored in weights, keyed on recent content.
   Not in-context induction/copying.
2. **Long, adaptive key.** The char-608 collision (identical 128-char window,
   two continuations) shows short contexts are ambiguous; up to ~256 chars are
   needed to disambiguate. Most positions need far fewer; only the self-similar
   refrains need the long key.

## Experiment 1 — component ablation (zero each sublayer, teacher-forced acc)

Char (baseline 94.9%): attn drops L0→L3 = 54.4 / 24.2 / 10.1 / 4.6;
mlp = 47.1 / 13.1 / 21.8 / 16.4. BPE (baseline 99.1%): attn = 33.9 / 7.6 / 2.0 /
1.5; mlp = 24.2 / 2.0 / 2.0 / 1.7.

- Layer 0 dominates both. In the **char** 4L model the deeper layers still carry
  real weight (mlp L2 = 22, L3 = 16); in **BPE** the deep layers are nearly
  vestigial (~2%). Read naively this said "char recruits depth to assemble; BPE
  retrieves shallow." **That reading was wrong — see Experiment 3.**

## Experiment 2 — attention range (char, last-query attention averaged over book)

Mean attention distance by layer: **L0 ≈ 90 chars (87% of mass beyond 15 back)**,
L1 ≈ 19, L2 ≈ 37, L3 ≈ 55. Layer-0 attention is the **far-reaching** one; deeper
layers are more local. (This overturned my prior guess that L0 would be *local*.)
Consistent with L0 doing the long-range "where am I" integration the collision
result demands, up front.

## Experiment 3 — depth sweep: how many layers are necessary? **ONE.**

TF accuracy / recitation by depth: 1L **98.2% / 92.1%**, 2L 97.7% / 92.0%,
3L 94.6% / 92.1%, 4L 94.9% / 92.1%. A single layer memorizes as well as (slightly
*better* than) four; accuracy drifts *down* with depth; recitation is the same
92% (train/val boundary) at every depth.

**Correction to Experiment 1's reading.** Ablation importance ≠ necessity. The
4L model *distributed* a 1-layer-sufficient solution across its layers, so
ablating any layer hurt — but the task needs only one layer. The "char recruits
deep layers for assembly" claim does **not** survive: both char and BPE are
shallow-solvable. The genuine char-vs-BPE difference is per-step compute (char's
3× longer sequences, diary 109), not required depth.

## Experiment 4 — the "fingerprint" test (1-layer model) — negative, and informative

Hypothesis (my simple story): attention builds a crisp per-position fingerprint;
the MLP just looks up fingerprint→char. Extracted the layer-0 attention output
(128-d) at 2,849 positions and tested it:

- **Decode book-position from fingerprint:** R² = 0.32, median error **545 chars**
  — position is NOT cleanly encoded.
- **Nearest-neighbor next-char lookup on fingerprints:** **74%** (model itself:
  98%). Above chance (~15%) but well short.
- Fingerprints are not well separated (nearest neighbor cosine 0.89–0.94, and
  typically 259 chars away in the book).

**Interpretation.** The attention output encodes *local text pattern*, not
*absolute position*: similar fingerprints sit far apart in the book but usually
share the next char (because the text is formulaic). So the model tracks "this
looks like an *I-do-not-like* context," not "I am at char 1500." Crucially, the
attention signal only gets you to ~74%; **the MLP carries it from 74% → 98%.**
The MLP is doing real nonlinear work — it is NOT a passive table — and that work
is the still-unexplained core of the memorization.

## Where the story stands (honest)

The clean "locate (attention) → retrieve (MLP lookup)" picture is about two-thirds
right: attention does a *soft, content-based* locate we partly understand; the
crisp memorization lives in the **MLP's nonlinear step**, which we have localized
but not opened. So: **more work is necessary to make this explainable**, and it
should target the layer-0 MLP.

## Methodological lessons (worth keeping)

- **Ablation-importance ≠ necessity.** Train-it-shallower beat inferring necessity
  from ablating a fixed model (Experiment 3 corrected Experiment 1).
- **Surprisal is the wrong lens on memorized text** — uncertainty is ~0 bits
  everywhere, so it can't localize the *work* (a full forward pass is spent even
  on foregone conclusions). [earlier attempt this session]
- Several tidy hypotheses (L0-is-local; char-needs-depth; fingerprint-is-crisp)
  were each overturned by a direct test. Trust the intervention over the story.

## Next (for diary 111)

Open the layer-0 MLP: is it a key–value memory (specific neurons for specific
contexts)? How many neurons are load-bearing? Does the fuller residual stream
(which also carries the current char's embedding) decode position/next-char
better than the attention contribution alone? These target the 74%→98% gap that
is the actual memorization mechanism. Ralph wants to genuinely understand the MLP
before the next write-up.

Related: 107–109 (memorization / BPE-vs-char), 055 (char-only dark subspace),
105 (look-up vs assemble), 076 (layer-by-layer char analysis).
