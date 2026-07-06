# Diary 111 — Inside the MLP: a sparse switch-combination memory (same mechanism in char and BPE)

Date: 2026-07-06

## Purpose

Diary 110 localized the memorization to the layer-0 MLP but left it a black box
("attention does a soft locate; the crisp memory lives in the MLP, unopened").
This entry opens it, for both a 1-layer char model and a 1-layer BPE model, and
finds they use the **same** mechanism.

Models: `green_eggs_char_L1_val08` (0.20M) and `green_eggs_bpe_L1_val08` (0.22M)
— 1 layer, 4 heads, n_embd 128, block 256 (char) / 85 (BPE), val_split 0.08,
3000 iters, seed 1337. Both memorize the training 92% (a single layer suffices —
see 110). Analysis is teacher-forced over the training region; the MLP hidden is
the post-GELU activation of `c_fc` (512 units); "switch on" = activation > 1.0.

## The two stages, measured (linear readout of the next token from each stage)

| | after ATTENTION (128) | after MLP switches (512) | model |
|---|---|---|---|
| char | 58% | **94%** | ~98% |
| BPE  | 62% | **91%** | ~98% |

The attention output is a *soft, partial* signal (~60%); the MLP is where the
token becomes cleanly decided (~90%+). This is the same "fuzzy locate → crisp
decide" split in both tokenizers. (The attention output also fails to encode
*absolute position* — R²≈0.32 for decoding book-position — so it is a
content/pattern signal, not a position coordinate; see 110.)

## The MLP is a sparse switch-combination code

| | switches ON per position | live switches (of 512) | clean single-token detectors |
|---|---|---|---|
| char | ~5 (median 4) | 446 | 37 |
| BPE  | ~10 | 437 | 42 |

- **Sparse:** only ~5 (char) / ~10 (BPE) of 512 units fire strongly at any spot.
- **A few crisp detectors:** 37 (char) / 42 (BPE) units fire almost exclusively
  at spots wanting one specific token, and their output weight (projected through
  the tied unembedding, `wte @ c_proj[:,n]`) votes for exactly that token
  (e.g. a unit that fires only before "k" and votes "k", 100% consistent).
- **Most units are feature detectors, not token-predictors:** mean key→value
  consistency across live units is only ~22% (char). So no single unit decides
  the token; the token is read out from the *combination* of the ~5–10 active
  units (that combination linearly predicts the token at ~90%+).

## Same mechanism, one honest difference

Char and BPE run the identical machine: soft content-based attention → sparse
switch-combination MLP that decides the token. The **only** substantive
difference: BPE lights up ~2× more switches per spot (~10 vs ~5). That follows
from its larger output alphabet — 182 chunks vs 40 letters. More possible
answers require a bigger switch-combination to specify one. Bigger alphabet →
bigger "chord," same instrument.

This explains diary 109's result (char and BPE memorize equally well): they
memorize the *same way*. The char-vs-BPE difference is granularity — char makes
~3× more decisions (one per letter), each a smaller ~5-switch chord; BPE makes
fewer, coarser decisions (one per chunk), each a larger ~10-switch chord — not a
difference in mechanism. It also finishes off the earlier "assemble vs retrieve /
char needs depth" idea: at the circuit level both are the same shallow
switch-memory (and 110 already showed depth is unnecessary).

## The memorization mechanism, stated plainly

1. **Locate (attention):** broadly integrate the recent context into a fuzzy
   "what kind of spot is this" summary (~60% of the answer, content- not
   position-based, long-range — L0 attention reaches ~90 chars back, 110).
2. **Decide (MLP):** a sparse handful of ~5–10 "switches" (of 512) turn on;
   their specific combination points at one token (~90%+). A minority of switches
   are crisp single-token detectors; most are text-pattern features that vote in
   combination.

The book is thereby stored as many rules of the form "when *these* few switches
are on, the next token is ___," distributed across ~440 switches, ~5–10 used at
a time.

## Caveats

Toy models (0.2M) on a 3.3 KB text; teacher-forced over the memorized region.
The token-vote projection ignores the final LayerNorm (approximation). The BPE
linear readout has fewer samples (946) than char (2849), so its 91% carries more
variance. These are intuitions/mechanism sketches, not scaling claims.

## Follow-on (in progress)

To make this concrete and falsifiable, we are writing two from-weights
reimplementations — `py/char_model_by_hand.py` and `py/bpe_model_by_hand.py` —
that reproduce each model's forward pass in plain numpy, structured around the
two stages, and verified to match the real model's greedy output on the standard
prompt and on new prompts.

Related: 107–110 (this arc), 109 (memorization tie), 055 (char dark subspace),
105 (look-up vs assemble), 076 (layer-by-layer).
