# Diary 114 — The Jacobian lens: "the answer forms at the last layer" is generic, not a memorization signature

Date: 2026-07-12
Prompt (Ralph): make use of Anthropic's new "Jacobian lens" / "J-space" work
(github.com/anthropics/jacobian-lens; transformer-circuits.pub/2026/workspace)
by adding layers to a Raven model and looking inside it.

## What we built

- `py/jacobian_lens.py` — a small, self-contained lens for our own tiny models.
  For every depth L (0 = raw embedding … n_layer = final residual) it reads the
  residual stream out into vocabulary two ways:
    - **logit lens**: decode directly with the model's own head, `lm_head(ln_f(h))`.
    - **Jacobian lens** (the Anthropic idea): first transport h to the final-layer
      basis with an *averaged* Jacobian `J_L = E[dh_final/dh_L]` (same-position,
      averaged over positions/windows, computed exactly by autograd on CPU), then
      decode. `J_final = I` by construction.
  Metric: rank of the *correct next token* at each layer (top-1 %, median rank,
  MRR). Also a word-by-word trace of one spot (e.g. "Nevermore").
- `py/plot_lens_comparison.py` — overlays several models, NORMALIZED to
  `MRR(L)/MRR(final)` so the *shape* (gradual build-up vs last-layer snap) is
  comparable across models with very different absolute accuracy.

## Three matched models (all 6 layers, width 16 — architecture identical)

| model | tok | text | train loss | val loss | regime |
|---|---|---|---|---|---|
| `raven_bpe_L6_d16`  | BPE  | Raven 6.2K | 0.040 nats/tok | ~16 (>>rand) | **memorized** |
| `raven_char_L6_d16` | char | Raven 6.2K | 0.37 nats/char | 5.9 (>rand 4.0) | **memorized** |
| `alice_gen_char_L6_d16` | char | Alice 146K | 1.48 nats/char | 1.63 (≈train) | **generalizes** |

All trained to a flat plateau (LR still healthy) — see logs/. Val >> random for
the two Raven models is the memorize-and-overfit fingerprint; val ≈ train for
Alice is the generalization fingerprint.

## The first (real but misleading) result

On the **BPE memorizer**, the correct next token is buried through layers 0–5
and only snaps to rank-1 at the final layer:

    logit-lens top1% by layer: 0.1 0.3 0.4 2.5 7.7 18.0  92.0
    "Nevermore" trace: '"' '"' '"' 'sorrow' 'pit' 'apping' -> Nevermore (final only)

Even the poem's most-repeated refrain word is **assembled at the last step**, not
carried up from below. This REFUTED the going-in prediction ("a memorizer holds
the answer early and just sharpens it"). Tempting to call it a memorization
signature.

## The control that killed the overclaim

Run the identical lens on all three (normalized shape, fig
`114d_lens_where_answer_forms.png`, Jacobian-lens fraction-of-final by layer):

    BPE memorizer :  0.22 0.19 0.22 0.24 0.34 0.48 1.00
    char memorizer:  0.27 0.30 0.31 0.35 0.32 0.40 1.00
    char GENERALIZER:0.28 0.36 0.28 0.23 0.25 0.47 1.00

**All three snap to 1.0 at the final layer.** The generalizer does NOT build its
answer up more gradually — it commits at the end just like the memorizers.

**Conclusion: "the exact next token only becomes top-1 at the final layer" is a
generic property of these tiny models, NOT a fingerprint of memorization.**

Sharper: the curves cluster by **tokenizer**, not by regime. The two *char*
models (memorizer and generalizer) track each other closely; the *BPE* model
sits a little apart. Whether the model memorized or generalized barely moves the
depth profile; char-vs-BPE does.

Where the real memorize/generalize difference actually lives: in the **height**
of the final layer (memorizers 83–92% top-1 = they know the exact token;
generalizer 56% = genuinely predicting English), i.e. "did it memorize" — which
we already knew. The lens revealed no new *mechanism*.

## Honest caveats about the tool on tiny models

- Logit-lens early failure is partly a **basis-mismatch artifact**: intermediate
  residuals aren't in the output basis, so a direct readout under-reads them.
- The Jacobian lens corrects some of that (orange/middle layers lift), but it
  uses a single **averaged** transport, which necessarily under-reads
  position-specific content — and `J_final = I` is exact, so a final-layer jump
  is partly baked into the method.
- These caveats apply EQUALLY to all three models, so they don't manufacture a
  false memorize/generalize split — but they do mean the dramatic "answer at the
  top" is as much about *what the lens can read* as about *what the model computes*.

## Bonus finding (genuine, and on-theme for the account)

**Depth buys memorization capacity.** At 1 layer, char CANNOT memorize The Raven
(diary 113, capacity floor ~1.6 bits/char). At **6 layers, char memorizes it
completely** (0.37 nats/char ≈ 0.53 bits/char, val exploding). So the capacity
floor of the key scene is not just about width/tokenizer — adding depth raises it.

## Verdict for the popular account

The layer-by-layer lens did **not** hand us a new memorization "scene." On models
this small it mostly shows a generic "the answer forms at the top." The real
tellable scene remains the 16-dim tokenization result (diary 113). The one honest
sentence worth keeping: *depth buys capacity* (char memorizes Raven at 6 layers,
not at 1).

Figures: `doc/figures/114_jacobian_lens_bpe_L6.png`,
`114b_jacobian_lens_char_L6.png`, `114c_jacobian_lens_gen_L6.png`,
`114d_lens_where_answer_forms.png`.
