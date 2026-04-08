# Diary Entry 024: The GELU Regime — Not Binary, but Extremely Sparse
## Most neurons live in the transition zone

### The distribution is shocking

Across all 6 layers, processing a 73-character sentence:
- **Dead** (pre-GELU < -2): 5.8% — fully off
- **Transition** (-2 to 0): **62.0%** — in GELU's nonlinear zone
- **Soft active** (0 to 1): 31.2% — on but weak
- **Hard active** (> 1): 1.0% — strongly on
- **Very strong** (> 2): 0.1% — the signature neurons

The MEDIAN pre-GELU value is -0.006 — essentially zero! This means
half of all MLP neurons sit RIGHT AT the boundary between on and off.

### The MLP is NOT a binary lookup table

I had been describing the MLP as "90% dark" based on counting neurons
with post-GELU > 0.01. But that misses the picture. 62% of neurons are
in the transition zone where GELU is nonlinear — their output is a
smooth, graded function of their input, not a binary on/off.

The 50th percentile at -0.006 means the typical neuron receives input
that is JUST BARELY below zero. GELU(-0.006) ≈ -0.003 — nearly zero
but not exactly zero. These neurons are "whispering" — contributing
tiny amounts to the residual stream that collectively may matter.

### Early vs late layers differ dramatically

**Layers 0-2**: 43-48% of neurons are active (pre-GELU > 0), but
almost NONE are strongly active (>1). The pre-GELU range is [-2.8, 1.2].
These layers operate in a regime where many neurons contribute small
amounts. The computation is distributed but mild.

**Layers 3-5**: Only 17-20% of neurons are active, but 1-3% are
STRONGLY active (>1). The pre-GELU range is much wider: [-5.3, 7.2].
These layers have a few neurons that fire very hard while most are
suppressed. The computation is SPARSE but intense.

Layer 5 has the widest range: up to 7.24. This is the neuron n391
(space detector) that fires at 7.2 — the single strongest activation
in the entire model. Layer 5 also has 10% dead neurons (permanently
off for this input), while Layers 0-2 have essentially zero dead.

### The two modes of MLP computation

**Layers 0-2: Distributed whispering**
Many neurons contribute small signals. No neuron dominates. The
computation is a gentle nudge to the residual stream from many
directions simultaneously. This matches their role: character
identity and context mixing, which are broad rather than specific.

**Layers 3-5: Sparse shouting**
A few neurons fire very hard while most are silent. The active neurons
write strong, specific signals: "this is 'said'" or "space is here."
This matches their role: word recognition and prediction, which
require specific, confident outputs.

### Implications for interpretability

This means my earlier analyses (counting neurons with activation > 0.01)
were OVERCOUNTING active neurons in Layers 0-2 and UNDERCOUNTING
the significance of the transition zone. Many of those "52 active
neurons" at position 0 in Layer 0 were just barely above zero,
contributing negligibly.

The neurons that actually matter for prediction are the 1-3% in the
"hard active" regime (pre-GELU > 1). These are the character detectors,
space detectors, and word-recognition neurons that write strong signals.

The 62% in the transition zone are implementing something more subtle:
a soft, distributed computation that adjusts the residual stream by tiny
amounts. Whether this "whispering" matters for predictions — or is just
noise around zero — I'm not sure. It might be how the model implements
the fine-grained calibration that Entry 008 revealed (the suppressor
heads that adjust probabilities by 0.01-0.10).

### The GELU is not just a gating function

In the standard interpretation, GELU gates the MLP: neurons are either
"on" (positive input) or "off" (negative input). But with 62% of
neurons near zero, the GELU is functioning more as a soft threshold
with a continuous transition. The model has arranged its computations
so that most neurons operate near the boundary, where small changes
in input can switch them between contributing and not contributing.

This suggests the model is more sensitive to input variations than a
binary gate would be. A small change in attention output could push
several transition-zone neurons from just-below to just-above zero,
changing their contribution. The GELU boundary is where the model's
sensitivity lives.
