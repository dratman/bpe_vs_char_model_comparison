# Diary Entry 026: Layer Ablation — Which Layers Are Load-Bearing?
## Skip, isolate, and progressively add layers

### Skipping individual layers

| Skipped | P(d) | Top prediction | Interpretation |
|---------|------|----------------|----------------|
| none | 0.70 | d (0.70) | Full model |
| L0 | 0.01 | a (0.26) | **CATASTROPHIC** — model breaks |
| L1 | 0.21 | l (0.38) | Degraded but partially works |
| L2 | 0.30 | n (0.42) | Falls back to bigram |
| L3 | 0.01 | m (0.52) | **CATASTROPHIC** — model breaks |
| L4 | 0.83 | d (0.83) | **IMPROVES** without L4! |
| L5 | 0.19 | d (0.19) | Works but weak |

### The shocking results

**Layer 0 is essential.** Without it, the model collapses to P(d)=0.007
and predicts 'a'. Layer 0's character detection MLP is the FOUNDATION —
everything else depends on it. Without knowing what character is at each
position, nothing works.

**Layer 3 is essential too.** Without it, the model predicts 'm' (0.52)
with P(d)=0.007. This is surprising — I showed earlier that Layer 3's
prediction is often WRONG (it says 'n' not 'd'). But removing it breaks
the model entirely. Layer 3 must be providing information that Layer 4
and 5 depend on, even if Layer 3's own prediction is wrong.

**Layer 4 is OPTIONAL — and removing it HELPS!** Without Layer 4,
P(d) goes UP from 0.70 to 0.83. The model is MORE confident without
Layer 4. This is consistent with Entry 008: Layer 4 contains suppressor
mechanisms (like L3H3) that calibrate confidence downward. Without those
suppressors, the raw prediction is stronger.

But wait — this doesn't mean Layer 4 is useless. It means Layer 4 is
doing calibration that HURTS performance on this specific example but
probably HELPS on average (preventing overconfident wrong predictions
across the corpus).

### Progressive addition confirms the pipeline

| Layers | P(d) | P(n) | Top pred | Stage |
|--------|------|------|----------|-------|
| embed only | 0.00 | 0.00 | i (1.00) | Raw character identity |
| L0 | 0.00 | 0.00 | i (0.99) | Still character identity |
| L0-1 | 0.00 | 0.01 | i (0.77) | Character fading slightly |
| L0-2 | 0.03 | 0.11 | i (0.42) | Starting to predict |
| L0-3 | 0.06 | 0.58 | n (0.58) | **Bigram prediction** |
| L0-4 | 0.19 | 0.15 | d (0.19) | **Word recognition begins** |
| L0-5 | 0.70 | 0.10 | d (0.70) | **Full word recognition** |

This is the logit lens data from Entry 012 but now with confirmation
from the layer ablation. The progression is:

1. L0-2: Still echoing the input character 'i'
2. L0-3: Bigram prediction kicks in → 'n' (0.58)
3. L0-4: Word recognition starts → 'd' overtakes 'n' (0.19 vs 0.15)
4. L0-5: Word recognition completes → 'd' dominates (0.70)

### Each layer roughly TRIPLES P(d)

- After L3: P(d) = 0.063
- After L4: P(d) = 0.191 (3.0× increase)
- After L5: P(d) = 0.701 (3.7× increase)

Each word-processing layer multiplies the correct prediction's
probability by about 3×. This is the MLP writing 'd'-aligned vectors
to the residual stream at each layer, with each layer's contribution
adding to the previous.

### Layer 3's hidden role

Layer 3 produces a WRONG prediction ('n') but is ESSENTIAL for the
model to work. Why? Because Layer 3 isn't just making a prediction —
it's ORGANIZING information in the residual stream. Its attention
gathers word characters, and its MLP processes them, even if the
resulting logits point to 'n'. The processed information that Layer 3
leaves in the residual stream is what Layer 4 and 5 need to make
their corrections.

In other words: Layer 3's prediction is a byproduct. Its real
contribution is the REPRESENTATION it writes — the way it arranges
information about s, a, and i in the 128-dimensional space so that
Layer 4's neurons can recognize the pattern.

### Summary of layer criticality

- **L0**: Essential foundation. Character identity. Cannot be skipped.
- **L1**: Important but not fatal. Global context broadcast.
- **L2**: Important. Begins the self+prev attention pattern.
- **L3**: Essential. Organizes word information, even though its own
  prediction is wrong. L4 and L5 depend on its representation.
- **L4**: Calibration layer. Actually HURTS specific predictions but
  probably helps average performance. Can be skipped without breaking.
- **L5**: Critical for final confidence. 3.7× boost to P(d). Handles
  long words and long-range context.
