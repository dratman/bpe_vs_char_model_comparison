# Diary Entry 052: Who Predicts 'd'? — Per-Component Decomposition
## Tracing the d logit through all 13 components

### The decomposition

The residual stream at position 5 is the SUM of 13 components:
embedding + 6 attention outputs + 6 MLP outputs. I computed each
component's dot product with the 'd' embedding to see which ones
push toward 'd' and which push away.

### The cumulative story of the 'd' prediction

| Step | Cumulative dot(d) | What's happening |
|------|------------------|-----------------|
| embed | -0.16 | 'i' embedding has negative d-alignment |
| +L0 | -0.31 | L0 pushes AWAY from d (detecting 'i') |
| +L1 | -0.32 | L1 barely changes anything |
| +L2 | +0.13 | L2 REVERSES the direction! (+0.45 from attn) |
| +L3 | +0.41 | L3 continues pushing toward d |
| +L4 | +0.63 | L4 MLP adds +0.29 |
| +L5 | +1.77 | L5 MLP adds +1.25 — the MASSIVE final push |

### Layer 5 MLP contributes 71% of the total d signal

| Component | dot(d) | % of total |
|-----------|--------|-----------|
| L5 MLP | +1.254 | 71% |
| L2 attn | +0.479 | 27% |
| L4 MLP | +0.293 | 17% |
| L3 attn | +0.259 | 15% |
| L0 embed | -0.160 | -9% |
| L0 MLP | -0.112 | -6% |
| L5 attn | -0.112 | -6% |
| Others | small | ~0% |

(Percentages don't sum to 100% because positive and negative
contributions partially cancel.)

### The surprising role of Layer 2 attention

L2 attention contributes +0.479 to the d logit — the SECOND largest
positive contribution! In Entry 046, I found that L2 reads the
PREVIOUS word's characters. Here, it's reading "he " and writing
a signal that already points toward 'd'.

How? L2's attention at position 5 reads back to 'h' and 'e' from
the word "he". The value vectors from those positions, when combined
with L2's output projection, produce a vector aligned with 'd'. This
means L2 has learned that after the word "he", certain characters
(including 'd') become more likely in the next word.

This is CROSS-WORD prediction: L2 uses the previous word to influence
the current word's prediction. It's not just gathering context — it's
already contributing to the prediction at a very early stage.

### Layer 5 attention HURTS the prediction

L5 attention contributes -0.112 to the d logit. This is surprising
because L5 is the primary prediction layer. But its ATTENTION hurts
while its MLP helps enormously (+1.254).

This is consistent with the Entry 008 finding about suppressor heads:
L5's attention reads context and writes a signal that slightly reduces
confidence, while L5's MLP writes the strong prediction. The attention
and MLP work in OPPOSITION within Layer 5, with the MLP winning by
an order of magnitude.

### The model overcomes initial negative d-alignment

The embedding and Layer 0 BOTH push AWAY from 'd' (total -0.31). The
model starts in the wrong direction and must overcome this to predict
'd'. It does so through layers 2-5, which collectively add +2.08 of
d-aligned signal, overwhelming the initial -0.31.

This means EVERY PREDICTION requires the model to overcome its initial
character-identity signal. The embedding points at the INPUT character
('i'), and the model must redirect toward the OUTPUT character ('d').
This redirection is the fundamental computation of prediction, and it
happens through the residual stream's accumulation of layer contributions.

### The essential components for 'd' prediction

If I had to keep only 3 components:
1. L5 MLP (+1.254) — the word recognition signal
2. L2 attn (+0.479) — cross-word context
3. L4 MLP (+0.293) — within-word processing

These three components alone give dot(d) = +2.026, which would
produce the correct prediction. Everything else could be removed
and the model would still predict 'd' (though with different confidence
calibration).
