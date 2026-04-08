# Diary Entry 056: 7-Layer Prediction Decomposition
## How the extra layer changes the internal economy

### The 7L model's per-layer d contributions

| Layer | Attn | MLP | Total | Cumulative |
|-------|------|-----|-------|-----------|
| embed | — | — | -0.14 | -0.14 |
| L0 | +0.02 | -0.13 | -0.11 | -0.25 |
| L1 | +0.16 | -0.00 | +0.16 | -0.09 |
| L2 | +0.28 | -0.02 | +0.27 | +0.18 |
| L3 | +0.14 | +0.24 | +0.38 | +0.56 |
| L4 | -0.18 | +0.46 | +0.29 | +0.84 |
| **L5** | **-0.05** | **-0.08** | **-0.13** | **+0.72** |
| L6 | -0.13 | +1.85 | +1.72 | +2.43 |

### Layer 5 in the 7L model is a NET SUPPRESSOR

Both L5's attention (-0.05) and MLP (-0.08) push AWAY from 'd'. The
total effect is -0.13, reducing the cumulative from +0.84 to +0.72.
This is why the logit lens showed P(d) dropping from 0.47 to 0.38
at Layer 5.

In the 6L model, L5 contributed +1.14 (the dominant positive push).
In the 7L model, L5 contributes -0.13 (a suppressor). The SAME
LAYER NUMBER does a completely different job when there's another
layer after it.

### L6 does the heavy lifting

L6 MLP contributes +1.85 — even larger than the 6L model's L5 MLP
(+1.25). The final layer's MLP is STRONGER in the 7L model. With
the extra layer providing more processed input, the final MLP can
write a more confident prediction.

### The 7L model produces a larger total signal

6L total cumulative dot(d): +0.10 (pre-layer-norm)
7L total cumulative dot(d): +2.43 (pre-layer-norm)

The 7L model produces 24× more d-aligned signal! This seems extreme,
but the layer norm will normalize the vector before computing logits,
so the actual logit difference is smaller. Still, the 7L model is
much more internally "certain" about its prediction.

### The redistribution pattern

Comparing 6L and 7L, the prediction contribution shifts:

| Layer | 6L contribution | 7L contribution |
|-------|----------------|----------------|
| L0 | -0.15 | -0.11 |
| L1 | -0.00 | +0.16 |
| L2 | +0.45 | +0.27 |
| L3 | +0.27 | +0.38 |
| L4 | +0.22 | +0.29 |
| L5 | +1.14 | -0.13 |
| L6 | — | +1.72 |

Key changes:
- L1 becomes positive (+0.16 vs -0.00) — it contributes more
- L3 becomes stronger (+0.38 vs +0.27) — earlier recognition
- L5 flips from +1.14 to -0.13 — becomes a suppressor
- L6 takes over as the predictor (+1.72)

### The general principle: non-final layers can afford to suppress

In the 6L model, L5 HAD to predict correctly because nothing came after
it. In the 7L model, L5 can afford to suppress because L6 will make the
final call. This frees L5 to do calibration work (adjusting the
distribution, preventing overconfidence) rather than prediction.

This is a general architectural principle: THE LAST LAYER IS ALWAYS
THE PREDICTOR. All other layers can suppress, calibrate, or reorganize
because they know another layer follows. Only the final layer is
forced to commit to a prediction.

This explains why "would" moved from L4-recognition in the 6L model
to L6-recognition in the 7L model — with L6 available, the model
delays commitment to the very end, using earlier layers for
preparation rather than prediction.
