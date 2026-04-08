# Diary Entry 053: Layer 5 Always Dominates the Prediction
## Confirmed across 5 different words

### Per-layer contribution to the correct next-character logit

| Word | L0 | L1 | L2 | L3 | L4 | L5 | Total |
|------|-----|-----|-----|-----|-----|------|-------|
| said (i→d) | -0.15 | -0.00 | +0.45 | +0.27 | +0.22 | +1.14 | +1.93 |
| church (c→h) | +0.01 | +0.30 | +0.09 | +0.12 | +0.33 | +1.47 | +2.31 |
| would (l→d) | +0.02 | -0.02 | +0.15 | -0.05 | +1.02 | +2.45 | +3.57 |
| nothing (n→g) | +0.06 | -0.25 | +0.17 | +0.45 | +0.20 | +2.16 | +2.79 |
| through (g→h) | +0.12 | +0.27 | -0.07 | +0.16 | +0.39 | +2.28 | +3.15 |

### Layer 5 contributes the majority in EVERY case

L5's share of the total prediction:
- said: 1.14 / 1.93 = 59%
- church: 1.47 / 2.31 = 64%
- would: 2.45 / 3.57 = 69%
- nothing: 2.16 / 2.79 = 77%
- through: 2.28 / 3.15 = 72%

Layer 5 contributes 59-77% of every prediction. It is consistently
the dominant layer, regardless of the word or position.

### Layer 4 is the second contributor, but variable

L4 ranges from +0.20 ("nothing") to +1.02 ("would"). For "would",
L4 is remarkably strong — nearly half as much as L5. For "nothing",
L4 barely contributes. This is consistent with Entry 035: "would"
is one of only two words recognized at L4.

### Layer 2 contributes to "said" specifically

L2 adds +0.45 to "said" but only +0.09 to +0.17 for other words
(and -0.07 for "through"). The large L2 contribution to "said"
(Entry 052) is NOT general — it's specific to contexts where the
previous word strongly predicts the current word (like "he" predicting
"said").

### Layers 0-1 are near-zero

L0 contributes -0.15 to +0.12 — small and sometimes negative.
L1 contributes -0.25 to +0.30 — variable but not consistently helpful.
These layers prepare the residual stream for later processing but
don't directly contribute to the correct prediction.

### The consistent pattern

1. L0-1: minimal or negative contribution (preparation)
2. L2-3: small positive contribution (context gathering, sketch)
3. L4: variable contribution (easy words get recognized here)
4. L5: dominant contribution, always 59-77% (final recognition)

The model really is a bottom-heavy architecture where most of the
"productive" computation (the part that actually pushes toward the
right answer) happens in the final layer. The earlier layers contribute
by providing the RIGHT INPUT to Layer 5, not by directly predicting.

### Implication for the 7-layer model

If Layer 5 contributes 60-77% of every prediction in the 6-layer model,
the 7-layer model might allow this load to be SHARED between Layer 5
and Layer 6. If each contributes 40-50%, the model would be more robust
(less dependent on a single layer) and potentially more capable
(each layer specializes in a subset of predictions).
