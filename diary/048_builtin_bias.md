# Diary Entry 048: The Model's Built-In Default Prediction
## The layer norm bias creates a baseline character distribution

### The layer norm bias IS the default prediction

When no meaningful signal reaches the final layer norm, the bias term
alone produces a prediction. This "null prediction" is:

| Character | Bias P | Corpus P | Ratio |
|-----------|--------|----------|-------|
| 'e' | 0.066 | 0.101 | 0.65 |
| 's' | 0.066 | 0.051 | 1.30 |
| 'i' | 0.059 | 0.055 | 1.06 |
| ' ' | 0.053 | 0.172 | 0.31 |
| 't' | 0.050 | 0.074 | 0.68 |

### It's NOT a uniform distribution

The bias produces a peaked distribution favoring vowels and common
consonants. But it does NOT match the corpus frequency distribution.
The biggest discrepancy: SPACE has bias P=0.053 but corpus P=0.172.
The model's default UNDER-predicts space by 3×.

### Why the default under-predicts space

Because of weight tying and the frequency-norm effect (Entry 020):
space has the SMALLEST embedding norm (0.781), so even when the bias
pushes slightly toward space, the small norm reduces its logit.

The bias is fighting the norm: it tries to give space its fair share,
but space's small embedding means the dot product is always reduced.
The model compensates by explicitly predicting space when appropriate
(via the space-detector neurons in L5), rather than relying on the
default.

### The default is a safety net

When the model is uncertain (like at word boundaries), it falls back
toward this default distribution. The default favors common letters
(e, s, i, t) — a reasonable fallback when context is ambiguous.

The default distribution is flatter than the corpus distribution
(max P = 0.066 vs max corpus P = 0.172). This is appropriate: when
uncertain, the model should spread probability across options rather
than committing to the most common character.

### The layer norm weight has norm 19.0

This is a large amplification factor. It means the layer norm scales
up the residual stream by roughly 19×, which amplifies the logits.
The bias (norm 2.3) adds a small offset that creates the default
prediction. The weight (norm 19.0) amplifies the directional signal
from the residual stream.

The 19× amplification explains the large logits I observed: a residual
vector of norm 4 becomes a logit vector with magnitudes up to
4 × 19 × 1 = 76 (though spread across 128 dimensions, so individual
logits are smaller).

### Implication: the model has three prediction channels

1. **Default** (layer norm bias): a gentle distribution favoring
   common characters, always present
2. **Residual direction** (from 6 layers of processing): the specific
   prediction based on context
3. **Residual magnitude** (amplified by layer norm weight): the
   CONFIDENCE of the prediction — larger residual = more peaked logits

The final prediction is the sum of all three: default + direction ×
magnitude. When the model is confident, the direction overwhelms the
default. When uncertain, the default provides a reasonable fallback.
