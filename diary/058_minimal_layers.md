# Diary Entry 058: How Few Layers Can the Model Work With?
## Answer: it degrades sharply — every layer matters

### Results

| Configuration | Accuracy | Avg P(true) |
|--------------|----------|-------------|
| All 6 layers | 67% | 0.550 |
| Skip L1 | 31% | 0.157 |
| Skip L1,L2 | 17% | 0.112 |
| Skip L1-3 | 7% | 0.084 |
| Only L0+L5 | 4% | 0.070 |
| Only L5 | 7% | 0.071 |
| Only L0 | 0% | 0.001 |
| L0+L3+L5 | 13% | 0.107 |
| L0+L2+L5 | 18% | 0.106 |
| L0+L4+L5 | 7% | 0.084 |

### The model degrades SHARPLY with each removed layer

Removing just Layer 1 cuts accuracy from 67% to 31%. Removing two
layers drops it to 17%. Three layers gone: 7%. The degradation is
steep and immediate.

### Even 3-layer subsets barely work

The best 3-layer subset is L0+L2+L5 at 18% accuracy — barely above
random for a 39-character vocabulary (random = 2.6%). The model
cannot work with fewer than about 5 layers.

### This answers the question: NO, the model could not have fewer layers

Despite the 7-layer model showing that the extra layer is used mainly
for calibration (a modest improvement), removing layers from the
6-layer model is CATASTROPHIC. Every layer is load-bearing in the
trained model.

### BUT: this tests the wrong thing

This test takes a MODEL TRAINED WITH 6 LAYERS and removes layers
from it. The layers were trained to work together in sequence, so
removing one breaks the chain. A model TRAINED with 4 layers from
scratch might learn to do with 4 layers what the 6-layer model does
with 6 — it would organize differently.

The question "could the model work with fewer layers?" really requires
TRAINING smaller models, not amputating layers from a trained model.
The amputation test shows that the trained model's layers are
interdependent, not that fewer layers can't work.

### What we CAN conclude

1. In a trained model, every layer is essential — they form a chain
   where each depends on the previous.
2. The most critical middle layer is L2 (removing it hurts more than
   removing L3 or L4 in the 3-layer subsets).
3. L0 alone is useless (0% accuracy), L5 alone gets 7% — the final
   layer can do SOMETHING even without setup, but not much.
4. The right test for "can fewer layers work" is to train a 4-layer
   or 5-layer model from scratch on the same data.
