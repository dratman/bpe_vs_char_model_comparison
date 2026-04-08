# Diary Entry 064: The Cat in the Hat — Pedagogical Minimum
## 28K parameters on a 7K character Dr. Seuss text

### The model

2 layers, 1 attention head, dim=32, MLP width=128 neurons.
Trained on lowercase Cat in the Hat (33-character vocab, 7099 chars).
28,608 parameters. Val loss 1.554.

### The principles are visible even here

**1. Layer 0 MLP has character-specific neurons** ✓

Each character activates different top neurons:
- 't': n6, n1, n63 (22/128 active)
- 'c': n119, n116, n100 (32/128 active)
- 'i': n120, n38, n6 (28/128 active)
- 'h': n125, n47, n26 (22/128 active)

Different top neurons for each character. Same sparse detection
pattern as the 1.2M param Gutenberg model.

**2. Logit lens: identity → prediction** ✓

At position 4 ('c' in "cat"):
- After L0: P(a)=0.024, top prediction is 'c' (0.40) — character identity
- After L1: P(a)=0.798, top prediction is 'a' (0.80) — word recognition!

One layer: "I am c." Next layer: "a follows." The identity-to-prediction
transition happens in a SINGLE STEP with only 2 layers.

**3. Attention evolves between layers** ✓

- L0: diffuse (self=0.20, prev=0.15, first=0.18) — no specialization
- L1: prev-dominant (self=0.27, prev=0.32, first=0.14) — reads previous char

At Layer 1, position 4 ('c') attends 45% to the SPACE before it.
This is the word-boundary reading pattern — exactly what Layer 4 did
in the big model.

**4. Word recognition works** ✓

The model predicts correctly:
- c→a (0.80) — knows "cat"
- a→t (0.43) — knows "cat"
- t→h (0.45/0.76) — knows "the" and "that"
- h→e (0.59/0.77) — knows "the" and "he"
- h→a (0.50) — knows "hat" and "had"
- n→SPACE (0.97) — "in" is a complete word

### What this 28K model DOESN'T have

- No dark subspace (vocab 33 > dim 32)
- No suppressor layers (only 2 layers, both needed for prediction)
- No FST broadcasting (only 1 head, can't specialize)
- Limited context (block_size=64)

### For pedagogy

This model demonstrates:
1. Character detection in Layer 0 MLP (sparse neuron activation)
2. The identity → prediction transition (logit lens shows it clearly)
3. Word recognition from character patterns (c→a→t predicted correctly)
4. Attention reading the previous character / word boundary

All in a model small enough to print every weight. With 128 MLP neurons
and 32-dimensional embeddings, a human could in principle trace every
computation by hand.

### The bigger 3L dim=48 model adds:

- 15 dark dimensions (dim 48 - vocab 33 = 15)
- A third layer for potential suppression/calibration
- 192 MLP neurons per layer (more word recognition capacity)
- But severely overfits (train loss 0.3, val loss 1.55)

### Which to use for teaching?

The 2L1H dim=32 model (28K params) is best for teaching the MECHANISM.
It has one head (no need to explain multi-head), two layers (minimum
for identity→prediction), and every principle is visible.

The dim=48 model is better for teaching about the dark subspace and
the effects of depth. But it overfits so badly that its predictions
are memorized, not generalized.
