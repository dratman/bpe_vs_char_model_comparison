# Diary Entry 025: Sensitivity — How Robust Is Word Recognition?
## Adding noise to the embedding reveals the model's margin of safety

### The experiment

I add Gaussian noise of increasing standard deviation to position 5's
embedding ('i' in "he said") and measure:
1. How the prediction P(d) changes
2. How many MLP neurons in Layer 4 flip across the GELU boundary

### Results

| Noise std | P(d) | Top pred | Neurons flipped |
|-----------|------|----------|----------------|
| 0.000 | 0.701 | 'd' | 0 |
| 0.010 | 0.689 | 'd' | 3 |
| 0.020 | 0.674 | 'd' | 8 |
| 0.050 | 0.610 | 'd' | 22 |
| 0.100 | 0.450 | 'n' | 46 |
| 0.200 | 0.142 | 'n' | 91 |
| 0.500 | 0.026 | ' ' | 156 |
| 1.000 | 0.024 | 't' | 187 |

### The word recognition breaks at noise_std ≈ 0.1

The embedding norm for 'i' is 0.869. Noise of std=0.1 is about 11% of
the signal. At that level, P(d) drops from 0.70 to 0.45 and 'n' becomes
the top prediction. The model has fallen back to its Layer 3 bigram
prediction (i→n), meaning Layer 4's word recognition has been disrupted.

At noise_std=0.05 (6% of signal), P(d) is still 0.61 and 'd' is the
top prediction. The model is reasonably robust to 5% noise but breaks
at 10%.

### GELU boundary flipping is the mechanism

At noise_std=0.05: 22 neurons flip (out of 512). This is enough to
degrade the prediction but not destroy it.

At noise_std=0.10: 46 neurons flip. Now the word recognition circuit
is significantly disrupted — too many neurons that should be contributing
to the "said" recognition are either wrongly active or wrongly silent.

At noise_std=0.50: 156 neurons flip — nearly a third of all neurons.
The prediction becomes meaningless.

The linear relationship is clear: noise_std of 0.1 flips ~46 neurons,
0.2 flips ~91 neurons (2×), 0.5 flips ~156 neurons. The neuron flipping
is approximately proportional to the noise level.

### What this tells me about the GELU transition zone

In Entry 024, I found that 62% of neurons are in the transition zone
(pre-GELU between -2 and 0). Now I can see what that MEANS functionally:
these neurons are SENSITIVE. A small perturbation (0.05-0.10 std on the
128-dim input) can flip 22-46 of them.

But the model is robust UP TO that point. At noise_std=0.02, only 8
neurons flip and P(d) barely changes (0.67 vs 0.70). The model has
about 5% margin before word recognition degrades significantly.

### The fallback cascade

As noise increases, the model falls through a hierarchy:
1. noise=0: Layer 4 word recognition → 'd' (0.70)
2. noise=0.10: Falls back to Layer 3 bigram → 'n' (0.32)
3. noise=0.50: Falls back to generic distribution → ' ' (0.56)

The noise destroys the most sophisticated processing first (word
recognition), leaving the simpler mechanisms (bigrams, then raw
character statistics) as fallbacks. The layers really DO form a
hierarchy of specificity, with each layer adding refinement that is
fragile compared to the cruder predictions of lower layers.

### Practical implication

The model's word recognition operates with a signal-to-noise ratio
of about 10:1 (embedding norm 0.87, critical noise level ~0.1). This
is not a large margin. It suggests the model is operating near the
limits of what 128 dimensions and 512 MLP neurons can reliably support.
A larger embedding dimension would presumably increase this margin,
allowing more robust word recognition.
