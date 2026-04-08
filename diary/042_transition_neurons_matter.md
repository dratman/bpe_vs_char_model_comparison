# Diary Entry 042: Transition-Zone Neurons DO Matter
## Zeroing them out changes predictions significantly

### The experiment

At Layer 5, zero out neurons in different activation regimes:
- Transition zone (pre-GELU -2 to 0): ~368/512 neurons per position
- Strongly active (pre-GELU > 1): ~13/512 neurons per position

### Results at position 5 ('i' → 'd' in "said"):

| Condition | P(d) | Top pred |
|-----------|------|----------|
| Normal | 0.701 | d |
| No transition neurons | 0.284 | l |
| No strong neurons | 0.778 | d |
| Only strong neurons | 0.434 | d |

### The transition neurons are ESSENTIAL for word recognition

Removing the 368 transition-zone neurons drops P(d) from 0.70 to 0.28
and changes the top prediction from 'd' to 'l'. The word recognition
circuit is BROKEN without these neurons.

But removing the 13 strongly-active neurons barely hurts: P(d) RISES
from 0.70 to 0.78! The strong neurons are actually SUPPRESSING the
prediction slightly (consistent with Entry 008's calibration finding).

### The strong neurons are NOT the prediction mechanism

Keeping ONLY the strong neurons gives P(d) = 0.43 — worse than the
full model. The strong neurons alone are insufficient. They contribute
to the prediction, but the transition-zone neurons contribute MORE.

### How 368 barely-active neurons outweigh 13 strongly-active ones

Each transition-zone neuron outputs a tiny value (GELU of something
between -2 and 0 gives roughly -0.01 to 0.0). But there are 368 of
them. Their COLLECTIVE effect through the c_proj projection creates
a directional push in the 128-dim space that is larger than the 13
strong neurons' contribution.

This is like 368 people each pushing a car with one finger versus
13 people each pushing hard. The many small pushes can add up to more
than the few strong ones, especially when they're coordinated (pointing
in similar directions after c_proj projection).

### This changes my understanding of the MLP

I've been focusing on the "signature neurons" — the few strongly-active
neurons per position. But the real computation is DISTRIBUTED across
hundreds of weakly-active transition-zone neurons. The signature neurons
are interpretable and memorable, but they're not where the prediction
power lives.

The MLP's word recognition comes from a DISTRIBUTED whisper of hundreds
of transition-zone neurons, not from a SPARSE shout of a few strong
neurons.

### Revisiting the GELU finding (Entry 024)

In Entry 024, I found that 62% of neurons are in the transition zone
and hypothesized they might be noise. This experiment proves they are
NOT noise — they are essential. The model has deliberately arranged its
computations so that most of the prediction signal comes from many
weakly-active neurons, with strongly-active neurons providing
calibration (suppression) rather than the primary prediction.

### Why this architecture?

Having the prediction distributed across many weak neurons rather than
concentrated in a few strong ones might provide:
1. **Robustness**: No single neuron failure can break the prediction
2. **Capacity**: 368 neurons can encode more information than 13
3. **Interference avoidance**: Weak, distributed signals for different
   words are less likely to clash than strong, concentrated ones

This is consistent with the near-orthogonal word ensembles (Entry 030,
038): each word activates a unique PATTERN across hundreds of transition-
zone neurons, and these patterns are orthogonal enough to not interfere.
