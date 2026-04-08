# Diary Entry 030: Words Use SEPARATE Circuits
## Testing whether different words share Layer 4 MLP neurons

### Different words activate completely different neurons

The top 5 neurons for each word at Layer 4:

| Word | Top neurons |
|------|------------|
| said | n244, n10, n510, n98, n147 |
| through | n51, n189, n471, n332, n497 |
| strange | n438, n492, n36, n45, n473 |
| quick | n442, n425, n397, n217, n491 |
| nothing | n47, n52, n16, n395, n57 |
| would | n413, n458, n157, n468, n307 |
| himself | n456, n443, n3, n200, n359 |

NO overlap in top-5 neurons across ANY pair of words. Each word has
its own private set of strongly-active neurons.

### The activation patterns are nearly orthogonal

Cosine similarities between words' Layer 4 activation patterns range
from -0.09 to +0.17. These are essentially zero — the activation
patterns are as unrelated as random vectors would be.

The highest similarity is between "nothing" and "would" (+0.17) and
between "would" and "himself" (+0.17). These are weak correlations
that could be noise or could reflect shared sub-patterns (both "would"
and "himself" contain 'l' near the measurement position).

### Only 2 neurons are shared across 3+ words

Out of 512 neurons, only 2 are active (>0.5) in 3 or more of the 7
words tested:
- n129: active in said, quick, himself
- n239: active in through, quick, nothing

These shared neurons are likely not word-specific — they may encode
generic features like "mid-word position" or "previous character was
a vowel" that apply across words.

### What this means

**The model does NOT have a general "word recognition" circuit.** Instead,
it has approximately WORD-SPECIFIC neuron ensembles in Layer 4. Each
common word activates its own set of 75-108 neurons, with negligible
overlap to other words.

This is consistent with the model having 512 neurons per layer. With
each word needing ~80-100 neurons, the model can support roughly
5-6 simultaneous word representations at Layer 4. Since only ONE word
is being processed at position 5 at any time, this is sufficient.

### But wait — 512 neurons and thousands of words?

The model's vocabulary includes thousands of words, but each only needs
~80-100 neurons. If the neuron ensembles are truly non-overlapping,
that limits the model to ~5-6 recognizable words. That's clearly not
enough.

The resolution: neurons are not binary. From Entry 024, 62% of neurons
are in the GELU transition zone. A neuron that is "active at 0.5" for
"said" might be "active at 0.3" for "paid" and "active at 0.1" for
"maid." The same neuron participates in many word representations at
different activation levels. The word identity is encoded in the
PATTERN of activations across all 512 neurons, not in which specific
neurons are above a threshold.

The near-orthogonality of activation patterns across words supports
this: the model uses the full 512-dimensional space to distinguish
words, with each word occupying a different direction. It's not a
lookup table with one neuron per word — it's a distributed code
where each word is a unique PATTERN across the neuron population.

### Comparison with Layer 0

In Layer 0, each CHARACTER had a signature neuron (n357 for space,
n174 for 's', etc.) — essentially a one-neuron lookup table. In Layer 4,
each WORD has a distributed pattern across ~80-100 neurons with no
single "said neuron." The model uses different representational
strategies at different layers:

- **Layer 0**: Sparse, localized (one neuron per character)
- **Layer 4**: Distributed, population-coded (pattern per word)

This makes sense: there are only 39 characters (easy to assign one
neuron each) but thousands of words (must use distributed coding).
