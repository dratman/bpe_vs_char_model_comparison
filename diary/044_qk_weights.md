# Diary Entry 044: Reading the QK Weight Matrices
## What are the attention heads structurally looking for?

### Caveat: this analysis is incomplete

I computed Q(char) · K(char) using the RAW character embeddings,
ignoring layer normalization and the accumulated residual stream.
The actual QK interactions depend on the PROCESSED representations at
each layer, not the raw embeddings. So these scores represent what
the heads would do if they saw pure character embeddings — a
structural baseline, not the actual runtime behavior.

### The QK scores are TINY

All the strongest QK interactions are 0.0 to 0.2. These are very
small scores. For comparison, the actual attention weights I observed
(like Layer 4 H2 attending to 's' at 0.98) require much larger QK
scores. The large actual scores come from the PROCESSED representations,
not from the raw character embeddings.

This confirms Entry 036's finding: the attention heads are reading
processed representations, not character identities. The pure
character→character interaction is weak; the strong attention patterns
emerge from what Layers 0-3 have written to the residual stream.

### Layer 0: Punctuation-seeking

The strongest QK pairs in Layer 0 all involve PUNCTUATION:
- H0: queries from '?','!' seek '(', '"'
- H1: queries from '"','?',"'",')','.' seek '\n'
- H2-3: mostly flat, slight preference for '('

This makes sense: Layer 0 is looking for punctuation marks, which
carry structural information (sentence boundaries, dialogue markers).
But the scores are so small (0.0-0.2) that this preference is very
mild — the diffuse attention pattern I observed is consistent with
these weak QK preferences.

### Layer 5: Period-seeking

At Layer 5, ALL FOUR heads have their strongest QK interaction
pointing toward '.':
- H0: ' ' seeks '.' (0.2)
- H1: '.' seeks '\n' (0.1) — but also everything seeks '\n'
- H2: ' ', ',' seek '.' (0.2)
- H3: ' ', ',' seek '.' (0.1)

This is the mechanism behind the quotation-mark circuit (Entry 007)!
Layer 5 heads are structurally biased toward reading back to periods
and sentence-boundary punctuation. When I found that Layer 5 reads
closing quotation marks, it was this same mechanism — the heads are
trained to seek punctuation positions, and quotation marks activate
similar QK patterns.

### The weights alone don't explain runtime behavior

The QK weight analysis shows STRUCTURAL BIASES but doesn't explain the
sharp attention patterns I observed at runtime (like H2 attending to
's' at 0.98). Those sharp patterns emerge from the interaction between
the QK weights and the processed residual stream. The weights create
a tendency; the processed representations determine the specific
attention pattern.

This means the attention heads' behavior can't be fully understood
from the weights alone. The weights create a POTENTIAL for certain
interactions, but the actual interactions depend on what previous
layers have written to the residual stream. It's a dynamic system,
not a static circuit.

### What this means for interpretability

Static weight analysis (looking at Q, K, V matrices) gives limited
insight compared to dynamic analysis (running inputs through the model
and observing activations). The weights tell us what the heads are
BIASED to look for; the activations tell us what they ACTUALLY find.

For this model, dynamic analysis (the diary's main approach) is more
informative than static analysis. The weights are too small and too
indirect to read directly — the computation happens in the interaction
between weights and processed inputs.
