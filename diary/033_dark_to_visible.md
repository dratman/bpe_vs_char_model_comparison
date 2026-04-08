# Diary Entry 033: Layer 5 MLP Translates Dark → Visible
## Confirmed: the MLP reads dark information and writes predictions

### The numbers

Layer 5 MLP at position 5 ('i' in "said"):

| Component | Char norm | Dark norm | Char fraction |
|-----------|-----------|-----------|--------------|
| MLP input | 2.16 | 2.96 | 35% |
| MLP output | 2.68 | 1.52 | 76% |

The MLP receives input that is 35% character-visible and 65% dark.
It produces output that is 76% character-visible and 24% dark.

**Char amplification**: 1.24× (input 2.16 → output 2.68)
**Dark amplification**: 0.51× (input 2.96 → output 1.52)

The MLP DOUBLES the char-to-dark ratio. It reads from the dark subspace
and writes to the character subspace. This is the translation mechanism.

### The prediction signal is in the char subspace

The MLP output's character-component has cosine 0.452 with the 'd'
embedding. The total output has cosine 0.393 with 'd'. The character
component is MORE aligned with 'd' than the full vector — meaning the
dark component slightly dilutes the prediction signal.

This confirms: the prediction signal ('d' is next) lives in the
character subspace, as it must (since logits = dot product with char
embeddings). The dark component of the MLP output is residual
computation that doesn't help the prediction at this position.

### Layer 5 attention doesn't translate

The attention output has char fraction 37% — almost the same as the
input (34%). The attention doesn't change the char/dark balance. It
moves information between positions (reading the quotation mark, etc.)
but doesn't translate dark→visible. That's the MLP's job.

### The complete picture of information flow

1. **Layers 0-2**: Write information to both char and dark subspaces.
   The dark subspace grows from 48% to 68% of the residual. The model
   is building up inter-layer computational state.

2. **Layers 3-4**: Stable char/dark balance (~34/66). These layers
   read and write dark information while modestly adjusting character
   predictions.

3. **Layer 5**: MLP translates dark→visible. The char fraction jumps
   from 34% to 61%. The MLP reads accumulated dark-subspace state
   and converts it into character-aligned prediction signals.

4. **Final LayerNorm**: Amplifies the whole vector uniformly (×4.3),
   which amplifies both char and dark by the same factor. Char fraction
   stays at ~66%.

### Why does the model need a dark subspace?

The dark subspace lets the model carry COMPUTATIONAL INTERMEDIATES —
information that is useful for computing the final answer but that
doesn't itself look like a character prediction.

Example: At position 5, the dark subspace might encode "the previous
word characters are s-a-i" as a pattern that doesn't align with any
single character but that Layer 5's MLP neurons can recognize. The
neurons then fire and write the 'd'-prediction into the character
subspace.

Without the dark subspace, every piece of information in the residual
stream would need to look like a partial prediction (aligned with some
character embedding). This would constrain the model to carry only
39 dimensions worth of intermediate state. With the dark subspace,
it has 89 extra dimensions for intermediate computations.

### This is analogous to how CPUs use registers

A CPU has a small number of output registers (like the char subspace)
that produce the final result, and a larger set of scratch registers
(like the dark subspace) for intermediate computations. The model has
independently evolved a similar architecture: a private workspace for
computation plus a public output channel for predictions.
