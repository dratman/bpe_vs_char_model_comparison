# Diary Entry 032: The Dark Subspace — Hidden Channels in the Residual
## Two-thirds of the residual stream is invisible to the prediction

### The decomposition

The 128-dimensional residual stream can be split into:
- **Character subspace** (39 dims): spanned by the 39 character embeddings.
  This is what the final logit computation can "see" — the dot product
  with character embeddings only reads this component.
- **Dark subspace** (89 dims): orthogonal to all character embeddings.
  INVISIBLE to the logit computation. Cannot directly influence the
  predicted character probabilities.

### The dark subspace grows, then shrinks

| Stage | Total norm | Char norm | Dark norm | Char fraction |
|-------|-----------|-----------|-----------|--------------|
| After L0 | 2.15 | 1.49 | 1.55 | 48% |
| After L1 | 3.42 | 2.13 | 2.68 | 39% |
| After L2 | 3.49 | 1.97 | 2.89 | 32% |
| After L3 | 3.84 | 2.22 | 3.14 | 33% |
| After L4 | 3.90 | 2.28 | 3.16 | 34% |
| After L5 | 4.51 | 3.53 | 2.80 | 61% |
| Final LN | 19.4 | 15.8 | 11.3 | 66% |

**Layers 0-2**: The dark subspace GROWS from 48% to 32% char fraction.
The model is writing MORE information to the dark subspace than to the
character subspace. This dark information can't be read by the logit
computation — it can only be read by LATER LAYERS' attention and MLP.

**Layers 3-4**: Stable. Char fraction stays at 33-34%.

**Layer 5**: The char fraction JUMPS from 34% to 61%. Layer 5 massively
increases the character-aligned component. This is the prediction layer
"surfacing" information from the dark subspace into the character
subspace where it can influence logits.

### The dark subspace is the model's inter-layer communication channel

The 89 dark dimensions are a PRIVATE channel between layers. Layers 0-2
write information to these dimensions that later layers can read (via
attention Q/K matching, which operates in the full 128-dim space, not
just the character subspace). But this information never reaches the
final prediction directly.

This means the residual stream serves TWO simultaneous purposes:
1. **Prediction building** (39 dims): gradually assembling the logit-
   visible representation
2. **Inter-layer messaging** (89 dims): carrying computational state
   between layers

The model has 89/128 = 70% more "private" bandwidth than "public"
bandwidth. The bulk of the residual stream is inter-layer communication
that the prediction never sees.

### Layer 5 translates dark→visible

The jump from 34% to 61% char fraction at Layer 5 means Layer 5's
MLP is reading dark-subspace information and writing it into the
character subspace. This is the "translation" step: information that
was accumulated in private channels across layers 0-4 gets converted
into prediction-relevant signals by Layer 5.

This is consistent with Layer 5 having the strongest neuron votes
(+0.40, Entry 022) and the most dramatic effect on predictions. It's
the layer that takes all the hidden computation and expresses it as
a final character prediction.

### The dark subspace carries positional/contextual information

After Layer 5, the dark components across different positions have
cosine similarity 0.32-0.83, with average 0.44. This means the dark
subspace is partially shared across positions (not fully position-
specific). The shared component likely encodes global context: "what
kind of text is this", "are we in dialogue", "what is the general
topic."

### Implications for model capacity

With 39 characters and 128 dimensions, the model has 89 "free"
dimensions for inter-layer communication. If we increased the
vocabulary (say to 200 BPE tokens), the character subspace would
grow and the dark subspace would shrink, potentially limiting the
model's ability to carry inter-layer state. This might explain why
character-level models need proportionally more dimensions than
token-level models — they need the dark subspace for computation.

### Summary

The residual stream is not just "building up a prediction." It's a
dual-channel system: one channel (39 dims) builds the prediction that
the output layer can read; the other channel (89 dims) carries
computational state between layers that is invisible to the output.
Layer 5 is the translator that converts dark-channel information into
prediction-channel signals.
