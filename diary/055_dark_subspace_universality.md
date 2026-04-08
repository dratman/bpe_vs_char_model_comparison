# Diary Entry 055: The Dark Subspace Is NOT Universal
## Correcting an overgeneralization

### The constraint

The dark subspace exists because vocab (39) < dim (128). The 89 dark
dimensions are a STRUCTURAL CONSEQUENCE of having fewer token types
than embedding dimensions.

### Most real-world models have NO dark subspace

| Model | Vocab | Dim | Dark dims |
|-------|-------|-----|-----------|
| Our char model | 39 | 128 | 89 |
| GPT-2 small | 50,257 | 768 | 0 |
| GPT-2 large | 50,257 | 1,280 | 0 |
| GPT-3 | 50,257 | 12,288 | 11,031 |
| LLaMA 7B | 32,000 | 4,096 | 0 |

Only GPT-3 (dim=12,288 >> vocab=50,257... wait, that's vocab > dim
for GPT-3 too). Actually: GPT-3 175B has dim=12,288 and vocab=50,257,
so vocab > dim and there is no dark subspace.

So the dark subspace is essentially a feature of character-level models
and small-vocabulary models. Real-world BPE/token models almost always
have vocab > dim.

### But the FUNCTION might be universal

Even without a guaranteed dark subspace, large models might learn to
use certain directions in the residual stream primarily for inter-layer
communication rather than output prediction. The output projection
(lm_head) could learn to ignore certain directions, creating a DE FACTO
dark subspace even when an architectural one doesn't exist.

In models WITHOUT weight tying (where lm_head is a separate matrix
from the embedding), the lm_head can freely choose to ignore certain
dimensions. This would create an effective dark subspace of whatever
size the model finds useful.

In models WITH weight tying (like ours and GPT-2), the dark subspace
is forced by the vocab/dim ratio. The model gets it for free.

### What this means for our research

The dark subspace mechanism is REAL and IMPORTANT for our model, but
we should NOT claim it as a universal transformer property. It's a
consequence of our specific architecture (char-level, vocab << dim,
weight tying).

The more general claim — that transformers use parts of the residual
stream for inter-layer communication rather than direct prediction —
might still be universal. The dark subspace is one specific mechanism
for achieving this; large models might use different mechanisms.

### What IS likely universal from our findings

Separating our findings into "depends on dark subspace" and "doesn't":

**Depends on dark subspace (char-model specific):**
- 89/128 dims invisible to output
- Word-position encoding in dark dims
- Layer 5 dark-to-visible translation
- The U-shaped char fraction across layers

**Doesn't depend on dark subspace (potentially universal):**
- Final layer dominates predictions
- Attention gathers, MLP processes
- Front-loaded gathering, back-loaded processing
- Distributed computation in GELU transition zone
- Word recognition ramps up with character evidence
- Internal prediction competition (boosters vs suppressors)
- Layer 0 character/token detection
