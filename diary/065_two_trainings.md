# Diary Entry 065: Two Trainings — Completely Different Weights, Same Behavior
## Ralph's prediction confirmed

### The experiment

Train the same 2,544-parameter model twice with different random
seeds. Compare the weights and the predictions.

### The weights are COMPLETELY DIFFERENT

| Component | Cosine similarity (A vs B) |
|-----------|---------------------------|
| L0 Q | -0.094 |
| L0 K | +0.046 |
| L0 V | -0.207 |
| L0 proj | -0.140 |
| L0 MLP_fc | -0.013 |
| L0 MLP_proj | +0.045 |
| L1 Q | -0.147 |
| L1 K | +0.031 |
| L1 V | -0.026 |
| L1 proj | -0.028 |
| L1 MLP_fc | +0.012 |
| L1 MLP_proj | +0.039 |

Every cosine is near zero (range -0.21 to +0.05). The two models
have COMPLETELY UNRELATED weights. They are as different as two random
matrices would be.

This is true for BOTH layers, not just Layer 0. Even Layer 1's
weights are uncorrelated between the two runs.

### But the predictions are SIMILAR

| Position | Run A | Run B |
|----------|-------|-------|
| t→h | 54% | 36% |
| h→e | 54% | 63% |
| e→space | 53% | 55% |
| space→c | 5% | 7% |
| c→a | 44% | 54% |
| a→t | 15% | 15% |

Both models predict the same characters correctly. Both recognize
"cat" (44% and 54% for c→a). Both know "the" (t→h, h→e). Both
struggle with the same position (space→c at 5-7%). The val losses
are similar (2.108 vs 2.075).

### What this means

Ralph's prediction was exactly right: the weights are completely
different between runs, but later layers adapt to whatever Layer 0
happened to do. However, it goes FURTHER than Ralph predicted —
even Layer 1's weights are completely different between runs. The
ENTIRE model is different, not just Layer 0.

Both models found DIFFERENT SOLUTIONS to the same problem. They use
different internal representations, different neuron roles, different
attention patterns — but they arrive at similar predictions because
there are many ways to solve "predict the next character in The Cat
in the Hat."

### This is a fundamental property of neural networks

The same architecture + same data + same training procedure produces
completely different weights but similar behavior. The weights are
NOT the knowledge — the knowledge is in the INPUT-OUTPUT MAPPING,
which is approximately preserved across runs even though the internal
representation is completely different.

This means: everything I found in the diary about specific neuron
roles (n357 is a space detector, n174 detects 's') is specific to
ONE particular training run. A different run would have different
neurons doing the same jobs. The STRUCTURE (character detection
in L0, word recognition in the final layer) is preserved, but the
specific neuron assignments are arbitrary.

### Implication for interpretability

Any finding about a specific neuron ("neuron 357 is a space detector")
is run-specific. But findings about layer ROLES ("Layer 0 does
character detection, the final layer does word recognition") should
be universal across runs. The architecture determines the roles;
the random seed determines which specific neurons fill them.
