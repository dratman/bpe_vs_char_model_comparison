# Diary Entry 067: Progressive Freezing Reveals Permutation Symmetry
## The weight non-convergence mystery, solved

### The question

When we train the same model twice with different random seeds, the
weights are completely different but the predictions are similar (Entry
065). Why don't the weights converge to the same solution?

### Progressive freezing experiments

Ralph designed a series of experiments, progressively freezing more
of the model to isolate where the non-convergence comes from.

**Experiment 1: Freeze L0 attention (different random each run)**
6 runs, each with different frozen random L0 attention.
Result: Model works fine (P(a|c) 0.42-0.58). Layer 1 weights don't
converge (cosines near zero). Later layers adapt to any random L0.

**Experiment 2: Same frozen L0, different L1 init**
Same frozen L0 across all runs, only L1 init varies.
Result: Layer 1 weights still don't converge (cosines near zero).
Many equally good L1 solutions exist for the same L0 input.

**Experiment 3: Freeze ALL of L0 (trained) + embeddings (trained)**
Everything frozen except Layer 1.
Result: With embeddings also frozen, model barely works (P(a|c)=0.03).
Embeddings need to train because of weight tying with output.

**Experiment 4: Freeze L0 (trained), fix embedding init but let train**
Result: Layer 1 still doesn't converge. BUT embeddings converge
(cosine 0.85-0.97). Embeddings have one good solution; Layer 1 has many.

**Experiment 5: Freeze L0 + embeddings + L1 layer norms (all trained)**
Only L1 attention + MLP weights train (768 params).
Result: L1 attention shows PARTIAL convergence (avg cosine +0.355).
L1 MLP still doesn't converge (+0.06).

**Experiment 6: Freeze everything except L1 MLP weights**
Only c_fc.weight and c_proj.weight train (512 params).
Result: Still doesn't converge (cosines +0.07).

**Experiment 7: Freeze c_fc, only c_proj trains**
Only the output projection of the MLP trains (256 params).
Result: **CONVERGENCE.** Cosine average +0.993. All 6 runs produce
essentially identical weights AND identical predictions (P(a|c)=0.437-0.440).

### The explanation: permutation symmetry

When both c_fc (input weights) and c_proj (output weights) are free,
the model can assign any neuron to any role. Neuron 5 in one run might
do the same job as neuron 17 in another run. The function is the same
but the neuron assignments are permuted. This creates a vast family of
equivalent solutions — 32! (32 factorial) permutations of neuron roles,
plus sign flips and continuous symmetries.

When c_fc is FROZEN, each neuron is pinned to a specific role (because
its input weights determine what pattern it detects). Now c_proj must
map THESE specific neurons to the correct output. There's essentially
only one way to do that — hence convergence.

### What this means

The non-convergence of neural network weights is NOT because the loss
landscape has many unrelated minima. It's because the loss landscape has
many EQUIVALENT minima connected by symmetry transformations (primarily
neuron permutations). The function computed is the same; only the
internal labeling of neurons differs.

This is a well-known phenomenon in the neural network theory literature
(sometimes called "permutation symmetry" or "mode connectivity"), but
seeing it directly in a 2,544-parameter model trained on Dr. Seuss
makes it concrete and tangible.

### Summary table

| What trains | Params | Converges? | Cosine |
|-------------|--------|-----------|--------|
| Everything | 2544 | No | ~0.00 |
| L1 only | 872 | No | ~0.00 |
| L1 attn + MLP weights | 768 | Partial (attn) | +0.36 / +0.06 |
| L1 MLP weights only | 512 | No | +0.07 |
| L1 c_proj only | 256 | **YES** | +0.993 |
