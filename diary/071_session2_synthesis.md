# Diary Entry 071: Session 2 Synthesis
## What we learned in two days of research

### The collaboration

Ralph (74, retired engineer with ADD/ADHD) and Claude (LLM) studied
transformer internals by having Claude "be the model" — stepping
through forward passes, writing observations in a diary, and
progressively correcting oversimplifications. Ralph provided research
direction, experimental design, and kept Claude honest about the
limits of understanding.

### Timeline

Day 1 (March 18-19):
- Trained dim=128 and dim=256 models on 7.7GB Gutenberg corpus
- 15 diary entries tracing a single forward pass through "he said"
- Discovered the 6-stage pipeline, dark subspace, distributed
  computation in transition-zone neurons
- 5 self-corrections as understanding deepened
- Trained 3L, 4L, 7L models for depth comparison

Day 2 (March 19-21):
- Depth comparison: phase transition between 4 and 6 layers
- 2-head vs 4-head: same pipeline regardless of head count
- Pedagogical minimum: 2,544-parameter Cat in the Hat model
- Animation script and Manim prototype
- Convergence experiments: permutation symmetry explains
  weight non-convergence (Ralph's experimental design)
- Confirmed at scale on W&P dim=128 model
- Functional summary tool extends comprehension to dim=256

### The major findings

**1. The universal pipeline** (Entries 005, 046, 060, 062)
Detect → Broadcast → Gather → Process → Predict. This structure
appears in every model tested: 3L to 7L, 2 to 4 heads, dim=32 to
dim=256. It's determined by the task (character prediction), not
the hyperparameters.

**2. Attention gathers, MLP predicts** (Entries 014, 017, 033)
In every case tested, attention moves information between positions
while MLPs interpret it and write predictions. The attention output
typically has near-zero or negative alignment with the correct
prediction. The MLP output has strong alignment.

**3. The dark subspace** (Entries 032, 033, 055)
When vocab < dim, the residual stream has invisible dimensions used
for inter-layer communication. This is specific to small-vocab models,
not universal. But the FUNCTION (carrying computational intermediates)
likely exists in all transformers through different mechanisms.

**4. Distributed computation** (Entry 042)
Word recognition comes from hundreds of weakly-active transition-zone
neurons, not from a few strongly-active signature neurons. Removing
the strong neurons barely hurts; removing the weak ones breaks the
model.

**5. The suppressor mechanism** (Entries 056, 057, 063)
Non-final layers can suppress the dominant prediction, reducing
confidence. This is generic (not targeted at specific alternatives)
and serves as calibration. It only appears when the model has enough
depth to afford a calibration layer.

**6. Permutation symmetry** (Entries 065, 066, 067, 068)
Weight non-convergence across training runs is entirely explained by
neuron permutation symmetry. When c_fc is frozen (pinning neurons to
roles), c_proj converges to cosine 0.991-0.993 across runs. Confirmed
at both 2.5K params (Cat in Hat) and 1.2M params (W&P).

**7. The phase transition at 5-6 layers** (Entry 060)
Below 5 layers, the final layer does only local word processing.
At 6+ layers, the final layer reads global context (FST attention),
freeing earlier layers for local processing. This is a qualitative
reorganization, not a gradual improvement.

### Self-corrections

1. "Layer 3 does bigrams" → It makes context-dependent predictions
2. "Layer 4 is the word recognition layer" → Layer 5 is primary
3. "Layer 0 attention is useless" → Essential for context mixing
4. "The MLP uses sparse signature neurons" → Distributed transition zone
5. "Attention reads previous characters" → Reads processed representations
6. "Suppression is targeted" → Generic confidence reduction

### Tools developed

- `train_with_attention_log.py` — tracks attention during training
- `compare_models.py` — multi-model diagnostic comparison
- `functional_summary.py` — compresses forward pass to one line per
  layer, extending comprehension to larger models

### Open questions

1. Can the functional summary tool extend to GPT-2 scale (dim=768)?
2. Does the permutation symmetry result have practical applications
   (e.g., model merging, transfer learning)?
3. What would a 5-layer model look like — the exact transition point?
4. Can we build tools that let Claude "be" a model with billions of
   parameters?

### The method

"Being the model" — stepping through computations, writing observations,
catching and correcting oversimplifications — works as a research
methodology. The diary serves as both a log and a forcing function:
writing down what you think you understand reveals when you're
pattern-matching instead of truly understanding. The 6 self-corrections
in this session all came from testing claims that felt right but weren't.
