# Diary Entry 037: Honest Confidence Assessment
## What I know, what I think I know, and what I don't know

### HIGH CONFIDENCE (verified across multiple inputs and methods)

1. **The model has a 6-stage pipeline**: L0 character detection → L1
   context broadcast → L2 transition → L3 identity erasure → L4
   pre-recognition → L5 word recognition + prediction.

2. **Attention gathers, MLP processes**: In every case tested, attention
   heads move information between positions while MLPs interpret that
   information and write predictions. This is not an interpretation —
   it's directly measurable.

3. **Layer 5 is the primary word recognition layer**: 8/10 words tested
   are first recognized at Layer 5. Layer 4 handles only 2.

4. **The dark subspace exists and follows a U-pattern**: 89 of 128 dims
   are invisible to the logit computation. The dark fraction peaks at
   L2-4 and shrinks at L5 as information is translated into predictions.

5. **Word recognition ramps up with evidence**: Confidence increases
   monotonically as more disambiguating characters are seen. This holds
   for every word tested.

6. **Layer 0 MLP uses sparse character detectors**: One signature neuron
   per character, 90%+ of neurons dark. Verified directly from activations.

7. **Weight tying creates frequency-norm anticorrelation**: r=-0.77
   between character frequency and embedding norm. Measured directly.

### MODERATE CONFIDENCE (consistent pattern but limited testing)

8. **Layer 4 attention heads have specific roles**: H0=earlier chars,
   H1=prev char, H2=word-start, H3=space. Tested on 5 words (Entry 015)
   and 2 words (Entry 034). Pattern holds but sample is small.

9. **Layer 3 selectively erases character identity**: Erases 'i' and '.'
   but not most characters. Tested on one sentence. Need to verify this
   selectivity across more inputs.

10. **Layer 4 is a calibration layer**: Removing it HELPS for "said"
    (Entry 026). But I only tested this on one example. It might hurt
    on other words.

11. **The model has proto-grammar**: Predicts different first letters
    after pronouns vs verbs vs articles (Entry 018). But the differences
    are modest (0.05-0.19 probability shifts), and I only tested with
    averaged predictions, not traced the mechanism.

### LOW CONFIDENCE (plausible but not well-tested)

12. **Layer 0 attention provides essential "context mixing"**: The
    ablation (Entry 031) shows it matters, but I haven't traced HOW
    the diffuse attention helps. I called it "context averaging" but
    haven't verified what information the average carries.

13. **The GELU transition zone matters for computation**: 62% of neurons
    are near zero (Entry 024). I hypothesized this makes the model
    sensitive and fluid. But I haven't shown that the transition-zone
    neurons actually contribute to predictions. They might be noise.

14. **Each word uses separate neuron ensembles in Layer 4**: Tested
    with 7 words (Entry 030), found near-orthogonal patterns. But
    Layer 4 is not the primary recognition layer (Entry 035). I haven't
    checked whether Layer 5 also has word-specific ensembles.

15. **Attention reads predictions from previous positions**: Entry 036
    showed position 7 transforms from 'c' to 'h'-prediction by L4. But
    the logit lens alignment at L3 was still mostly 'c'. The processed
    representation is partially transformed, not fully a prediction.
    My description was too strong.

### WHAT I DON'T KNOW

- Why Layer 0 attention never specializes during training (noted but
  unexplained)
- What Layer 2 specifically contributes (haven't investigated)
- Whether the dark subspace carries specific, interpretable information
  or is just unstructured computational residue
- How the model handles truly novel character sequences it hasn't
  trained on
- Whether these findings would hold for a model with different
  hyperparameters (dim=64, dim=256, different layer count)
- Whether the layer roles are NECESSARY (i.e., the only way to solve
  this task) or CONTINGENT (i.e., one of many possible solutions)

### Corrections made during this investigation

1. Entry 012: "L3 makes bigram predictions" → Entry 016: L3 is context-
   dependent, not pure bigrams
2. Entry 012: "L4 handles 3-char prefixes" → Entry 035: L5 is the
   primary recognition layer, L4 handles only the easiest cases
3. Entry 003: "L0 attention is unused" → Entry 031: L0 attention is
   essential for context mixing
4. Entry 014: "Attention reads previous characters" → Entry 036:
   Attention reads processed representations, not raw characters

These self-corrections are the most valuable part of this process.
Each one reveals a way my initial pattern-matching was too simplistic.
