# Diary 080: Thermodynamics of Training and Layer Crystallization

Date: 2026-03-31

## The question

How do character-level models transition from getting characters right to
getting words right? What governs the order in which different levels of
linguistic structure emerge?

## Empirical finding

Layer stability analysis (py/layer_stability.py) on 12 snapshots of the
7L8H model (iter 5 to 220,000) shows:

- **Lower layers freeze first.** By iter 200k-220k, layers 0-2 have
  relative change ~0.08-0.09, while layers 5-6 are at 0.19-0.27.
- The gap widens over training — lower layers stabilize monotonically
  while higher layers remain active.
- The "final" projection layer is the last to stabilize (0.34).

This means the developmental wave seen in topology (diary 079 — layer 5
topology exploding first) was *exploration*, not crystallization. Layer 5
was the most active early on but the last to settle.

## Theoretical framework: neural thermodynamics

Tegmark et al. (arXiv:2505.10559, May 2025) prove that the equipartition
theorem holds for LLM training:

- Learning rate = temperature
- Thermal loss distributes equally per degree of freedom (equipartition)
- The three laws of thermodynamics apply to training dynamics
- Cosine LR schedule = gradual annealing

## Key analogies explored

### Crystallization from a melt
Different "substances" (character patterns, word patterns, phrase patterns)
crystallize at different temperatures. Lower layers = high melting point
minerals (freeze first). Higher layers = low melting point (freeze last).
When one component crystallizes, it changes the liquid composition for
everything else — just as frozen lower layers change the input to higher layers.

### Chemical synthesis
The order of reactions is constrained by the structure of the target.
Some bonds must form before others. Similarly, character representations
must stabilize before word representations can form. LayerNorm acts
like a protecting group — preventing any one layer from overwhelming others.

### Smelting
Different metals separate at different temperatures from a common melt.

## Dynamic balance mechanisms identified

- **LayerNorm**: ongoing normalizer, keeps all layers' contributions
  viable in the residual stream. The "juggler" keeping balls in the air.
- **Softmax**: the selector within each attention head, choosing what
  to attend to (soft argmax). A budget constraint — total attention = 1.
- **Multi-head structure**: parallel explorers, each finding different patterns.
- **Residual stream**: shared workspace with many "on-ramps."

## Open question: WHY do lower layers freeze first?

Two candidate explanations discussed, neither fully satisfying:

1. **Few valid solutions**: Character-level constraints are tight (q->u
   admits no debate). There's essentially one good solution, so the model
   falls in and stays. Higher layers have many approximately-good solutions,
   so they keep exploring.

2. **Permutation count**: Fewer characters than words, so fewer arrangements
   to search. But this was questioned — characters have many permutations too,
   and higher layers have many things happening simultaneously.

Ralph's observation: "We are chasing something that is not so simple."
The true explanation likely involves the interaction between constraint
tightness, search space size, gradient signal strength, and the dependency
structure of language. This needs more work.

## Tools created

- py/layer_stability.py — measures relative activation change per layer
  between consecutive training snapshots

## Literature confirmation

Multiple groups have independently confirmed that lower layers converge
first (searched 2026-03-31):

- Raghu et al. (2017, NeurIPS): SVCCA analysis shows bottom-up convergence.
- Chen & Yuille (2023, ICLR): Shallower layers converge faster, occupy
  flatter minima. Shallow = low-frequency, deep = high-frequency.
- Saxe et al. (2014): Features learned sequentially by singular value
  magnitude. Each feature shows sigmoidal transition (plateau then rapid drop).
- Achille et al. (2019, ICLR): Critical learning periods — first few epochs
  lock in connectivity permanently. Analogous to biological critical periods.
- Hong & Hong (2025): Phase transitions in small character-level transformers.
  Invisible in loss curves, visible through vocabulary probes. Very close
  to our work.
- Decelle et al. (2024, NeurIPS): Cascade of phase transitions, coarsest
  structure first. Symmetry breaking via Landau theory.
- Pezeshki et al. (2021, NeurIPS): Gradient starvation — easy features
  dominate gradient signal, starving harder features. May explain why
  character patterns lock in while semantic patterns are still starved.
- Olsson et al. (2022, Anthropic): Induction heads form via sharp phase
  transition early in training.

## References

- Liu, Liu, Gore, Tegmark (2025): "Neural Thermodynamic Laws for LLM Training"
  arXiv:2505.10559
- Feng & Tu (2021): Inverse variance-flatness relation in SGD (PNAS)
- Sadrtdinov et al. (2025): Ideal gas law for scale-invariant networks
  arXiv:2511.07308
- Raghu et al. (2017): SVCCA. arXiv:1706.05806
- Chen & Yuille (2023): Layer-wise convergence rate. ICLR 2023
- Saxe, McClelland, Ganguli (2014): Exact solutions, deep linear networks.
  arXiv:1312.6120
- Achille, Rovere, Soatto (2019): Critical learning periods. arXiv:1711.08856
- Hong & Hong (2025): Phase transitions in small transformers. arXiv:2511.12768
- Decelle et al. (2024): Cascade of phase transitions. arXiv:2405.14689
- Pezeshki et al. (2021): Gradient starvation. arXiv:2011.09468
- Olsson et al. (2022): In-context learning and induction heads.
  transformer-circuits.pub
