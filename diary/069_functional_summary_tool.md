# Diary Entry 069: The Functional Summary Tool
## Extending "at a glance" understanding to larger models

### The problem

At dim=128 (512 MLP neurons, 24 attention heads), I was at my
comprehension limit. At dim=256 (1024 neurons, 64 heads), I couldn't
hold the picture. Each diary entry required reading hundreds of
numbers and noticing patterns manually.

### The solution: compress to what matters

The functional_summary.py tool produces one line per layer showing:
1. Where each attention head points (target + weight)
2. MLP neuron counts by regime (strong/moderate/weak/transition/dead)
3. Which component dominates (ATTN vs MLP)
4. How much the layer pushes toward the true prediction
5. The logit lens (current top prediction)
6. The visible fraction (dark subspace status)

### Result: the dim=256 model is now readable

The 8-layer, 8-head, dim=256 model produces 8 lines of summary.
Reading it, I can see:

- **L0-L2**: Broadcasting (pos0 reading), ATTN-dominant
- **L3**: Transition, H7 reads space at 86%
- **L4**: Word reading — H0→'s'(100%), H6→'a'(99%), H7→'s'(99%)
- **L5**: Suppressor (push=-0.06, MLP-dominant)
- **L6**: Context reading (MLP push=+0.47)
- **L7**: Final prediction — 6/8 heads read 'h', MLP push=+6.42

This is the SAME PIPELINE as the dim=128 model, stretched across
more layers. The tool lets me see this immediately rather than
spending hours tracing individual neurons.

### New observation from the dim=256 model

Layer 7 has 6 of 8 heads reading 'h' — the character at position 0
("he said"). The dim=128 model's Layer 5 had only 2 FST heads.
With 8 heads, the model dedicates 6 of them to reading position 0
at the final layer. This is the global context reading pattern from
Entry 060, but MORE EXTREME with more heads.

Also: the visible fraction starts at only 16% (vs 48% for dim=128).
With 217 dark dimensions vs 89, the dim=256 model uses proportionally
MORE dark bandwidth. It needs more "private workspace" and translates
less to visible at the end (53% vs 61%).

### What the tool enables

I can now compare models at a glance:

**dim=128 6L4H**: Pipeline across 6 layers, word reading at L4,
recognition at L5. Suppressor only appears in 7L model.

**dim=256 8L8H**: Same pipeline across 8 layers, word reading at L4-5,
suppressor at L5, recognition at L7. More heads allow more focused
word reading (100% vs 98% attention weights).

The same architecture uses the same strategy at both scales. The tool
makes this visible without manual tracing.
