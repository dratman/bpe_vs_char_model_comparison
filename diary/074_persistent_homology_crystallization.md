# Diary Entry 074: Persistent Homology Reveals Topological Crystallization
## Linguistic hierarchy may map to homological dimension

### The experiment

Applied persistent homology (via ripser) to the activation spaces of
character-level transformer models at each layer. The key question: how
does the topology of the model's internal representations change as
information flows through the layers?

We computed per-position activation vectors, measured pairwise distances
(cosine for activations, Jensen-Shannon divergence for output distributions),
and fed the distance matrices to ripser to find topological features:
- H0: connected components (clusters)
- H1: loops (cyclic relationships)
- H2: voids (enclosed cavities — only computed in the initial overnight run)

### Results: Cat in the Hat (2L1H, dim=32, block_size=64)

First run: 2000 subsampled positions, max_dim=2, output distributions only.
Took ~5 hours. Found 1977 H0 features, 851 H1 loops, 264 H2 voids.
The loops are real (max persistence 0.22 out of max distance 0.83).

Per-layer run: 1000 subsampled positions, max_dim=1, all layers.

| Layer | H1 loops | H1 persistence | Interpretation |
|-------|----------|----------------|----------------|
| embedding | 131 | 2.6 | Simple character structure |
| layer0 | 389 | 5.5 | Context mixing triples complexity |
| layer1 | 743 | 26.7 | Peak complexity — maximum restructuring |
| final | 718 | 23.3 | Layer norm preserves topology |
| output | 386 | 8.8 | Projection compresses — internal world richer than predictions |

### Results: Gutenberg 3L4H (dim=128, block_size=256)

1000 subsampled positions from 3.2M char War and Peace, max_dim=1.

| Layer | H1 loops | H1 persistence | Interpretation |
|-------|----------|----------------|----------------|
| embedding | 142 | 2.5 | Same starting point as tiny model |
| layer0 | 630 | 6.0 | Build |
| layer1 | 1567 | 61.1 | MASSIVE creative explosion |
| layer2 | 662 | 13.5 | Simplify — editorial pass |
| final | 700 | 13.1 | Stable |
| output | 353 | 8.1 | Compress for prediction |

### Key finding: build → peak → simplify → compress

Both models show the same pattern:

1. **Embedding**: low topological complexity (~140 loops). Raw character
   embeddings have simple structure regardless of model size.

2. **Early layers**: complexity builds. Each layer roughly doubles the
   number of loops.

3. **Peak layer**: maximum topological complexity. For 3L4H this is the
   MIDDLE layer (layer 1), not the last. Persistence 61.1 — 10x the
   Cat in the Hat model's peak.

4. **Later layers**: complexity DECREASES. The model reorganizes its
   rich internal structure into a form suitable for prediction. The
   2-layer Cat in the Hat model can't do this — it goes straight from
   peak to output, which may explain its poorer generation quality.

5. **Output**: about half the internal loops survive projection to the
   vocabulary. The model's internal world is topologically richer than
   what it expresses.

### The central hypothesis: homological dimension = linguistic level

Each level of linguistic structure may correspond to a homological
dimension in the activation topology:

| H dim | Linguistic level | What it detects |
|-------|-----------------|-----------------|
| H0 | Characters clustering into groups | Connected components |
| H1 | Characters → words (transition cycles) | Loops |
| H2 | Words → phrases (enclosing surfaces) | Voids |
| H3 | Phrases → clauses | 3-cavities |
| H4+ | Higher discourse structure | Higher cavities |

Prediction: each homological dimension first appears at a specific layer.
The layer at which Hn first appears tells us where linguistic level n
is constructed.

### Why this matters

A character-token model must build EVERY level of linguistic structure
from scratch — characters, words, phrases, clauses, sentences. A BPE
model gets word-pieces for free from the tokenizer. Comparing persistent
homology across both model types would measure how many topological
dimensions the tokenizer eliminates — framing tokenization as a
topological operation.

If homological dimension reliably tracks linguistic hierarchy across
different texts, model sizes, and architectures, this would be a
mathematically rigorous characterization of how language models organize
knowledge. Potentially significant for the study of natural language
itself.

### Connections to other formalisms

- **Cosmology**: the two-point correlation function (galaxy clustering at
  different scales) is analogous to what attention computes — pairwise
  relevance across positions. The "power spectrum" of language.

- **Biology**: each linguistic level has its own kind of content and rules,
  like molecular → cellular → organ → organism. Not just bigger clumps
  of the same thing.

- **Function spaces**: each position's continuation distribution is a point
  on a probability simplex. The topology of this space is the topology
  of the language as the model sees it.

- **Discrete-continuous boundary**: text is discrete (characters), but the
  model immediately maps it into continuous space where topology, geometry,
  and calculus apply. Like analytic number theory placing integers on the
  real line.

### The telescope analogy

block_size = aperture (how much sky). dim = resolution (how many pixels).
block_size/dim should be ~1-4. The model is a telescope with limited
aperture and resolution, observing a universe of linguistic structure.
Persistent homology measures what it has actually resolved.

### Experimental status

| Experiment | Status | Key result |
|------------|--------|------------|
| Cat in Hat block64, output only, max_dim=2 | DONE | 851 loops, 264 voids |
| Cat in Hat block64, all layers, max_dim=1 | DONE | Crystallization: 131→743 loops |
| Gutenberg 3L4H, all layers, max_dim=1 | DONE | Build→peak→simplify: 142→1567→353 |
| Gutenberg 7L4H, all layers, max_dim=1 | DONE | 7-layer arc: build→peak→collapse |
| Single 256-char window, max_dim=3-4 | NEXT | Test higher H dimensions |
| BPE model comparison | FUTURE | Compare topological dimensions |

### Tools

- py/persistent_homology.py — computes persistent homology on model
  activations at any layer. Supports --layers (embedding, layer0..N,
  final, output, all), --metric (jsd, cosine, euclidean), --max_dim,
  --max_positions. Uses PyTorch forward hooks for activation extraction,
  sliding-window chunking for large texts, ripser for topology.

### Output files

**Cat in the Hat block64:**
- doc/persistent_homology_summary.txt (output-only run)
- doc/persistence_diagram.png, persistence_barcode.png,
  distribution_distances.png
- doc/persistent_homology_layer_comparison.txt (per-layer)
- doc/persistence_across_layers.png, persistence_totals_by_layer.png

**Gutenberg 3L4H:**
- doc/gutenberg_3L4H/persistent_homology_layer_comparison.txt
- doc/gutenberg_3L4H/persistence_across_layers.png, etc.

### Results: Gutenberg 7L4H (dim=128, block_size=256, 7 layers)

1000 subsampled positions from 3.2M char War and Peace, max_dim=1.
This is the model that produces coherent English prose.

| Layer | H1 loops | H1 persistence | Role |
|-------|----------|----------------|------|
| embedding | 196 | 2.0 | Raw characters |
| layer0 | 225 | 1.9 | Minimal change |
| layer1 | 423 | 3.3 | First build |
| layer2 | 983 | 11.3 | Accelerating |
| layer3 | 1202 | 28.2 | Building fast |
| layer4 | 1501 | 52.2 | Still building |
| layer5 | 1957 | 66.3 | PEAK — maximum complexity |
| layer6 | 671 | 14.9 | Massive simplification |
| final | 734 | 16.9 | Stable |
| output | 358 | 8.6 | Compress for prediction |

The full arc is now visible across 7 layers:

1. **Layers 0-1**: quiet. Embedding topology barely changes. Low-level
   work that doesn't alter topological structure.

2. **Layers 2-5**: steady accelerating build. Each layer roughly doubles
   complexity. 423 loops → 1957 loops over 4 layers (10x increase).
   This is the model constructing rich internal structure.

3. **Layer 5**: peak. 1957 loops, persistence 66.3. The most topologically
   complex point in the entire network.

4. **Layer 6**: dramatic collapse. 1957 → 671 loops. Two thirds of
   topological complexity destroyed in one layer. This is the editorial
   pass — reorganizing rich understanding into prediction-ready form.

5. **Output**: further compression to 358. The model knows far more than
   it can express through 40-character distributions.

### Cross-model comparison

| Model | Peak layer | Peak H1 loops | Peak H1 persistence |
|-------|-----------|---------------|---------------------|
| Cat in Hat 2L1H | layer1 (last) | 743 | 26.7 |
| Gutenberg 3L4H | layer1 (middle) | 1567 | 61.1 |
| Gutenberg 7L4H | layer5 (penultimate) | 1957 | 66.3 |

All models: peak near but not at the end, final layer(s) simplify.
The 7-layer model uses most depth to build gradually, last layer to prune.
Extra depth gives more time to build and more capacity to simplify,
but does not create proportionally more topology.

All models start from the same place: ~140-196 loops at the embedding
layer with persistence ~2. Raw character embeddings have similar
topological complexity regardless of model size.

### See also

- doc/conceptual_vision_2026_03_22.md — the full conceptual framework
- doc/toward_a_formalism.md — mathematical framework notes
- diary/073_block_size_entropy_substrings.md — entropy and substring analysis
