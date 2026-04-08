# Diary Entry 076: Layer-by-Layer Activation and Per-Head Attention Analysis
## Watching the 7L4H model process a War and Peace passage

### What we did

Fed a 256-character passage from War and Peace (lowercased) through the
trained gutenberg_dim128_7L4H model and examined:

1. Activation changes between consecutive positions at each layer
2. Attention vs MLP contributions separately at each layer
3. Per-head attention outputs for all 4 heads at all 7 layers

### The passage

"it is settled," she added in a low voice.
anna pavlovna had already managed to speak to lise about the match
she contemplated between anatole and the little princess'
sister-in-law.
"i rely on you, my dear," said anna pavlovna, also in a low tone.
"write

### Three phases of processing

**Phase 1 — Consolidation (layers 0-2):**
Adjacent characters pushed together. Cosine similarity between
consecutive positions rises from 0.00 (embedding) to 0.74 (layer 2).
Words and short phrases emerge as coherent units. Activation changes
are small — the model is building local agreement.

**Phase 2 — Differentiation (layers 3-5):**
Positions pulled apart by function. Cosine similarity drops back to
0.23. The model breaks proximity-based groups and rebuilds by
syntactic/semantic role. Activation changes are large — maximum
restructuring. This is where persistent homology found peak
topological complexity.

**Phase 3 — Segmentation (layer 6):**
Sharp word boundaries emerge. 63% of activation-change boundaries
fall at spaces/punctuation. The model produces clean word-level
representations. MLP dominates (magnitude 6.51 vs attention 1.32).

### Attention vs MLP division of labor

**Attention finds word boundaries.** At every layer, attention's
contribution aligns more strongly with word edges than MLP's does.
Layer 1 attention: 71% at word edges. Layer 5 attention: 80%.

**MLP processes within words.** MLP boundaries tend to fall at
character-level morphological points — doubled letters, consonant
clusters, suffixes. At every layer, MLP is less word-aligned than
attention.

**Layer 6 MLP** produces the cleanest word segmentation of all,
treating "anna pavlovna" as a single named-entity unit.

### Per-head specialization

**Layer 0:** All heads scan the full context (avg_dist 43-64).
Heads 2-3 already find word boundaries (57-59% at word edges).

**Layer 2 — the word-finding layer:**
- Head 1 (avg_dist=3.0, 82% word edges): dedicated word-boundary
  detector. Looks 3 chars back, finds words precisely.
- Head 3 (avg_dist=20.8, 86% word edges): also word boundaries
  but looks further back. Nearly perfect individual word segmentation.
- Head 2 (avg_dist=3.1, 24% word edges): breaks within words —
  possibly morphological processing.
- Head 0 (avg_dist=68.2, 43%): full-context scanning, not word-level.

**Layer 3:**
- Head 0 (prev=0.46, avg_dist=1.4): the bigram detector — heavily
  attends to the immediately previous position.
- Head 1 (80% word edges): continues word-boundary finding.

**Layer 5:**
- Head 3 (avg_dist=2.9, 82% word edges): clean word segmentation
  at a late layer.
- Head 0 (avg_dist=6.0, 67%): phrase-level boundaries.

**Layer 6:** All heads look far back (avg_dist 16-24). No head
specializes in word boundaries anymore — that information is already
in the residual stream from earlier layers. All heads do long-range
contextual integration for final prediction.

### Key insight: the model predicts via a pipeline

The model doesn't predict the next character directly from characters.
It builds words (layers 0-2), builds sentence context (layers 3-5),
then uses sentence-contextualized word representations to predict
(layer 6). This is the linguistic processing pipeline rediscovered
by gradient descent:

character recognition → word recognition → syntactic analysis → prediction

### Correction: the corpus is lowercase

The Gutenberg model's vocabulary is 39 characters: newline, space,
punctuation, and a-z only. The training pipeline lowercased the corpus.
All analysis should use lowercased text (earlier topology analysis
on mixed-case text was misleading but the topology itself was valid
since the model received lowercased tokens).

### Tools used

All analysis done with direct Python scripts extracting activations
via PyTorch forward hooks — no external tools needed beyond numpy
for the change-magnitude analysis. The activation-change boundary
method is simple and effective: compute ||act[i+1] - act[i]|| at
each position, mark positions where this exceeds the 80th percentile.

### Connection to persistent homology

The persistent homology findings (build→peak→collapse) correspond
to the three phases:
- Consolidation (topology building): H1 features increase as nearby
  positions become similar
- Differentiation (peak complexity): maximum H1/H2/H3 as the model
  builds rich relational structure
- Segmentation (collapse): topology simplifies as word-level
  boundaries sharpen

The topology and the activation-change analysis see the same process
from different angles. The topology measures collective shape; the
activation changes measure local dynamics. They agree.
