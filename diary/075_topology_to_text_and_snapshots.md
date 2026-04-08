# Diary Entry 075: Mapping Topology to Text and Activation Snapshots
## Seeing what the model organizes, and preparing to watch it learn

### Topology mapped to characters

Built py/topology_to_text.py — takes ripser's topological features and
maps them back to specific characters and text positions. Run on the
7L4H Gutenberg model processing a 256-character War and Peace window:

"helping into the carriage. "Princesse, au revoir," cried he,
stumbling with his tongue as well as with his feet. The princess,
picking up her dress, was taking her seat in the dark carriage,
her husband was adjusting his saber; Prince Hippolyte, under pret"

Results at layer 5 (the topological peak) confirm the hierarchy:

**H1 loops (4-5 positions):** The most interpretable is positions
12-15 forming a loop through " the" — a single word recognized as
a cyclic unit. The model's activation vectors for space, t, h, e
form a closed path in R^128.

**H2 voids (24-37 positions):** The most persistent void spans 37
positions covering "helping into the carriage" — a multi-word phrase.
The boundary surface is made of 48 triangles connecting these positions.

**H3 cavities (70-138 positions):** The most persistent H3 feature
involves 138 of 255 positions — more than half the text. Connected
by 1334 tetrahedra. This is passage-level structure.

The hierarchy:
- H1: word-level (4-5 characters)
- H2: phrase-level (24-37 characters)
- H3: passage-level (70-138 characters)

This directly supports the central hypothesis from diary 074:
homological dimension corresponds to linguistic hierarchy level.

### Related work: Imperial College London

Found a recent paper: "The Shape of Adversarial Influence: Characterizing
LLM Latent Spaces with Persistent Homology" (Fay, García-Redondo, Wang,
Dubossarsky, Monod — Imperial College London).

They use persistent homology on LLM activations but for a different
question: how adversarial inputs change topology. They find "topological
compression" under attack. Key differences from our work:

1. They compare across inputs (clean vs adversarial); we compare
   across layers at fixed input
2. They use large BPE models; we use small character models where
   every linguistic level is visible
3. They don't go above H1; we found massive H3 structure
4. They don't track layer-by-layer crystallization
5. They don't connect homological dimension to linguistic hierarchy

This validates the method (serious researchers using PH on LLM
activations) while our specific findings appear to be novel.

### Activation snapshots added to train.py

New command-line arguments (all optional, zero overhead when disabled):
- --snapshot_interval N: save activations every N iterations
- --snapshot_positions M: number of subsampled positions (default 1000)
- --snapshot_windows: comma-separated character positions for tracked
  256-char windows
- --snapshot_dir: output directory

Each snapshot saves a .pt file (~5 MB for 7L4H with 1000 positions)
containing activation tensors at all layers for the fixed positions.
Tested on Cat in the Hat: works correctly, saves embedding + all
transformer layers + final layer norm.

### Purpose of snapshots

With snapshots saved every 500-1000 iterations during a week-long
training run on the Mac Studio, we can later analyze (on the laptop)
how topology develops during training:

- When does H1 (word-level) structure first appear?
- When does H2 (phrase-level) emerge?
- Does H3 (passage-level) appear suddenly or gradually?
- Does the build→peak→collapse pattern exist from early training,
  or does it develop?

This would show the temporal dimension of crystallization — not just
what the model knows at the end, but when each level of linguistic
structure was learned.

### Tools

- py/topology_to_text.py — maps topological features to characters
- py/train.py — updated with --snapshot_interval capability
- py/persistent_homology.py — persistent homology with layer support

### Output files

- doc/war_peace_window_topology_mapped.txt — full topology-to-text
  report for 7L4H on 256-char War and Peace window

### Next steps

1. Design and launch a training run on the Mac Studio with snapshots
   enabled (7L4H dim=128, Gutenberg corpus, ~28000 iterations)
2. Analyze snapshot sequence with ripser to track topology over training
3. Run topology_to_text on multiple windows with different linguistic
   content to test consistency
4. Train a BPE model for comparison
