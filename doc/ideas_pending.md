# Ideas Pending

Collected from diary entries 001-089. Reviewed 2026-04-29.

## High priority — directly relevant to current work

### LARQL and gated-FFN (SwiGLU) model
Diary 082. A full architecture was designed but never trained. Planned
as "the run after the current BPE run." The BPE run is now done. The
gated-FFN uses SwiGLU activation instead of GELU, which is what modern
architectures (Llama, etc.) use. Implementation steps are detailed in
the diary entry.

### Topological analysis on the BPE model
Diary 081. The persistent homology analysis (consolidation →
differentiation → segmentation) was done on the character model. Does
the same pattern appear in the BPE model? The BPE model has a different
internal structure because words are already tokenized. This comparison
was proposed but never done.

### 5-layer model — locate the phase transition
Diary 060. A phase transition was found between 4 and 6 layers: 6+
layers enable global context reading via the final layer, while 3-4
layers keep the final layer local. A 5-layer model would pinpoint the
exact transition. Never trained.

### Word frequency vs recognition accuracy
Diary 023. Systematic test of which words the model can recognize,
correlated with corpus frequency. Called "most promising" but never
done. Would reveal capacity boundaries of the model.

### Whole-word tokenizer
Diary 089. Tokenize using the 100K most frequent whole words plus
the alphabet. Covers 97% of text with 4.9x compression. Token
boundaries always at word boundaries, unlike BPE. Designed but not
implemented.

## Medium priority — open questions worth investigating

### Why lower layers freeze first
Diary 080. During training, lower layers stabilize their weights
before upper layers. Two candidate explanations discussed, neither
satisfying. Ralph noted: "We are chasing something that is not so
simple." Open question.

### Why Layer 0 attention never specializes
Diary 023. Layer 0 attention stays diffuse through all training while
every other layer develops specialized heads. Noted but unexplained.
Diary 066 showed that frozen random Layer 0 attention works almost as
well as trained — the model adapts downstream. But why this is so
remains unknown.

### Dark subspace content
Diary 037, 050, 055. The dark subspace (dimensions invisible to the
output projection) carries inter-layer communication. It encodes
position-within-word (diary 050). But whether it carries other
specific interpretable information or is partly unstructured
computational residue is still unknown.

### Topology during training
Diary 075. When do H1 (word-level), H2 (phrase-level), H3
(passage-level) topological structures first emerge during training?
Infrastructure was built (activation snapshots in train.py) but the
experiment was never run. Would connect the crystallization findings
(diary 080) to the topological analysis (diary 074-075).

### Layer 2's specific contribution
Diary 023, 037, 046. Called a "transition layer" with attention-
dominated processing. Reads previous word's characters. But its
specific contribution to predictions has not been traced in detail.

### GELU transition zone contribution
Diary 024, 037, 042. 62% of neurons are in the GELU transition zone
(weakly active). Diary 042 showed removing 368 weakly-active neurons
breaks word recognition while removing 13 strongly-active ones barely
hurts. But the mechanism by which transition-zone neurons contribute
to predictions is not fully understood.

## Lower priority — future experiments

### Holographic/spectral analysis
Diary 087. Are the layers doing successive spectral decomposition?
The Wiener filter connection to attention is promising but untested.
Measuring the actual spatial frequency content of each layer's
contribution would test this hypothesis.

### Imitator at different split layers
Diary 086. We tried layer 8 and layer 10. Layers 4, 6, 12, 14
would show how imitator performance varies with depth. Earlier
splits give more corrective back-half depth.

### Retokenization of predicted vectors
Diary 086. Cluster the imitator's predicted vectors into discrete
symbols — a learned vocabulary for the model's mid-layer language.
VQ (vector quantization) would be one approach. Not yet attempted.

### Bilingual French+English corpus
Memory file. Add French literature from Gutenberg for bilingual
training. Would test whether the model develops separate internal
representations for the two languages.

### Stop-words-only training
Memory file. Train on text with all content words removed, leaving
only function words and punctuation — a "syntax skeleton." Would
reveal what the model can learn from syntax alone.

### BPE model: does it show lower-layers-freeze-first?
Diary 081. The character model shows layer crystallization order.
Does the BPE model show the same pattern? Never checked.

### Multi-computer git sync
Diary 083. Synchronize all three machines (Mac Studio, M3 Pro,
M2 laptop) via GitHub. Partially done (M3 has a clone). Not
completed for all machines.
