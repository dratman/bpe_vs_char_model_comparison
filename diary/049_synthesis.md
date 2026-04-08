# Diary Entry 049: Synthesis — How This Model Works
## Integrating 48 entries into a coherent picture

### The model in one paragraph

A dim=128, 6-layer, 4-head character-level transformer processes text
by first detecting what character is at each position (Layer 0 MLP),
then broadcasting global context from position 0 (Layer 1 attention),
then gathering the previous word's characters (Layer 2 attention),
then erasing character identity and installing a tentative prediction
(Layer 3 MLP), then gathering the current word's characters (Layer 4
attention), and finally recognizing the word and writing the prediction
(Layer 5 MLP). The prediction is made by computing a dot product between
the processed residual stream and each character's embedding, with rare
characters having larger embeddings to compensate for their low frequency.

### The three deepest insights

**1. The dual-channel residual stream (Entry 032-033)**

The 128-dimensional residual stream has two channels: 39 "visible"
dimensions aligned with character embeddings (readable by the output
projection) and 89 "dark" dimensions (invisible to output, used for
inter-layer communication). Early layers write to the dark channel;
Layer 5 translates dark information into visible predictions. The model
uses 70% of its bandwidth for private computation.

**2. Distributed computation in the GELU transition zone (Entry 042)**

Word recognition is NOT performed by a few strongly-active "signature
neurons." It's performed by ~368 transition-zone neurons per position,
each contributing tiny amounts that collectively determine the
prediction. Removing the 13 strongly-active neurons barely hurts;
removing the 368 transition-zone neurons breaks word recognition.
The model's computation is distributed, not sparse.

**3. Context flows through controlled cascades (Entry 043)**

Layer 0's diffuse attention (which seems insignificant) creates a
context signal that propagates through all layers, ultimately
determining which of Layer 5's 364 transition-zone neurons fire.
Removing Layer 0's attention flips 88 neurons at Layer 5 and destroys
word recognition. The model is a precisely calibrated cascade where
each layer depends on the exact output of the previous layer.

### The architecture's emergent properties

1. **Front-loaded gathering, back-loaded processing**: Layers 1-2
   are attention-dominated (ratio 2-5×), Layers 3-5 are MLP-dominated
   (ratio 0.3-1.2×). The model gathers information first, then
   processes it. (Entry 047)

2. **Selective identity erasure**: Layer 3 erases character identity
   only at positions where word recognition is needed (like 'i' in
   "said"). At other positions, identity persists. (Entry 028)

3. **Word-specific distributed codes**: Different words activate
   near-orthogonal patterns across ~80-130 neurons, with zero overlap
   in top-5 neurons. The model's vocabulary capacity comes from
   distributed coding, not localist coding. (Entry 030, 038)

4. **Frequency encoded in norms**: Rare characters have large embedding
   norms (r=-0.77 correlation), compensating for the model's inability
   to strongly align toward them. (Entry 020)

5. **Three prediction channels**: The final prediction combines a
   default bias (layer norm), directional signal (residual stream),
   and confidence (residual magnitude). (Entry 048)

### What I got wrong and corrected

1. "Layer 0 attention is useless" → It's essential for context mixing
2. "Layer 3 does bigrams" → It does context-dependent predictions
3. "Layer 4 is the word recognition layer" → Layer 5 is primary
4. "The MLP uses sparse signature neurons" → It uses 368 distributed
   transition-zone neurons
5. "Attention reads previous characters" → It reads processed
   representations from previous positions

### What I still don't know

1. Why exactly Layer 0's attention never specializes
2. What specific information the dark subspace carries
3. Whether the Move-Process pattern is optimal or just one solution
4. How the 7-layer model will differ (testing now)
5. Whether these findings apply to larger models (dim=256+)
6. Whether the layer roles are necessary consequences of the
   architecture or contingent outcomes of training
