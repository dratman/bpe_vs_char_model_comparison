# Diary Entry 013: Watching Heads Develop During Training
## How does the layer structure emerge from uniform initialization?

### Starting point: step 0

All 24 heads are identical: DIF 0.13/0.10/0.13 3.7b. Perfectly uniform
attention, maximum entropy. The model knows nothing.

### Steps 0-800: Almost nothing happens

Through step 800, every head is still labeled DIF with entropy 3.4-3.7b.
The only change: Layers 3-4 start showing slightly elevated self-attention
(0.17-0.20 vs the initial 0.13). The middle layers are the FIRST to
specialize, not the early or late layers.

### Step 1000: The middle layers wake up

Loss has dropped from 3.76 to 2.04. Suddenly:
- L3 H0: SLF 0.32/0.22/0.06 2.3b — self-attention dominant, entropy halved
- L3 H2: SLF 0.26/0.19/0.07 2.9b
- L3 H3: SLF 0.32/0.22/0.06 2.5b
- L4 H0-H2: SLF with 0.21-0.22 self-attention

Layers 0, 1, and 5 are still completely uniform (DIF, 3.7b).
Layers 3-4 have specialized while the boundaries haven't moved.

### Steps 1200-2000: L3-L4 become strongly self-attending

By step 2000:
- L3: Three SLF heads (0.33-0.36 self), one DIF
- L4: Three SLF heads, L4H2 and L4H3 developing PRV tendency
- L2: Starting to show SLF (0.24-0.31 self)
- L0, L1: Still completely DIF at 3.7b entropy

The entropy in L4 drops dramatically — L4H0 goes from 3.7b to 0.8b.
This head has become extremely focused.

### Steps 3000-4000: Layer 1 starts developing FST

By step 3800:
- L1 H0: first=0.17 (up from 0.13)
- L1 H2: FST 0.12/0.10/0.20 — first head to get FST label!
- L5: starting to show FST tendency (first=0.15-0.17)

Layer 1's first-position pattern is emerging, but slowly. It's the
last interior layer to specialize.

### Final state: step 30000

| Layer | H0 | H1 | H2 | H3 |
|-------|-----|-----|-----|-----|
| L0 | DIF 3.6b | DIF 3.7b | DIF 3.7b | DIF 3.7b |
| L1 | FST 3.2b | FST 3.5b | FST 3.3b | FST 3.6b |
| L2 | SLF 2.1b | PRV 1.8b | SLF 2.3b | DIF 3.5b |
| L3 | PRV 1.2b | FST 3.1b | SLF 1.5b | PRV 1.3b |
| L4 | SLF 0.6b | PRV 1.2b | PRV 1.4b | SLF 1.3b |
| L5 | FST 2.8b | FST 2.8b | DIF 2.2b | FST 2.8b |

### The specialization order

1. **Steps 400-1000**: L3-L4 develop self-attention first (middle layers)
2. **Steps 1000-2000**: L2 develops self+prev attention
3. **Steps 2000-4000**: L4 develops prev-attention; L1 starts FST
4. **Steps 4000-10000**: L1 becomes firmly FST; L5 develops FST
5. **Steps 10000-30000**: L3 H3 becomes strongly PRV (0.41)

### Key observations

**Layer 0 NEVER specializes.** It stays DIF at 3.7b entropy throughout
all 30,000 steps. Its attention is permanently diffuse. This makes sense:
Layer 0's job is character detection via the MLP, not attention-based
processing. Its attention heads are essentially unused.

**Layer 1 specializes LATE.** It's the last to get a label (FST emerges
around step 3800). The first-position broadcasting behavior develops
slowly, even though the final model relies on it heavily.

**The middle layers specialize FIRST and MOST.** L3-L4 develop focused
attention (entropy 0.6-1.5b) long before the boundary layers. These are
the word-processing layers, and they find their roles quickly.

**Layer 5 develops FST, matching our circuit trace.** At the final state,
three of L5's four heads are FST. This is the layer that reads back to
earlier positions for long-range context (quotation marks, etc.). This
pattern develops in the latter half of training (after step 10000).

### A different specialization order than the prior model

The Gutenberg-trained model's dynamics differ from what we saw with the
W&P model. The W&P model (trained on 3.1MB in sentence mode) showed
all Layer 1 heads fixating on position 0 immediately. This Gutenberg
model (trained on 7.7GB in continuous mode) develops middle-layer
specialization first, and first-position attention much later.

This might be because continuous mode doesn't privilege position 0 the
way sentence mode does. The model has to discover that position 0 is
useful, rather than having it handed to it by the training format.

### Connection to the logit lens findings

The logit lens (Entry 012) showed:
- Layers 0-2: character identity
- Layer 3: bigram prediction
- Layer 4: word-level override

The training dynamics show that Layers 3-4 are the first to specialize
(steps 400-1000), which means the word-processing pipeline develops
BEFORE the context-distribution pipeline (Layer 1, steps 2000-4000)
and the context-reading pipeline (Layer 5, steps 4000+).

The model learns to process characters locally first, then learns to
distribute and read context. Local before global.
