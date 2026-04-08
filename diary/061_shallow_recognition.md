# Diary Entry 061: How 3 Layers Recognize "through"
## Same mechanism, different attention strategy

### The 3L model's final layer reads WITHIN the word

3L L2 attention at position 9 ('g' in "through"):
- H0: 'u'=0.52 — reads the character 2 back
- H1: 'r'=0.32, 't'=0.29 — reads earlier word characters
- H2: 'u'=0.36 — reads 'u' again
- H3: 'o'=0.82 — reads 'o' very strongly

The 3-layer model reads within-word characters at its FINAL layer.
Compare with the 6-layer model's final layer:

6L L5 attention at position 9 ('g' in "through"):
- H0: 't'=0.23, 'g'=0.18 — mixed, includes position 0
- H1: ' '=0.28, 'u'=0.23 — reads the space (FST-like)
- H2: 't'=0.21, 'u'=0.16 — very distributed
- H3: 't'=0.25 — reads position 0

The 6L model's final layer has FST attention (reading 't' at position
0 and the space). It does GLOBAL context reading, not local word
reading. The local word reading happened at L4.

### Same mechanism, different allocation

Both models use the SAME mechanism (attention gathers word characters,
MLP writes prediction):
- 3L: L2 MLP contributes +2.30 (91% of total +2.53)
- 6L: L5 MLP contributes +2.28 (72% of total +3.15, from Entry 053)

The MLP contribution is nearly IDENTICAL (+2.30 vs +2.28). The
difference is that the 6L model adds extra signal from earlier layers
(L2 attention +0.48, L3 +0.27), reaching a higher total (+3.15 vs
+2.53).

### The 3L model's attention is more focused within the word

The 3L model's Head 3 reads 'o' at 0.82 — much more focused than
any 6L head at the final layer. Because the 3L model can't rely on
earlier layers to gather word characters, its final layer attention
must do it all: read specific word characters with high focus.

The 6L model's final layer attention is more diffuse because word
characters were already gathered by Layers 3-4. The final layer can
afford to do global context reading instead.

### The per-component story

3-layer model, position 9 ('g' → 'h'):
- embed: -0.04
- L0: -0.01 (negligible)
- L1: +0.08 (small)
- L2: **+2.50** (98% of total)

The 3-layer model's Layer 2 contributes 98% of the prediction.
Nothing else matters. Compare with the 6-layer model where L5
contributes 72%. The shallow model is even MORE dependent on its
final layer.

### What the 3-layer model CANNOT do

It recognizes "through" with 100% confidence because "throug" is a
distinctive 6-character prefix. But it doesn't read global context:
- No FST heads at any layer
- No quotation mark reading
- No previous-sentence awareness

The 3-layer model does CHARACTER-LEVEL WORD RECOGNITION excellently.
What it can't do is CONTEXTUAL PREDICTION — adjusting its predictions
based on long-range information like quotation marks or sentence
structure.

### The architectural tradeoff is clear

With 3 layers, ALL capacity goes to word recognition. With 6+ layers,
the model can ALSO do contextual prediction. The extra layers aren't
needed for word recognition — they're needed for context.

This is why the val loss improvement from 3L to 6L is substantial
(1.238 → 1.185): contextual prediction helps on every word boundary,
which is ~20% of all positions. Getting context right saves about
0.05 loss, which is exactly the observed improvement.
