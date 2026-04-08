# Diary Entry 007: Tracing the Quotation Mark Circuit
## How does '"' at position 0 cause 's' at position 7 to predict 'a'?

### The experiment

Two inputs, same length, same position for "said":
- QUOTE: `"yes," said` — P(a after s) = 0.613
- BARE:  `aaaa   said` — P(a after s) = 0.128

### Layer-by-layer comparison at position 7 ('s')

**Layer 0: No difference yet**
In QUOTE, 's' attends to nearby characters including the earlier 's'
in "yes" and the quotation marks. In BARE, 's' attends fairly uniformly.
But at this layer, the model is mostly doing character detection — the
context hasn't been processed yet.

**Layer 1: The quotation mark signal reaches 's'**
QUOTE: All four heads read position 0 ('"'): H0=0.39, H1=0.34, H2=0.21, H3=0.28.
BARE: All four heads read position 0 ('a'): H0=0.31, H1=0.36, H2=0.24, H3=0.32.

The attention pattern is similar! Both contexts heavily attend to position
0 in Layer 1. But the VALUE at position 0 is different: '"' vs 'a'. So
the same attention weights pull DIFFERENT information into position 7.
The quotation mark signal enters via the value vectors, not via changed
attention weights.

This is important: the attention pattern is largely content-independent
at Layer 1 (always read position 0). The content matters because different
tokens at position 0 produce different value vectors.

**Layer 2: The model starts looking at structure**
QUOTE: Head 1 strongly attends to p6(' ') at 0.45 and p4(',') at 0.17.
It's looking at the comma and the space before "said".
BARE: Head 1 also attends to p6(' ') at 0.45 and p5(' ') at 0.20.
Similar attention to the space, but there's no comma to attend to.

**Layers 3-4: Local processing takes over**
In both contexts, the attention becomes very self-focused:
- L3 H3: self-attention 0.96 (QUOTE) and 0.98 (BARE)
- L4 H1: self-attention 0.97 (QUOTE) and 0.96 (BARE)

These layers attend to 's' itself and the immediately preceding space.
The patterns are nearly identical between QUOTE and BARE. The difference
is NOT in the attention — it's in what the residual stream ALREADY
CONTAINS by this point.

**Layer 5: The critical divergence**
QUOTE: Heads 0, 1, 2 all attend heavily to position 5 ('"'):
  H0: p5('"') = 0.67
  H1: p5('"') = 0.63
  H2: p5('"') = 0.25

BARE: Heads 0, 1, 2 attend to position 0 ('a'):
  H0: p0('a') = 0.37
  H1: p0('a') = 0.46
  H2: p0('a') = 0.46

This is the smoking gun. In the QUOTE context, Layer 5's attention
explicitly reads the CLOSING quotation mark at position 5. The model
has learned: "if there's a closing quote nearby, the next word is
likely 'said'." Layer 5 goes back and reads that quotation mark signal
to boost the 'a' prediction.

In the BARE context, there's no quotation mark to read, so the model
falls back to reading position 0, which contains generic 'a' information.

### The circuit

1. **Encoding**: The quotation marks at positions 0 and 5 are encoded
   with their character embeddings + position embeddings.

2. **Layer 0**: Character detector neurons mark the quotation marks.

3. **Layer 1**: Position 0's enriched representation (including '"')
   is broadcast to all positions. But the closing quote at position 5
   also gets enriched through local attention.

4. **Layers 2-4**: Mostly local processing. The 's' position processes
   itself and nearby characters. The quotation mark information is
   present in the residual stream (from Layer 1's broadcast) but not
   actively being read.

5. **Layer 5**: Three attention heads explicitly reach back to position
   5 (the closing quote) and read its value. This is when the model
   says: "I see a closing quote nearby → boost probability of 'said'."

6. **MLP Layer 5**: The enriched representation (now containing the
   quotation mark signal) is processed by the MLP, which adjusts the
   logits to favor 'a' after 's'.

### Why this architecture works

The model doesn't need to attend to the quotation mark at every layer.
It only needs to do it once — at the final layer, right before making
the prediction. The earlier layers do character recognition and local
processing. The final layer does the long-range lookup.

This is efficient: most of the computation is local (Layers 3-4 with
0.96-0.98 self-attention), and the expensive long-range attention only
happens once (Layer 5).

### Residual norm observation

The BARE context 's' has a larger residual norm (10.6) than the QUOTE
context (7.5). This is counterintuitive — the less informative context
produces a larger vector. I don't fully understand why. It might be
that the quotation mark signal causes cancellation in some dimensions,
or that the BARE model is more "confused" and producing a larger,
less focused representation.
