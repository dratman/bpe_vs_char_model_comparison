# Diary Entry 043: Layer 0 Attention Feeds Layer 5's Distributed Code
## The causal chain from diffuse attention to word recognition

### Removing Layer 0 attention flips 88 neurons at Layer 5

At position 5 ('i' in "said"), removing Layer 0's diffuse attention
causes 88 of 512 Layer 5 MLP neurons to change their on/off state.
That's 17% of neurons — a massive disruption at the far end of the
pipeline, caused by a seemingly insignificant change at the start.

### The transition-zone neurons shift dramatically

Of the 364 transition-zone neurons at Layer 5, removing Layer 0
attention shifts their pre-GELU values by an average of -0.035, with
individual shifts ranging from -2.58 to +3.12. The maximum shifts
are LARGER THAN the neurons' original values (which are between -2
and 0). This means Layer 0's attention contribution can flip neurons
from deeply off to on, or from marginally off to strongly on.

39 transition-zone neurons get pushed from off to on — they weren't
going to fire, but Layer 0's attention enables them to fire.

### The MLP output loses its word prediction

| Condition | cos(d) | MLP norm |
|-----------|--------|----------|
| With L0 attn | +0.393 | 3.08 |
| Without L0 attn | +0.129 | 4.16 |

Without Layer 0 attention, the MLP output's alignment with 'd' drops
from 0.393 to 0.129 — a 3× reduction. Interestingly, the NORM increases
(4.16 vs 3.08), meaning the MLP outputs MORE energy but in the WRONG
direction. Without the right input, the neurons fire more but produce
less useful predictions.

### The complete causal chain

1. **Layer 0 attention** averages nearby character representations,
   creating a mixed context signal at each position.

2. This mixed signal propagates through **Layers 1-4** via the
   residual stream, getting transformed at each step.

3. By the time it reaches **Layer 5**, the context signal has been
   converted into specific biases in the pre-GELU activations of
   hundreds of transition-zone neurons.

4. These biases determine which of the 364 transition-zone neurons
   fire (above zero) or stay silent (below zero).

5. The pattern of 364+ firing decisions collectively determines the
   MLP output direction, which determines the prediction.

6. Removing Layer 0's attention disrupts step 1, which cascades
   through steps 2-5, flipping 88 neurons and destroying the word
   prediction.

### This is how context flows through the model

Layer 0's diffuse attention seems insignificant because it's not
focused on anything specific. But it provides the FOUNDATION of the
context signal. Without it, every subsequent layer receives a slightly
different input, which accumulates through 5 layers of processing into
a massive disruption at the prediction layer.

This is a form of chaos — small changes at the input cascade into
large changes at the output. But it's CONTROLLED chaos: the model
has been trained so that the correct context signal (from Layer 0's
attention) produces the correct pattern of neuron firings at Layer 5.
The system is chaotic but trained to be chaotic in the right way.

### Connection to the sensitivity finding (Entry 025)

In Entry 025, I found that 10% noise at the embedding breaks word
recognition by flipping 46 Layer 4 neurons. Now I see that removing
Layer 0 attention flips 88 Layer 5 neurons — an even larger disruption.
The model's signal chain is precise: each layer depends on the exact
output of the previous layer, and any disruption cascades.
