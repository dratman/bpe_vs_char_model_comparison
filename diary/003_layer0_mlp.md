# Diary Entry 003: Layer 0 MLP and Complete Layer
## What happens after attention, and what Layer 0 has accomplished

### Attention dramatically changes position 0

The attention output for position 0 ('h') has norm 2.52, more than
double the residual stream (1.16). Ratio = 2.17. For all other positions,
the attention output is roughly the same size as the residual (ratio
0.87-1.13).

Why is position 0 special? Because Heads 2 and 3 both attend heavily to
position 0 from ALL other positions. Those attention weights pull
position 0's value vector into every position's output, but they also
pull every other position's value into position 0's output (since position
0 can only attend to itself, the self-attention is 1.0 for all 4 heads).
Actually wait — position 0 can only attend to itself, so all 4 heads
give it 100% self-attention. The large attention output at position 0
means the V vectors are being projected and summed in a way that
amplifies the signal.

The cosine similarity between the residual and the attention output is
positive but modest (+0.36 for position 0, +0.08 to +0.26 for others).
The attention output is not just reinforcing the residual — it's adding
substantially new information.

### The MLP is extremely sparse

This is the most striking finding so far. Out of 512 neurons per position:
- Position 0 ('h'): only 52 active, only 7 strongly active
- Position 2 (' '): only 19 active, only 1 strongly active
- Position 6 ('d'): only 18 active, only 2 strongly active

The MLP is about 90-96% dark. Only a handful of neurons fire for each
character. This means the MLP is doing very specific, targeted operations
rather than distributed computation.

### Each character has a "signature neuron"

Looking at the top neuron per position:
- 'h' (pos 0): n115 = 1.52 (also fires for 'e')
- 'e' (pos 1): n313 = 2.35
- SPACE (pos 2): n357 = 2.58
- 's' (pos 3): n174 = 2.49
- 'a' (pos 4): n482 = 1.62
- 'i' (pos 5): n393 = 1.74
- 'd' (pos 6): n269 = 1.24

Every character has a different top neuron! And these neurons fire
STRONGLY (1.2 to 2.6) while everything else is near zero. The MLP
in Layer 0 is essentially implementing a character detector bank.

### Neuron 357 is a space detector

n357 fires at 2.578 for SPACE and essentially 0.010 for all letters.
That's a 250x difference. This is a pure space detector. The model
has dedicated one of its 512 neurons in Layer 0 to saying "there is
a space here." This signal then gets projected back to 128 dimensions
and added to the residual stream, where later layers can use it.

### Neuron 266 is a "letter detector" (anti-space)

n266 fires at 0.562 average for letters but -0.168 for space. It's
the inverse of n357. When this neuron is active, it means "we are
inside a word, not at a boundary."

### What Layer 0 accomplishes

After Layer 0, the residual stream has grown from norm ~0.9-1.2 to
~2.1-3.9. Position 0 has grown the most (1.16 → 3.90) because it
receives heavy attention from multiple heads.

The cosine similarity between the input and output of Layer 0 ranges
from 0.57 (position 0) to 0.77 (position 3 's'). So the original
character identity is still present but has been substantially
mixed with new information: contextual signals from attention,
and character-type signals from the MLP (space vs letter, specific
character identity).

Layer 0's job appears to be: **mark what type of character this is**
(using sparse neuron detectors) and **establish positional context**
(using attention to mix information from nearby and first positions).

### Open questions going into Layer 1

1. Does each layer have its own set of character-specific neurons,
   or does it rely on Layer 0's detectors?
2. Do the attention patterns change significantly once the MLP has
   enriched the residual stream with character-type information?
3. At what layer does the model start detecting WORDS rather than
   characters?
