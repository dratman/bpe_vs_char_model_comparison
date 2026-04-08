# Diary Entry 005: All Six Layers
## The complete picture of "he said" through the model

### The layer-by-layer progression

**Layer 0 — Character detection**
Attention: mixed (self/first/diffuse). MLP: extremely sparse (16-52
active neurons), each character has its own signature neuron. This layer
marks what character is at each position.

**Layer 1 — Global broadcast**
Attention: ALL four heads fixate on position 0 (first_attn 0.47-0.66).
MLP: nearly dead (3-48 active), just two neurons (n463, n510) fire
everywhere. This layer distributes position-0 information to all
positions. It does almost nothing character-specific.

**Layer 2 — Transition layer**
Attention: mostly first-position still, but Head 2 breaks free
(self=0.37, first=0.17) and starts doing local processing. From 'd',
Head 2 attends to 'e' with 0.45! It's looking backward — possibly
reading what happened before the current word.
MLP: moderate activity (11-34 active), some character variation returns
but weak.

**Layer 3 — Word-internal processing begins**
Attention: Head 3 becomes a strong PREV head (prev=0.45). From position
6 ('d'), Head 3 attends 0.65 to 'a' (the character two positions back,
not one — interesting). Head 2 from 'd' attends 0.56 to 's'. The heads
are learning to look at earlier characters within the word "said."
MLP: MUCH more active. Position 5 ('i') has 84 active neurons, 22
strong. Position 6 ('d') has 75 active, 21 strong. The MLP is now
doing real computation at the positions where the word "said" is
nearing completion.

**Layer 4 — Something dramatic happens**
Attention: Head 1 develops extreme prev-attention (0.58 avg). From 'd',
Head 1 attends 0.91 to 'i'. Head 2 from 'd' attends 0.97 to 's'!
Head 0 from 'd' attends 0.57 to 's'. The model is looking back at the
specific characters of "said" with near-certainty.
MLP: 117-125 active neurons at positions 5-6, with 38-39 strong ones.
The MLP has WOKEN UP. Far more computation than any previous layer.

**Layer 5 — Final prediction preparation**
Attention: Returns to first-position (Head 1: first=0.75, Head 2:
first=0.66). After the intense local processing of Layers 3-4, the
model reads global context one more time before predicting.
MLP: Position 2 (' ') has n391 at 7.2! This is the highest single
activation I've seen anywhere. The space position lights up enormously
in the final layer.

### The SPACE EXPLOSION at Layer 5

Position 2 (' ') has residual norm 23.1 after Layer 5 — almost 6x
larger than any other position (4.1-11.1). Before Layer 5 it was 3.7,
so Layer 5's MLP amplified it by 6x.

Neuron n391 fires at 7.2 for the space, while other neurons fire at
3-4. This is the model's strongest single signal: "there was a space
here, and it matters for what comes next."

Why does the space need such a large signal? Because predicting the
first letter after a space is the hardest prediction. The model needs
to switch from "what letter comes next within a word" to "what word
comes next." That requires a completely different computation, and
apparently the model signals this by making the space representation
enormous.

### The predictions

The model gets 4/7 correct:
- h→e (0.49) ✓ — "he" is a common word start
- e→SPACE (0.64) ✓ — "he" is a complete word
- SPACE→s (0.10) ✓ — barely, 10% confidence
- s→a (0.11) ✗ — predicts 't' (0.16), meaning "st..." is more common
  than "sa..."
- a→i (0.17) ✗ — predicts 'm' (0.32), meaning "sam..." is more common
  than "sai..."
- i→d (0.70) ✓ — once it sees "sai", "said" is near-certain
- d→? predicts SPACE (0.71) — correct! After "said" comes a space.

### The key insight: when does the model "know" it's processing "said"?

Not until position 5 ('i'). At position 3 ('s'), the model thinks 'st'
is most likely. At position 4 ('sa'), it thinks 'sam' is most likely.
Only at position 5 ('sai') does it become confident: 'said' is the only
common English word matching this pattern, and the prediction jumps to
70% for 'd'.

This is consistent with the MLP activation pattern: Layers 3-4 show
massive activity at positions 5 and 6, where the model has enough
evidence to recognize "said" as a word. Positions 3 and 4 have less
activity because the model is still uncertain about which word this is.

### The architecture in action

The six layers implement a clear pipeline:
1. **L0**: Mark character types (sparse detector neurons)
2. **L1**: Broadcast position-0 context (all heads → first position)
3. **L2**: Transition — start looking backward at specific characters
4. **L3**: Word-internal processing — attention looks within the word
5. **L4**: Intense local processing — heads attend 0.91-0.97 to nearby
   characters. MLP fires heavily. This is where word recognition happens.
6. **L5**: Final preparation — read global context again, amplify space
   signal, prepare prediction logits

This is not a uniform pipeline where each layer does the same thing.
Each layer has a distinct functional role, and the roles emerged from
training — they were not designed.
