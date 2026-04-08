# Diary Entry 022: What Each MLP Neuron Votes For
## Reading the c_proj weight matrix to understand neuron function

### The method

Each MLP neuron j, when it fires, writes `activation[j] * c_proj[:, j]`
to the residual stream. Because of weight tying, this contribution's
dot product with each character embedding gives the neuron's "vote" for
that character:

    vote[j→c] = c_proj[:, j] · embedding[c]

A positive vote means: "when I fire, I boost character c's logit."
A negative vote means: "when I fire, I suppress character c."

I can compute these votes directly from the weights, without running
any input. This tells me what each neuron IS FOR, structurally.

### Layer 0: Character identity voters

Layer 0's top neurons each vote for a specific letter:
- n174 votes for 's' (+0.184)
- n120 votes for 'l' (+0.177)
- n305 votes for 'v' (+0.172)
- n180 votes for 'c/g' (+0.164/+0.115)
- n182 votes for 'g' (+0.160)

These match what I found in Entry 003: Layer 0's MLP acts as a character
detector bank. Now I can see that these detectors aren't just marking
"this is character X" — they're directly voting for "predict character X."
When n174 fires (because it detected 's'), it pushes the logit for 's'
up by 0.184 × activation.

This makes sense for Layer 0: the best prediction early in the pipeline
is just "the current character will probably repeat or continue" — a
very primitive prediction that later layers will override.

### Layer 5: The prediction layer has MUCH stronger votes

Layer 5's top neurons have votes 2-3× larger than Layer 0's:
- n50 votes for 'b' (+0.402)
- n199 votes for 'g' (+0.400)
- n342 votes for 'o' (+0.397)
- n103 votes for 'y' (+0.385)
- n504 votes for 'n' (+0.372)

And the suppressive neurons are equally strong:
- n350 suppresses 'e' (-0.343)
- n510 suppresses 'y' (-0.327)
- n465 suppresses 'l' (-0.326)

Layer 5 is where the model MAKES its final prediction. The votes
need to be strong because Layer 5 is the last chance to influence
the logits. If n199 fires at activation 2.0, it adds 0.80 to the 'g'
logit — a substantial push.

This is consistent with the "through" analysis (Entry 017): Layer 5's
MLP achieved cosine 0.552 with the 'g' embedding. That high alignment
came from multiple 'g'-voting neurons firing simultaneously.

### Layer 3 and 4: Interesting structural patterns

**Layer 3** has strong votes for RARE characters: 'x' (three different
neurons!), 'z', 'q', 'k'. These are characters that are hard to predict
from bigrams alone. Layer 3 may be where the model handles unusual
character sequences.

**Layer 4** also favors rare characters: 'k', 'q', 'j' in the top
voters. But it also has strong SUPPRESSORS for 'q' (neurons 201 and
64 suppress 'q' at -0.21). This is the calibration mechanism from
Entry 008: some neurons boost rare characters when they're appropriate,
while others suppress them when they're not.

### The suppression patterns

Layer 0: suppresses '(' — parentheses should almost never follow
a letter in normal prose.

Layers 3-4: suppress ')' and '(' — these are very rare and should
almost never be predicted.

Layer 5: suppresses COMMON letters ('e', 'y', 'l', 'n', 'r'). This
is the calibration/confidence adjustment. Layer 5's suppressors prevent
the model from defaulting to the most common characters when the
context says otherwise.

### The voting economy

Early layers (L0): vote for the CURRENT character (primitive echo)
Middle layers (L3-4): vote for RARE characters (handling unusual cases)
Final layer (L5): strong votes for the PREDICTED character + strong
suppression of wrong alternatives

This progression makes sense with the logit lens: early layers contribute
character-identity signals (which are eventually overridden), while later
layers contribute the actual prediction signals that determine the output.

### A neuron-level view of the "said" prediction

In Entry 014, I showed that Layer 4's MLP at position 5 ('i' in "said")
collectively pushes 'd' up (+0.293) and 'n' down (-0.264). Now I can
see HOW: there must be 'd'-voting neurons (like Layer 5's n325, which
votes +0.326 for 'd') and 'n'-suppressing neurons that together create
this push. Each neuron contributes a small vote; the sum of 100+
active neurons determines the prediction.
