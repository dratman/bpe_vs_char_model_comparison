# Diary Entry 014: How Layer 4 Recognizes "said"
## The actual mechanism of word-level override

### The question

After Layer 3, position 5 ('i') predicts 'n' (P=0.58). After Layer 4,
it predicts 'd' (P=0.19). How does Layer 4 switch from 'n' to 'd'?

### Layer 4's attention gathers the word's characters

At position 5 ('i'), Layer 4's four heads attend to:

| Head | Target | Weight | What it reads |
|------|--------|--------|---------------|
| H0 | 's' | 0.80 | First letter of "said" |
| H1 | 'a' | 0.73 | Second letter of "said" |
| H2 | 's' | 0.98 | First letter again (extremely focused) |
| H3 | SPACE | 0.62 | The space before "said" |

Three heads read the preceding characters of the current word with
near-certainty (0.73-0.98). This is the model gathering evidence:
"the characters before me are s, a — what word is this?"

Head H2 is extraordinary: 98% of its attention goes to a single position
('s'), making it essentially a pointer. This head has a specific job:
read the first character of the current word.

### But the attention output doesn't directly help

Surprisingly, the attention output at position 5 has NEGATIVE dot product
with both 'd' (-0.076) and 'n' (-0.072). The attention itself doesn't
push toward any prediction — it slightly suppresses everything.

So what's the attention doing? It's not directly writing a prediction.
It's **gathering information** — pulling the representations of 's' and
'a' into position 5's residual stream so the MLP can use them.

### The MLP does the actual prediction switch

The MLP output has:
- dot(mlp_out, d_emb) = +0.293 (pushes 'd' UP)
- dot(mlp_out, n_emb) = -0.264 (pushes 'n' DOWN)

This is the smoking gun. Layer 4's MLP is the component that:
1. Takes the attention-gathered information (s + a + space context)
2. Recognizes the pattern "s-a-i" as the word "said"
3. Pushes the prediction from 'n' toward 'd'

Net Layer 4 effect: d gets +0.217 logit boost, n gets -0.336 suppression.

### The division of labor is clear

**Attention**: gathers evidence. Reads the previous characters of the
word (s, a, space) with near-certainty. Does NOT make predictions.

**MLP**: interprets evidence. Receives the gathered character information
and outputs a direction in embedding space that boosts 'd' and
suppresses 'n'.

This is exactly the attention→MLP pipeline described in the mechanistic
interpretability literature: attention moves information between positions,
MLP processes it.

### The MLP uses many neurons, not just one

117 of 512 neurons are active (much more than Layer 0's 16-52). The top
neurons have activations up to 2.2. But individually, each neuron
contributes only small logit pushes (±0.05 to ±0.11). The prediction
switch is the COLLECTIVE effect of many neurons, each contributing a
small nudge.

Some neurons push 'd' up: n244, n98, n173
Some neurons push 'n' down: n244, n510, n98, n235
Some push both: n244 simultaneously pushes d slightly up (+0.017) AND
n strongly down (-0.094).

This distributed computation is different from Layer 0, where single
neurons acted as sharp character detectors. Layer 4's word recognition
is a distributed pattern match across dozens of neurons.

### The complete circuit for "sai" → "d"

1. **Layer 0**: Marks each character (n357 fires for space, n174 for 's',
   n393 for 'i', etc.)
2. **Layer 1**: Broadcasts position 0 context to all positions
3. **Layer 2-3**: Local processing; Layer 3 predicts 'n' (bigram i→n)
4. **Layer 4 attention**: Three heads read back to 's' (0.80, 0.98) and
   'a' (0.73). One head reads the space before the word (0.62).
5. **Layer 4 MLP**: Receives the combined s+a+i+space representation.
   117 neurons activate. Collectively they push 'd' up (+0.293 logit)
   and 'n' down (-0.264 logit), overriding Layer 3's bigram prediction.
6. **Layer 5**: Adjusts confidence and reads long-range context.

### What this means

The model recognizes words through a specific, traceable mechanism:
attention gathers the word's characters, MLP interprets the pattern.
The word "said" is not stored anywhere as a unit — it's recognized
dynamically by Layer 4's attention reading 's' and 'a' back from
earlier positions, and Layer 4's MLP converting that evidence into
a prediction.

This is a form of computation that emerges from training, not from
architecture. Nothing in the transformer design says "Layer 4 should
read previous word characters." The model discovered this strategy
because it reduces prediction error.
