# Diary Entry 012: The Logit Lens — Watching Predictions Form
## Layer by layer, how does the model go from character to prediction?

### The technique

At each layer, I read the residual stream and project it through the
final layer norm + lm_head to get logits. This shows what the model
would predict if it stopped processing at that layer. It's called
the "logit lens" in the interpretability literature.

### Position 5: 'i' → 'd' (confident prediction)

| Layer | P(d) | Top prediction | What's happening |
|-------|------|----------------|-----------------|
| L0    | 0.000| 'i' (0.99)     | Just knows it's 'i' |
| L1    | 0.000| 'i' (0.77)     | Still mainly 'i', starting to mix |
| L2    | 0.029| 'i' (0.42)     | 'i' fading, 't' and 'n' emerging |
| L3    | 0.063| 'n' (0.58)     | DRAMATIC SHIFT: 'n' takes over |
| L4    | 0.191| 'd' (0.19)     | 'd' emerges as top prediction |
| L5    | 0.701| 'd' (0.70)     | 'd' becomes dominant |

The prediction doesn't gradually fade from 'i' to 'd'. Instead:
1. Layers 0-1: The model just knows "I am the letter i"
2. Layer 2: It starts to forget its own identity
3. Layer 3: It THINKS 'n' is next (P=0.58!) — as in "in", "sin"
4. Layer 4: It corrects to 'd' — recognizing "sai" → "said"
5. Layer 5: Confidence doubles (0.19 → 0.70)

Layer 3's belief that 'n' follows is WRONG but understandable. The
bigram 'in' is very common. Layer 3 is doing bigram prediction based
on the current character 'i'. Layer 4 overrides this with word-level
knowledge: "the previous characters are 's', 'a', 'i' — this is
'said', so 'd' follows."

This is the EXACT MOMENT where bigram knowledge (Layer 3) is replaced
by word-level knowledge (Layer 4). The transition happens between
Layer 3 and Layer 4.

### Position 2: ' ' → 's' (uncertain prediction)

| Layer | P(s) | Top prediction | What's happening |
|-------|------|----------------|-----------------|
| L0    | 0.002| ' ' (1.00)     | Knows it's a space |
| L1    | 0.036| ' ' (0.90)     | Still thinks "space" |
| L2    | 0.055| ' ' (0.86)     | Space identity fading |
| L3    | 0.061| ' ' (0.64)     | Space still dominant |
| L4    | 0.117| 'o' (0.14)     | SUDDENLY uncertain |
| L5    | 0.105| 's' (0.10)     | Nearly flat distribution |

Layers 0-3 all think the answer is SPACE — as if predicting "another
space follows this space." The model can't forget it's looking at a
space until Layer 4, when the prediction suddenly becomes uncertain.

At Layer 4, the space identity is finally overwritten by the next-word
prediction circuit. The distribution becomes very flat (top prediction
'o' at just 0.14), and by Layer 5 it settles on 's' at 0.10 — barely
above chance.

### Position 3: 's' → 'a' (wrong prediction)

| Layer | P(a) | Top prediction | What's happening |
|-------|------|----------------|-----------------|
| L0    | 0.000| 's' (1.00)     | Knows it's 's' |
| L1    | 0.000| 's' (0.99)     | Still 's' |
| L2    | 0.003| 's' (0.96)     | Still 's' |
| L3    | 0.023| 'u' (0.45)     | Thinks 'su...' — like 'such', 'sure' |
| L4    | 0.116| 's' (0.20)     | Confused — 's' comes back, 't' is #2 |
| L5    | 0.113| 't' (0.16)     | Settles on 't' (wrong, 'a' is true) |

Layer 3 predicts 'u' (0.45)! It thinks "su..." is coming — "such",
"sure", "suddenly". This is a pure bigram prediction: 'su' is more
common than 'sa' in English.

Layer 4 partially corrects but ends up confused. Layer 5 settles on
't' (as in "st...") over 'a' (as in "sa..."), getting it wrong.

### The key insight: layer specialization in prediction

- **Layers 0-2**: Character identity. The model mainly knows "I am
  looking at letter X." The logit lens shows the current character
  as the top prediction (which makes sense given weight tying).

- **Layer 3**: Bigram prediction. The model uses the current character
  to predict common next characters: i→n, s→u, etc. These are often
  wrong in context but statistically reasonable.

- **Layer 4**: Word-level correction. The model uses information from
  attention (which has gathered characters from the current word) to
  override the bigram prediction. i→d (not n), because "sai" is
  recognized.

- **Layer 5**: Confidence calibration and context adjustment. The
  probabilities are sharpened or flattened based on how sure the model
  is. Also handles long-range context (quotation marks, etc.).

### Why this matters

The logit lens shows that the model builds predictions in a specific
order: identity → bigram → word. Each layer adds a different type of
knowledge, and LATER LAYERS CAN OVERRIDE EARLIER ONES. Layer 3's
bigram prediction of 'n' after 'i' gets completely overridden by Layer
4's word-level prediction of 'd'.

This means the residual stream is not just accumulating information —
it's a site of COMPETITION between different prediction strategies.
The final prediction is whichever strategy writes the strongest signal
to the residual stream.
