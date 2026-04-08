# Diary Entry 017: How Layer 5 Recognizes "through"
## The deepest word recognition in the model

### The problem

After Layer 4, the model at position 7 ('u' in "through") predicts 'c'
at 0.60. This is WRONG. After Layer 5, it predicts 'g' at 1.00. How does
Layer 5 make this enormous correction?

### Layer 5 attention reads the FIRST character

At position 7 ('u'), all four heads attend most to 'h' — but NOT the 'h'
at position 0 ("he"). They attend to the 'h' at position 4, which is
the second character of "through":

| Head | Top attention |
|------|--------------|
| H0 | 'h'=0.44, 'u'=0.30 |
| H1 | 'h'=0.36, ' '=0.16, 'r'=0.15 |
| H2 | 'h'=0.42, 'u'=0.19 |
| H3 | 'h'=0.36, 'u'=0.20, 't'=0.16 |

Wait — I need to be more precise about WHICH 'h'. There are two: position
0 ('h' in "he") and position 4 ('h' in "through"). The attention weights
shown include both. Layer 5 is reading a mix of the 'h' from "he" and the
'h' from "through", plus 'u' (self), 'r', 't', and the space.

This is different from Layer 4, which had extremely focused attention
(0.80-0.98 on single positions). Layer 5's attention is more distributed,
gathering information from multiple characters of "through" simultaneously.

### The MLP is the hero again

The critical numbers:

| Component | cos(g) | dot(g) | cos(c) | dot(c) |
|-----------|--------|--------|--------|--------|
| Residual in | +0.058 | — | +0.238 | — |
| Attn output | -0.044 | -0.088 | -0.124 | -0.224 |
| MLP output | +0.552 | +2.284 | -0.022 | -0.085 |

The residual coming INTO Layer 5 already points toward 'c' (cos=+0.238)
and barely toward 'g' (cos=+0.058). This is why Layer 4 predicted 'c'.

The attention slightly suppresses both 'c' (-0.224) and 'g' (-0.088).
It doesn't help with the prediction directly.

The MLP output has cosine +0.552 with 'g' — by far the strongest
alignment I've seen any component have with a target character. It adds
+2.284 to the 'g' logit. And it slightly suppresses 'c' (-0.085).

So the MLP alone transforms the prediction from 'c' to 'g'. The
attention gathered the word's characters; the MLP recognized "throu"
and wrote the 'g' signal with extreme confidence.

### The pattern holds: attention gathers, MLP predicts

This is now confirmed across three examples:
1. Layer 4 on "said": attention reads 's','a'; MLP writes 'd'
2. Layer 5 on "through": attention reads characters; MLP writes 'g'
3. Layer 5 on "strange": attention reads 's'; MLP writes 'e' (0.99)

The division of labor is always the same: attention MOVES information
between positions, MLP PROCESSES it into predictions. The attention
heads don't know what character should come next — they just gather
the evidence. The MLP neurons collectively recognize the word pattern
and push the prediction in the right direction.

### Why "through" needs Layer 5 while "said" only needs Layer 4

"said" is recognizable from a 3-character prefix: "sai" → only "said"
is possible. Layer 4's attention reads 2-3 characters back, which is
enough.

"through" needs a 5-character prefix: "throu" → only "through". But
Layer 4's attention pattern doesn't read far enough back — it only
looks at 2-3 characters in the current word. Layer 5 reads further
back (attending to 'h', 'r', 't' spread across the word) and its
MLP has the depth to recognize the longer pattern.

### The cosine of 0.552 is remarkable

In Entry 011, I noted that the final prediction for 'd' after "sai"
had cosine only 0.370 with the 'd' embedding. Here, the MLP output
alone has cosine 0.552 with 'g'. The MLP is writing a vector that is
MORE THAN HALF aligned with the target character embedding. This is
the most confident single-component signal I've found anywhere in the
model. The MLP at Layer 5 is VERY sure this is "through".

### Summary

Layer 5 is the model's deepest word-recognition layer. It handles
long words that earlier layers can't resolve. Its MLP can recognize
5-character prefixes and write predictions with extreme confidence
(cosine 0.552, resulting in P(g)=1.00).

The model effectively has THREE levels of word recognition:
- Layer 3: handles 2-char patterns (bigrams and easy trigrams)
- Layer 4: handles 3-char patterns (distinctive short words like "said")
- Layer 5: handles 4-5+ char patterns (long words like "through")

Each layer extends the recognition window by roughly one character.
This is an emergent property of the stacked architecture — deeper
layers naturally have access to more accumulated context through
the residual stream.
