# Diary Entry 011: How Predictions Actually Get Made
## The final step: residual stream → logits via weight-tied dot products

### The mechanism is beautifully simple

Because lm_head is tied to wte (the same weight matrix), the prediction
logit for character j at position i is just:

    logit[i][j] = dot(final_repr[i], char_embedding[j])

The model predicts whichever character's embedding is most aligned with
the final representation. It's literally asking: "which character does
this representation LOOK LIKE?"

### The cosine similarities are small

At position 2 (the space predicting what follows "he "):
- Top prediction 's': cosine = +0.156
- Top 10 range: +0.086 to +0.156

These cosines are tiny! The final representation is not "pointing at"
any particular character. Instead, the 128-dimensional vector encodes
information in many directions simultaneously, and each character
embedding picks up a slightly different amount of that signal.

But the logits are larger (1.5-2.2) because the final representation
has norm 16.68, far larger than the character embeddings (~1.0). So
even a small cosine similarity, multiplied by a large norm, produces
a meaningful logit difference.

### The anti-predictions are decisive

Punctuation characters ('!', '?', ';', ')', ':') have cosines of
-0.34 to -0.58 and logits of -8 to -9. The model is VERY sure that
punctuation does not follow "he ". This is much more decisive than
the positive predictions (logits 1.5-2.2). The model knows what
CAN'T follow better than what WILL follow.

### Weight tying means the model predicts by similarity

Since the same matrix is used for input and output, a character is
predicted when the final representation RESEMBLES the embedding of
that character. This creates a deep constraint: the model must arrange
its 128-dimensional space so that embeddings serve BOTH as inputs
(distinguishing characters from each other) AND as outputs (being
the targets of prediction).

This dual constraint might explain why character embeddings are nearly
orthogonal (Entry 001). If they were correlated, a prediction for
one character would partially predict another, creating noise.

### "sai" → "d" is NOT a simple embedding match

After "sai" (position 5), the final representation has cosine 0.370
with the 'd' embedding — the highest of any character. But 'd's
embedding only explains 13.7% of the variance in the final
representation. The other 86% encodes other information: context,
position, what came before.

The final vector at position 5 is a 19.4-norm vector in 128-D space.
The 'd' embedding is a 1.0-norm vector. The model achieves the 'd'
prediction by nudging a tiny fraction of its representation in the
'd' direction while simultaneously encoding everything else the model
knows at that position.

This is important: predictions are made by small directional biases
in a high-dimensional space, not by transforming the representation
to literally match the target embedding. The model carries a vast
amount of information through the residual stream, and the prediction
is just one facet of it.

### The prediction difficulty gradient

| Position | True next | Cosine | Logit | Correct? |
|----------|-----------|--------|-------|----------|
| 0 ('h')  | 'e'       | +0.214 | +3.47 | Yes     |
| 1 ('e')  | ' '       | +0.278 | +4.21 | Yes     |
| 2 (' ')  | 's'       | +0.156 | +2.22 | Yes (barely) |
| 3 ('s')  | 'a'       | +0.201 | +3.14 | No (predicts 't') |
| 4 ('a')  | 'i'       | +0.220 | +3.54 | No (predicts 'm') |
| 5 ('i')  | 'd'       | +0.370 | +7.46 | Yes     |

The cosine at position 5 (0.370) is almost double position 2 (0.156).
The logit at position 5 (7.46) is more than triple position 2 (2.22).
When the model is confident, it pushes the final representation much
more strongly toward the target embedding direction.

### Why position 3 ('s') fails

At position 3, the true next is 'a' (cosine +0.201, logit +3.14) but
the model predicts 't' (cosine +0.206, logit +3.18). The difference
is razor-thin: 0.04 logit units. The model nearly gets it right — it
just slightly prefers the 't' direction over the 'a' direction.

This is because 'st' (logit 3.18) and 'sa' (logit 3.14) are both
very common in English, and without more context, the model can't
distinguish them. The internal conflict from Entry 008 (L3H3 as
suppressor) contributes to this — it's pulling probability away from
the dominant prediction, which in this case makes the margin between
't' and 'a' razor-thin.

### Summary of what I now understand

The model predicts by building up a 128-dimensional representation at
each position through 6 layers of attention + MLP, then asking "which
character embedding does this final vector most resemble?" The prediction
is determined by small directional biases (cosines 0.15-0.37), amplified
by the large norm of the final representation (16-20). The model is
much more decisive about what characters CAN'T follow (logits -8 to -9)
than about which specific character WILL follow (logits 1.5-7.5).
