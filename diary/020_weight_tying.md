# Diary Entry 020: Weight Tying and the Frequency-Norm Paradox
## The embedding matrix encodes character frequency in its norms

### The embeddings are nearly orthogonal — as expected

Average absolute cosine similarity between character pairs is 0.108,
close to the theoretical random baseline of 0.088 for 128-dimensional
vectors. The characters are nearly as orthogonal as random vectors.
This is good for the INPUT role: distinct characters have distinct
representations.

Vowels are slightly more similar to each other (0.123) than consonants
(0.063). This makes sense: vowels appear in similar contexts (between
consonants) and predict similar next-characters, so the OUTPUT role
pulls them slightly closer together.

### The stunning finding: rare characters have LARGER embeddings

| Character | Frequency | Norm |
|-----------|----------|------|
| space     | 17.2%    | 0.781 |
| e         | 10.1%    | 0.844 |
| t         | 7.4%     | 0.931 |
| q         | 0.1%     | 1.383 |
| x         | 0.1%     | 1.360 |
| z         | 0.0%     | 1.370 |

Correlation between frequency and norm: **-0.766**

This is a STRONG negative correlation. The rarest characters have the
BIGGEST embeddings. Space (most common) has the smallest norm (0.781).
'q' (rarest) has the largest (1.383).

### Why? Weight tying creates this pressure

Because of weight tying, the logit for character j is:
    logit[j] = dot(final_repr, embedding[j])

If character j's embedding has a larger norm, it takes less alignment
(smaller cosine) to produce a given logit value. The model can "whisper"
in the direction of a large-norm character and still produce a meaningful
prediction.

For common characters (e, t, a), the model doesn't need this boost.
These characters are predicted often, so the model learns to align the
final representation with them directly. Their embeddings can be small
because the cosine will be large when they're the right prediction.

For rare characters (q, x, z), the model rarely predicts them, so it
can't afford to dedicate much of the final representation's direction
to them. Instead, it makes their embeddings larger, so even a small
alignment produces a reasonable logit. The embedding norm compensates
for the model's inability to strongly point toward rare characters.

### Space has the smallest embedding

Space at 0.781 is the smallest embedding norm of all. This seems
counterintuitive — space is the most common character. But space is also
the most PREDICTABLE character: you can usually tell that a space is
coming because the current word is ending. The model doesn't need a
large space embedding because when space is the right prediction, the
final representation is already strongly aligned with it (Entry 011:
cosine of 0.278 for he→space).

### The dual role creates an efficiency

The negative frequency-norm correlation is not a conflict between the
input and output roles — it's a SYNERGY. For the INPUT role, rare
characters with large norms will have a stronger initial signal in the
residual stream, which helps the model detect them even though they
appear infrequently. For the OUTPUT role, the large norms make rare
characters easier to predict without strong alignment.

Both roles benefit from rare characters having large embeddings.

### Vowels vs consonants

Vowel-vowel similarity (0.123) is nearly double consonant-consonant
similarity (0.063). This is the OUTPUT role at work: vowels predict
similar next-characters (all vowels tend to be followed by consonants),
so they're pulled slightly closer together in embedding space. But
the effect is modest — the model prioritizes distinguishability (INPUT
role) over contextual similarity (OUTPUT role).

### Implications for understanding the model

The embedding matrix is not just a lookup table. It encodes:
1. Character identity (through direction — nearly orthogonal)
2. Character frequency (through norm — rare characters are larger)
3. Character class (through slight clustering — vowels slightly closer)

All three signals are packed into a single 128-dimensional vector per
character, serving both input and output roles simultaneously. This is
a remarkably efficient use of the weight-tying constraint.
