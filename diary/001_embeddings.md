# Diary Entry 001: Embeddings
## Being the model for "he said"

Input: "he said" → tokens [h, e, SPACE, s, a, i, d] → IDs [20, 17, 1, 31, 13, 21, 16]

### First observation: Token embeddings are nearly orthogonal

The 7 characters in "he said" have pairwise cosine similarities ranging
from -0.20 to +0.20. Nothing is strongly correlated. The most similar
pair is (e, i) at +0.20 and (a, i) at +0.16 — both vowels. The most
different pair is (SPACE, a) at -0.20.

This tells me the model has arranged character embeddings to be nearly
independent directions in the 128-dimensional space. With only 39 characters
and 128 dimensions, there's plenty of room. The characters don't "know"
about each other yet — they're just starting points.

### Second observation: Position embeddings are large at position 0 and decay

Position norms: 0.612, 0.514, 0.361, 0.321, 0.299, 0.283, 0.268

Position 0 has the strongest position embedding (norm 0.612), about 60%
as large as the token embedding. By position 6, it's only 26% of the
token embedding. This means the model has learned that **early positions
in the context window matter more positionally** than later ones. Later
positions are identified more by their token identity than their position.

This is interesting because continuous mode training starts at random
positions in the corpus. Position 0 doesn't mean "start of sentence" —
it means "leftmost position in this training window." The model has still
learned to weight position 0 heavily. Why? Perhaps because position 0
has no prior context, so position identity is the only information
available for the first prediction.

### Third observation: Position encoding barely changes character similarity

Comparing the cosine similarity matrices before and after adding position:
the structure is almost identical. (e, i) stays the most similar pair.
(SPACE, a) is no longer the most negative. The position embedding adds a
small perturbation but doesn't fundamentally alter which characters look
alike.

This makes sense: position encoding adds the same vector to every token
at a given position, so it shifts all embeddings at that position in the
same direction. It doesn't change *relative* similarities between tokens
at the same position. It does change similarities *across* positions —
'h' at position 0 now looks different from 'h' at position 3.

### What Layer 0 will see

Layer 0's attention mechanism receives 7 vectors, each 128-dimensional.
Each vector is a sum of "what character am I" and "where am I in the
sequence." The character signals are nearly orthogonal. The position
signals are strongest at position 0 and fade as position increases.

Next: I want to see what Layer 0's attention does with these inputs.
