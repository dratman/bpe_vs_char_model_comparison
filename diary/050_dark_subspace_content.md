# Diary Entry 050: What the Dark Subspace Carries
## The dark dimensions encode position-within-word

### Character type is NOT encoded in the dark subspace

The dark-subspace centroids for vowels, consonants, and spaces are
HIGHLY similar (cosine 0.67-0.84). The dark subspace does NOT strongly
distinguish character types. This makes sense: character type
information is encoded in the visible (character) subspace, where the
character embeddings live.

### Word position IS encoded in the dark subspace

The centroid norms decrease with word position:
- Word position 0 (first letter): centroid norm = 2.22
- Word position 1: norm = 1.68
- Word position 2: norm = 1.51
- Word position 3: norm = 1.55

And the cosine similarity between word positions DECREASES with
distance:
- pos 0 ↔ pos 1: 0.84 (adjacent, similar)
- pos 0 ↔ pos 3: 0.68 (distant, less similar)

The dark subspace encodes "how far into the current word am I" as a
gradually changing signal. Position 0 (word start) has the strongest
dark-subspace signal (norm 2.22), which fades as we go deeper into
the word.

### Why word position belongs in the dark subspace

Word position is NOT a character prediction — it's a structural feature
that helps the model organize its processing. The model needs to know
"am I at the start of a word or in the middle?" to decide which
attention pattern to use (previous word vs current word). But this
information doesn't directly predict which character comes next, so
it lives in the dark subspace where it can be read by attention Q/K
matching but doesn't pollute the logit computation.

### The dark subspace is a STRUCTURAL SCAFFOLD

It encodes the structural features the model needs for internal
processing:
- Position within word (confirmed here)
- Possibly: distance from sentence boundary, dialogue state, etc.

These features are invisible to the final prediction but essential for
organizing the intermediate computation. They're like the margin notes
in a book: not part of the text, but essential for the typesetter.

### Connection to the position embedding finding (Entry 021)

The position EMBEDDING encodes absolute position in the context window
(with the three regimes: strong pos 0, flat middle, rising end). The
dark SUBSPACE encodes relative position within the current word. These
are complementary: one says "where am I in the window" and the other
says "where am I in the word."

The absolute position information lives in the embedding (128 dims).
The word-relative position lives in the dark subspace (89 dims).
Together they give the model a two-level positional awareness.

### Caveat

The cosine similarities between word-position centroids (0.68-0.84)
are fairly high, meaning the dark subspace representations at different
word positions share a lot of structure. The word-position signal is
a GRADIENT, not a categorical distinction. The model knows "deeper in
the word" vs "at the start" but doesn't have sharp word-position markers.

This is appropriate for a character-level model: word position changes
gradually with each character, and the model's internal representation
changes gradually too.
