# Diary Entry 004: Layer 1
## Layer 1 is almost entirely a position-0 reader

### All four heads fixate on position 0

This is the most uniform layer I've seen. Every single head, at every
position except position 0 itself, attends most strongly to position 0:

- Head 0: first_attn = 0.62
- Head 1: first_attn = 0.61
- Head 2: first_attn = 0.47
- Head 3: first_attn = 0.66

Head 3 gives 87% of attention to position 0 when queried from position 1
('e'). Even at position 6 ('d'), it still gives 43% to position 0.

In Layer 0, only Heads 2 and 3 showed first-position bias. Now ALL FOUR
heads do it. What changed? Layer 0's attention enriched position 0's
representation. It was already the strongest position embedding, and
now it carries even more information. So Layer 1's Q/K projections have
learned: "whatever is at position 0 is important — read it."

This means Layer 1 is essentially broadcasting position 0's enriched
representation to all other positions. It's not doing local processing.
It's distributing global context.

### The MLP is even sparser — and almost uniform

Layer 1's MLP is remarkably dead. Only 3-5 neurons fire at positions
3-6, compared to 16-52 in Layer 0. Two neurons dominate EVERY position:

- n463: fires at 0.63-0.91 everywhere
- n510: fires at 0.48-0.77 everywhere

These two neurons fire for EVERY character. They're not character-specific
at all. They're adding the same signal everywhere. What signal? Since
they fire at every position after Layer 1's first-position-dominated
attention, I suspect they're encoding something generic like "amount of
context available" or "layer 1 was here."

The difference between positions is tiny. n463 fires slightly more
for early positions (0.91 at SPACE, 0.63 at 's'). n190 fires only
at positions 0-2 (the "he " part). But the differences are small
compared to Layer 0's sharp character-specific activations.

### Layer 1 doesn't care about character identity

Layer 0's job was to mark character types (space detector, letter
detectors). Layer 1's job is completely different: it reads the enriched
position-0 signal and broadcasts it everywhere. The MLP barely
does anything character-specific.

This is a CONTEXT DISTRIBUTION layer, not a FEATURE DETECTION layer.

### Residual stream is growing and stabilizing

The residual norms grew from 2.1-3.9 (after L0) to 3.4-5.6 (after L1).
But the cosine similarity between input and output is 0.77-0.93 — much
higher than Layer 0's 0.57-0.77. Layer 1 is making smaller relative
changes. The residual stream is settling down.

Position 0 ('h') is now norm 5.55, nearly 5x its original 1.16. It has
been amplified by two layers of heavy self-attention (it's the only
position that always attends to itself with 100% attention at Layer 1).

### The emerging picture

- **Layer 0**: Character detection (sparse, character-specific MLP neurons)
  + initial context mixing (attention)
- **Layer 1**: Global context distribution (all heads read position 0)
  + near-uniform MLP (not character-specific)

This makes functional sense. Layer 0 says "what characters are here."
Layer 1 says "let everyone know about the global context." Later layers
can combine character identity (from Layer 0's residual contributions)
with global context (from Layer 1's residual contributions) to make
predictions.

### Questions

1. When does the model start doing LOCAL processing again? Layer 0 had
   some local attention (Heads 0 and 1). Layer 1 has none. Where does
   the self+prev pattern emerge?
2. At what layer does the model start processing WORD identity rather
   than character identity? The word "said" hasn't been recognized yet
   as far as I can tell — each character has its own neuron, but nothing
   binds s-a-i-d together as a unit.
3. Position 0 keeps growing. Does it ever stop, or does it keep getting
   amplified through all 6 layers?
