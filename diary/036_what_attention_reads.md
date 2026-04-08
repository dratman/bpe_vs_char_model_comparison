# Diary Entry 036: What Layer 4 Attention Actually Reads
## The V vectors contain processed information, not raw characters

### Position 7 ('c' in "church") evolves through layers

| Layer | Top alignments | Logit lens prediction |
|-------|---------------|---------------------|
| L0 | c=+1.54 | c=1.00 (pure identity) |
| L1 | c=+1.63 | c=1.00 (still identity) |
| L2 | c=+1.40, h=+0.45, o=+0.40 | c=0.98 (h, o emerging) |
| L3 | c=+0.90, q=+0.70, o=+0.59 | c=0.61 (identity fading) |
| L4 | h=+0.73, c=+0.58, q=+0.54 | h=0.23 (h overtakes c!) |
| L5 | o=+3.38, h=+3.31 | o=0.39 (now predicting next) |

By Layer 4, position 7 is no longer "c" — it has become a representation
where 'h' is the strongest alignment. The residual stream at this
position has been transformed from "I am the letter c" into "the next
character is probably h" (which is correct: 'ch' in "church").

### Layer 4 attention reads PREDICTIONS, not characters

When Layer 4's Head 0 reads position 7 with 0.83 attention weight
(Entry 034), it is NOT reading "c". It's reading a residual stream
that already encodes "h is coming next." The V vector computed from
this residual carries prediction information, not raw character identity.

This means the attention mechanism is reading PREDICTIONS FROM EARLIER
POSITIONS, not characters. Position 7 has already been processed by
Layers 0-3, which have built up a prediction at that position. Layer 4
reads that prediction and uses it to inform its own processing at
position 9.

### This changes the picture of how word recognition works

My earlier description was: "Layer 4 reads back to previous characters
(s, a) and recognizes the word." The corrected description is:

"Layer 4 reads back to previous positions, which already contain
PROCESSED REPRESENTATIONS that encode predictions about their own
next characters. By reading what position 7 predicts (h), position 9
gets information about the character sequence without directly seeing
character identity."

In other words: **attention reads other positions' predictions, not
their inputs.**

### But position 7 at Layer 3 doesn't know it's in "church"

After Layer 3, position 7 ('c') predicts P(h)=0.003 and P(o)=0.005 —
essentially identical to 'c' in "coming" (same values). Layer 3 has
NOT yet figured out what word this 'c' starts. The 'c' representation
is still generic at this point.

So when Layer 4 reads position 7, it's reading a representation that
is BETWEEN character identity and word prediction — it has been partially
transformed by Layer 3 but doesn't yet carry word-specific information.

### The real information is in the PATTERN of readings

Layer 4 doesn't get word identity from any single position it reads.
It gets word identity from the COMBINATION:
- H0 reads pos 7 → gets 'c'-ish representation
- H1 reads pos 8 → gets 'h'-ish representation
- H2 reads pos 7 → gets 'c'-ish representation
- H3 reads the space → gets word-boundary signal

The COMBINATION of these four readings — c + h + c + space — is what
the MLP uses to narrow down the word. No single reading is sufficient,
but together they provide enough evidence for the MLP to make its
computation.

### This is ensemble reading

The four attention heads are like four people each reading one part
of a message. None of them has the full picture, but the MLP receives
ALL FOUR readings simultaneously (via the 128-dim concatenated output)
and can combine them. The word recognition is not in any single head —
it's in the MLP's ability to interpret the four-headed ensemble reading.

### Implication for the dark subspace

The processed representation at position 7 contains information in
BOTH the character subspace (what character is here, what it predicts)
and the dark subspace (computational state from earlier layers). When
Layer 4 reads this via the V projection, it reads BOTH subspaces.
The dark subspace information — invisible to the logit computation —
is visible to the attention V projection. This is how dark information
flows between positions: through the V vectors of attention heads.
