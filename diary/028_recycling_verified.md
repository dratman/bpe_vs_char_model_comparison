# Diary Entry 028: Layer 3 Recycling Is Selective, Not Universal
## Testing the recycling hypothesis across all positions

### The result: Layer 3 only erases 'i' and '.'

Checking all 19 positions in "he said to himself.":

**Fully erased** (current char drops out of top 3):
- Position 5 ('i' in "said"): i drops, n takes over
- Position 12 ('i' in "himself"): i drops, c/o/u take over
- Position 18 ('.'): . drops, \n takes over

**Partially erased** (drops from rank 0 to rank 1-2):
- Position 3 ('s' in "said"): s drops to rank 1, u overtakes
- Position 7 (' ' before "to"): space drops to rank 2, o overtakes
- Position 8 ('t' in "to"): t drops to rank 1, h overtakes
- Position 10 (' ' before "himself"): space drops to rank 2, t overtakes

**Not erased** (current char remains rank 0):
- Position 0 ('h'): stays rank 0
- Position 1 ('e'): stays rank 0
- Position 6 ('d'): stays rank 0
- Position 9 ('o'): stays rank 0
- Position 13 ('m'): stays rank 0
- Position 14 ('s'): stays rank 0

### Layer 3 erases 'i' SPECIFICALLY

Both instances of 'i' get erased. No other letter gets fully erased.
This is not a universal "erase identity" mechanism — it's specific
to certain characters in certain positions.

Why 'i'? Because 'i' is the character where word recognition is
MOST NEEDED. At position 5, the model needs to recognize "sai" →
"said". At position 12, it needs to recognize "hi" in "himself."
These are positions where the input character identity must give way
to a word-level prediction.

Characters like 'd' (position 6) DON'T get erased because 'd' is
already the END of the word "said" — the next prediction is space,
which is simple and doesn't require word recognition.

### Layer 3 erases '.' for sentence boundary handling

Position 18 ('.'): the period is fully erased and replaced by '\n'
(newline) as the top alignment. This is Layer 3 saying "a period
means the sentence is ending — predict a newline." This matches the
corpus format (Entry 019).

### Spaces get partially redirected

Both space positions (7 and 10) drop from rank 0 to rank 2. Layer 3
begins the hard work of predicting the first letter of the next word.
At position 7 (' ' before "to"), 'o' overtakes (because "to" is
common). At position 10 (' ' before "himself"), 't' overtakes
(because "the" is always a strong bet).

### 't' gets redirected to 'h'

Position 8 ('t' in "to"): Layer 3 boosts 'h' to overtake 't'. This
is the bigram prediction: t→h (as in "the", "that", "this"). But
the actual next character is 'o' (it's "to"). Layer 3's bigram
prediction is wrong here too — 'h' is a more common follower of 't'
than 'o' is.

### The refined recycling theory

Layer 3 doesn't erase character identity everywhere. It erases identity
SELECTIVELY at positions where:
1. Word recognition is needed (mid-word positions like 'i')
2. A structural transition is happening (period → newline)
3. The next character is hard to predict (spaces, word boundaries)

At positions where the prediction is already easy (like 'd' in "said"
where space clearly follows), Layer 3 KEEPS the character identity and
lets later layers handle the simple prediction.

### Connection to why Layer 3 is essential

This explains why skipping Layer 3 breaks the model (Entry 026).
Without Layer 3's selective erasing:
- Position 5 still thinks it's 'i' (+0.49) instead of presenting
  a prediction hypothesis for Layer 4 to refine
- Position 18 still thinks it's '.' instead of signaling "sentence
  boundary → newline"
- Spaces still strongly encode "I am a space" instead of beginning
  the next-word prediction

Layer 3 is the layer that TRANSITIONS the residual stream from
input-encoding to prediction-encoding, and it does so selectively
at exactly the positions where this transition is needed.
