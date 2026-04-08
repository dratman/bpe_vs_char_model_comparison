# Diary Entry 015: Layer 4 Reads Words — A General Mechanism
## Confirmed across 5 different words

### The pattern is consistent

Testing Layer 4 attention on "quick", "through", "nothing", "himself",
"beautiful" — the same reading pattern appears in every case:

**Head H2** consistently reads the FIRST LETTER of the current word:
- "quick": H2 reads 'q' at 95-99% from all positions
- "through": H2 reads 'h' (the second char, after 't') at 75-90%
- "nothing": H2 reads 'n' at 47-87%
- "himself": H2 reads 'h' or 'm' (early word chars) at 61-82%
- "beautiful": H2 reads 'b' at 59-83%

**Head H1** reads the PREVIOUS character (one position back):
- "quick": H1 reads 'u'→'i'→'i' as it advances through the word
- "through": H1 reads 'r'→'o'→'u'→'u' — always one step behind
- "nothing": H1 reads 'o'→'i'→'i'→'i'
- "himself": H1 reads 'm'→'e'→'e'→'l'

**Head H0** reads a MIX of earlier characters:
- Often reads 2-3 positions back in the word
- In "nothing": reads 'h' at 69-72% from later positions
- In "himself": reads 's' at 84-85%

**Head H3** reads the SPACE before the word:
- Consistently gives 20-90% attention to the space character
- This anchors "where does this word start?"

### The division of labor within Layer 4

Each head has a specific role in word recognition:

| Head | Role | What it provides |
|------|------|-----------------|
| H0 | Reads mid-word characters | Context from 2-3 positions back |
| H1 | Reads previous character | Immediate local context (bigram) |
| H2 | Reads word-start character | Word identity anchor |
| H3 | Reads the space | Word boundary marker |

Together, these four heads assemble a representation of the current word
at every position. The MLP then interprets this assembled information to
predict the next character.

### How "strange" gets 99% confidence

In "a strange", position 7 ('g'):
- H0 reads 'g' (61%) and 'r' (20%)
- H1 reads 'a' (97%) — the 'a' in 'strange'
- H2 reads 's' (92%) — first letter
- H3 reads the space (51%)

The model sees: word starts with 's', contains 'a'...'r', currently at
'g'. This matches only "strange", and 'e' follows with 99% confidence.

Compare with "quick" at position 6 ('i'):
- H0 reads 'q' (93%)
- H1 reads 'u' (66%)
- H2 reads 'q' (99%)
- H3 reads space (55%)

It sees: word starts with 'q', has 'u', currently at 'i'. This could be
"quick" (→'c') or "quiet" (→'e'). The model predicts 'e' at 53% and
'c' at 24%. It's not fully sure which word this is yet.

### This is a genuine word-recognition architecture

Layer 4 has independently evolved a reading strategy that mimics how a
human might process a word:
1. Check the first letter (Head H2)
2. Check the previous letter (Head H1)
3. Check earlier letters (Head H0)
4. Know where the word started (Head H3)

This emerged from training on raw character sequences. Nobody told the
model to organize its attention heads this way. But it's the most
efficient way to use 4 attention heads for character-level word
recognition: each head specializes in reading a different part of the
word, and together they provide enough information for the MLP to
determine which word this is.

### The limitation

This reading strategy works well for common words where the character
sequence is distinctive. But for words that share prefixes (like
"quick" vs "quiet"), the model remains uncertain until a disambiguating
character appears. The model processes words CHARACTER BY CHARACTER,
accumulating evidence. It can never "see" the whole word at once — it
can only look backward at characters already processed.
