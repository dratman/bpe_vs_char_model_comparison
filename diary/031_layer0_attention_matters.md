# Diary Entry 031: Layer 0 Attention IS Important Despite Being Diffuse
## Ablating it is almost as bad as ablating the MLP

### The experiment

Remove Layer 0's attention (keep MLP) vs remove Layer 0's MLP (keep
attention). Compare predictions on "he said to himself."

### Both components are important

| Condition | Avg P(true) | Damage |
|-----------|------------|--------|
| Normal | 0.498 | — |
| No L0 attention | 0.109 | -0.389 |
| No L0 MLP | 0.144 | -0.354 |

Removing L0 attention drops performance by 0.389. Removing L0 MLP drops
it by 0.354. The attention is SLIGHTLY MORE important than the MLP!

This contradicts my earlier assumption that Layer 0's attention is
unused because it's diffuse. Diffuse attention is not zero attention —
it still mixes information between positions.

### What diffuse attention actually does

Diffuse attention (each position attending ~equally to all visible
positions) computes an AVERAGE of all previous positions' representations.
This average is then added to the residual stream.

At position 5 ('i' in "said"), diffuse attention over positions 0-5
gives position 5 a mixture of h, e, space, s, a, i. This mixture
doesn't favor any particular character — but it tells position 5
"what characters have appeared so far."

This is a DIFFERENT kind of information than Layer 4's focused attention
(which reads specific word characters). Layer 0's diffuse attention
provides a SUMMARY of the local context, while Layer 4 provides
SPECIFIC lookback.

### Without L0 attention, specific predictions collapse

| Position | Normal | No L0 attention | What broke |
|----------|--------|----------------|------------|
| 5 ('i') | d=0.70 | l=0.04 | Word recognition gone |
| 12 ('i') | m=0.63 | t=0.02 | Word recognition gone |
| 16 ('l') | f=1.00 | a=0.00 | Word completion gone |
| 9 ('o') | SPACE=0.94 | n=0.01 | Word ending gone |

Without L0 attention, the model can't recognize words at all. P(d|sai)
drops from 0.70 to 0.04. P(f|sel) drops from 1.00 to 0.00. The model
becomes unable to do ANY word-level processing.

### Without L0 MLP, character detection fails

| Position | Normal | No L0 MLP | What broke |
|----------|--------|----------|------------|
| 4 ('a') | m=0.17 | l=0.03 | Can't recognize 'sa...' |
| 5 ('i') | d=0.70 | SPACE=0.00 | Word recognition gone |
| 14 ('s') | e=1.00 | e=0.56 | Partial — still works for 'se' |

Without L0 MLP, the character detectors (n357 for space, n174 for 's',
etc.) are gone. Later layers can't tell what characters are at each
position, so word recognition fails.

### Layer 0 has TWO essential functions, both needed

1. **MLP**: Character detection. Creates a sparse signal marking "what
   character is here" via signature neurons. Without this, later layers
   don't know what characters they're processing.

2. **Attention**: Context mixing. Creates a diffuse summary of "what
   characters are nearby" by averaging all previous positions. Without
   this, later layers don't have the mixed representation they need
   for word recognition.

### Why the attention is diffuse but essential

The attention doesn't NEED to be focused because its job is NOT to read
specific positions. Its job is to create a general context signal — a
background "average of everything before me." This average doesn't
have much information on its own, but it provides the SUBSTRATE that
later layers' focused attention builds on.

Think of it as: Layer 0's diffuse attention creates a "tone" or
"atmosphere" in the residual stream. Layer 4's focused attention reads
specific characters against this background. Without the background,
the focused reading doesn't work — like trying to read a word against
a blank wall vs against a page of text.

### Correcting Entry 004

In Entry 004, I described Layer 1's MLP as "nearly dead" (3-48 active
neurons, just n463 and n510 everywhere). I interpreted this as Layer 1
"doing almost nothing character-specific." But now I see that the
attention part of each layer carries more weight than I credited.
Layer 1's ALL-heads-FST attention pattern is its primary contribution,
not its MLP.

The general lesson: I was too focused on MLP neurons (because they're
interpretable) and not enough on attention (because diffuse patterns
seem meaningless). Diffuse attention is NOT meaningless — it's
essential context mixing.
