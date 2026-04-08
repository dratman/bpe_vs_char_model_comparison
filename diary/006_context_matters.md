# Diary Entry 006: Context Changes Everything
## How does prior context affect the processing of "said"?

### The experiment

I fed the model "said" in five different contexts:
1. "he said" — bare
2. '"yes," said' — after a quote
3. "she said" — different pronoun
4. "he said to" — with continuation
5. "the man said" — after a noun

### The stunning result: s→a prediction depends on context

In "he said" (bare context), the model gets s→a wrong (0.11, predicts 't').
But in '"yes," said', it gets s→a right with 0.61 confidence!

The full comparison at the critical positions:

| Context        | s→a  | a→i  | i→d  |
|----------------|------|------|------|
| "he said"      | 0.11 | 0.17 | 0.70 |
| '"yes," said'  | 0.61 | 0.91 | 1.00 |
| "she said"     | 0.26 | 0.43 | 0.99 |
| "the man said" | 0.13 | 0.30 | 0.93 |

After a quotation mark and comma, the model is VERY confident that "said"
follows. After "she", it's moderately confident. After "he" or "the man",
it's uncertain.

### What this tells me

The model has learned the statistical structure of English fiction:
- After a quote ends with '," ' the most common next word is "said"
- After "she " the probability of "said" is higher than after "he "
  (perhaps because "she said" is a very common attribution in novels)
- After "the man " it's less clear — could be "said" but could be
  many other verbs

This is NOT just character-level prediction. The model's confidence at
s→a depends on characters that appeared 5-10 positions earlier (the
quotation mark, the comma). That information must travel through the
residual stream across those positions via attention.

### The attention mechanism IS the long-range memory

In Layers 1-2, all heads read position 0. But position 0 in '"yes," said'
is '"' — a quotation mark. The model has learned that a quotation mark
at the start of the context window is a strong signal that dialogue
attribution ("said") is coming. The quotation mark signal gets broadcast
to all positions by Layer 1, and Layers 3-4 can use it to boost "said"
predictions.

### The i→d prediction is always right

Once the model has seen "sai", it ALWAYS predicts "d" — ranging from
0.70 (bare context) to 1.00 (after a quote). By the third character
of "said", the word is effectively certain regardless of context. This
confirms what I found in Entry 005: word recognition happens at the
character level as enough evidence accumulates, but context determines
how early that recognition kicks in.

### Confidence ordering is linguistically sensible

After-quote > after-"she" > after-"the man" > after-"he"

This reflects real English usage patterns in fiction:
- '"...", said' is extremely common
- "she said" is very common (high-frequency attribution)
- "the man said" is less common (most dialogue uses pronouns)
- "he said" is common but "he" precedes many verbs, not just "said"

The model has captured these distributional facts from the Gutenberg
corpus, and they affect predictions at every character position.

### Open question

I described the information flow theoretically (quotation mark at
position 0 → Layer 1 broadcasts → later layers use it). But I haven't
verified this mechanistically. I could test it by feeding in '"yes," '
and tracing position 0's contribution through each layer to see if
the quotation mark signal literally reaches position 7 ('s') and
boosts the "a" prediction. That would be a circuit trace, which is
the next level of understanding.
