# Diary Entry 051: Long-Range Context Is Weak but Real
## Does text 20-30 characters back affect predictions?

### The experiment

All inputs end with " and he " — the same 8 characters. Only the
preceding 20-25 characters vary. What follows "he " depends on
what came before.

### Context changes the distribution, but modestly

| Context | Top prediction | P(top) |
|---------|---------------|--------|
| (no context) | s=0.10 | 0.10 |
| ...man was tired... | w=0.22 | 0.22 |
| ...could not believe... | w=0.25 | 0.25 |
| ...king looked angry... | w=0.21 | 0.21 |
| ...the gun from... | w=0.21 | 0.21 |
| ...letter said that... | w=0.32 | 0.32 |

### With context, 'w' becomes dominant

Without context, "he " produces a flat distribution (top=0.10).
With ANY context, 'w' jumps to 0.21-0.32. The model has learned
that "and he" followed by long context strongly predicts "was" or
"would" or "went" — all common continuations of "and he" in
narrative fiction.

The most interesting case: "the letter said that... and he " produces
w=0.32 — the strongest prediction. "said that" might trigger a
continuation-of-narrative pattern.

### But the variation BETWEEN contexts is small

The difference between the highest prediction (w=0.32 for "said that")
and the lowest (w=0.21 for multiple contexts) is only 0.11. The
distant context shifts the distribution by about 10%, while the
IMMEDIATE context ("he ") already accounts for a much larger shift
(flat 0.10 → peaked at 0.21).

### What the model CAN'T do

The model doesn't generate context-specific predictions like:
- "the king looked angry... and he" → predict "punished" (king-related)
- "he took the gun from... and he" → predict "fired" (gun-related)

These would require semantic understanding beyond character statistics.
Instead, all contexts produce similar distributions dominated by
common verbs (w for was/would/went, h for had, s for said). The model
knows "and he" is followed by past-tense verbs regardless of the
specific narrative context.

### The 256-character context window limits long-range processing

With block_size=256, the model can theoretically use up to 256
characters of context. But in practice, the attention mechanism
(which reads position 0 and a few specific positions) doesn't carry
information from the entire window. The effective context is probably
much shorter — maybe 50-100 characters for the strongest effects.

### Connection to the attention patterns

In Entry 007, quotation marks 10 characters back changed the "said"
prediction dramatically (P=0.11 → 0.61). That was a STRONG long-range
effect. But quotation marks are structurally distinctive — they
strongly activate Layer 5's attention. General narrative context
20 characters back has a much weaker effect (0.10 probability shift).

The model's long-range capabilities are concentrated on STRUCTURAL
markers (punctuation, quotation marks, sentence boundaries) rather
than semantic content. It reads punctuation from a distance but not
narrative meaning.
