# Diary Entry 009: Word Boundaries
## What the model knows about which words follow which

### The model uses preceding words to predict next-word-start

After "from " the model strongly predicts 't' (0.42), because "from the"
is an extremely common phrase. After "of " it's similar: 't' at 0.35.
After "and " it also likes 't' (0.17) — "and the", "and then".

But after "he " or "the " the prediction is nearly flat — top prediction
is 's' at only 0.10-0.11. The model doesn't know which word follows
"he" because many words can follow it.

This is linguistically sensible:
- "from" and "of" are prepositions that strongly predict "the" follows
- "he" and "the" are followed by many different words

### The "from the" circuit

"from " → 't' at 0.42 is the strongest word-boundary prediction I've
found. The model has learned that "from" is almost always followed by an
article or demonstrative. Let me rank these:

| Context | P(t) after space | Likely completion |
|---------|------------------|-------------------|
| from    | 0.42             | from the          |
| of      | 0.35             | of the            |
| in      | 0.33             | in the            |
| said    | 0.31             | said the / said to |
| but     | 0.25             | but the           |
| all     | 0.24             | all the           |
| to      | 0.22             | to the            |
| not     | 0.20             | not the / not to  |
| had     | 0.11             | various           |
| and     | 0.17             | and the           |
| he      | 0.06             | many options      |
| the     | 0.06             | many options      |

This is a clear gradient from high predictability (prepositions →
"the") to low predictability (pronouns/articles → anything).

### "she " vs "he " — genuinely different predictions

After "he " the distribution is nearly flat: s=0.10, c=0.09, p=0.07...
After "she " the model predicts: w=0.17, h=0.14, c=0.10, s=0.09.

"she" predicts 'w' (was, would, went) and 'h' (had, has) more strongly
than "he" does. The model has learned that in Gutenberg fiction, "she"
is more often followed by past-tense verbs (was, had) and activity verbs
(went, would), while "he" is followed by a wider variety.

This is WORD-LEVEL knowledge computed from CHARACTER inputs. The model
doesn't have word tokens. It built "he" and "she" representations from
individual characters, and learned different next-word distributions
for each.

### "had " → 'b' (0.19)

This is interesting. After "had ", the most likely next character is 'b'
(0.19). "had been" is the completion — an extremely common past perfect
construction. The model knows this specific two-word phrase.

Other distinctive predictions:
- "was " → 'a' (0.16): "was a" is very common
- "it " → distribution is fairly flat but 't' leads (0.12): "it the"?
  Actually probably "it to" or "it that".

### The hardest prediction: after "he " or "the "

After these contexts, the model's best guess is only 10-11%. The
entropy is high. This is the "space→lower" prediction that scored
only 59% in our probe — much better than the dim=64 model's 18%,
but still the model's weakness.

The model has hit a fundamental limit at this context length. With
only 2-3 characters of prior context ("he "), there are too many
possible next words. Longer context windows would help — if the model
could see "he said to himself, but he " it could predict much more
specifically. But at position 3 in "he said", it has almost no
information about which verb follows.

### This is where attention to position 0 becomes critical

Remember from Entry 004: Layer 1 broadcasts position 0 to everywhere.
If position 0 happens to contain a quotation mark, or a period, or a
specific letter, that information reaches the space position and helps
narrow the prediction. The model's word-boundary prediction is heavily
dependent on what character happens to be at the start of the context
window — which in continuous mode is essentially random.

This suggests the model would benefit from a larger context window
(block_size > 256) so it can see more prior text at word boundaries.
But that's a hyperparameter change, not a mechanism question.
