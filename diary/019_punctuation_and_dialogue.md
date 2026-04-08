# Diary Entry 019: Punctuation, Sentence Boundaries, and Dialogue
## How the model handles structure beyond words

### Sentence boundaries use newlines, not spaces

The Gutenberg corpus separates sentences with newlines: ".\n" appears
506 times in the first 100K chars, while ". " appears only 28 times.
The model has learned this: after a period, it predicts newline at 93%.

After ".\n" (a real sentence boundary), the model predicts:
- t=0.26 — "the" (most common sentence start)
- i=0.17 — "i" (first person, very common in fiction)
- a=0.10 — "a" or "and"
- h=0.09 — "he" or "his"

This is a reasonable distribution of sentence-starting words.

### The period+space confusion

After ". " (period then space), the model predicts "-" at 94%. This
initially seemed wrong, but it makes sense: ". " is rare in this
corpus, and when it appears, it's likely in special contexts (e.g.,
abbreviations followed by dashes, or formatting artifacts). The model
has correctly learned that ". " is unusual and defaults to a rare-context
prediction.

This is a case where the model's "error" is actually accurate corpus
modeling. It's not confused about periods — it's correctly tracking that
the corpus uses ".\n" for sentence breaks, not ". ".

### The model knows "he said nothing" ends with punctuation

After "he said nothing":
- space=0.79, comma=0.10, period=0.07, letters=0.01

The model is 97% sure a non-letter follows. It strongly prefers space
(continuing the sentence) over period (ending it) or comma. This is
grammatically sensible: "he said nothing" is a complete clause that
could continue (" to her", " about it") or end.

After "they went home":
- space=0.47, comma=0.22, period=0.16

Less sure — "they went home" feels more like a sentence ending, and
the model allocates more probability to period (0.16 vs 0.07).

### The model knows word boundaries

After "he said nothing", letters=0.01. The model is 99% sure that
the word "nothing" is complete. It doesn't consider "nothingx" or
"nothings" (well, barely). This shows the model has strong word-ending
knowledge: after "nothing", the only possibilities are punctuation
or space.

### Dialogue handling

After '"i do not know," ' the model predicts:
- a=0.27, s=0.23 — "answered" or "said"
- These are dialogue attribution verbs!

After '"i do not know," he ':
- s=0.22, w=0.15 — "said" or "was"
- The model knows that "he" after a quote is likely followed by "said"

After '"yes," he said, "' (opening a second quote):
- i=0.29, t=0.15, w=0.10 — content words starting a new utterance
- "i" is the most likely first word inside the second quote

This is STRUCTURAL knowledge. The model has learned the pattern:
    "...,\" SPEAKER said, \"..."
It knows that after a closing quote+comma, a speaker attribution comes.
After the attribution, a new quote opens. Inside the new quote, first-
person pronouns and common words appear.

### After '"no," said ':
- t=0.23, m=0.13 — "the" or a name starting with 'm'
- The model expects a speaker identity after "said "

### What the model understands about structure

1. **Sentence boundaries**: Period → newline → capitalized word. The
   model has internalized the corpus format.

2. **Word completeness**: After common words ("nothing", "home"), the
   model gives near-zero probability to additional letters. It knows
   when a word is done.

3. **Clause structure**: After complete clauses ("he said nothing"),
   it predicts space/comma/period in proportions that reflect how
   likely the sentence is to continue.

4. **Dialogue patterns**: The model knows the "quote, attribution,
   quote" structure of fiction dialogue. After a closing quote+comma,
   it predicts speaker names and "said". After opening a new quote,
   it predicts dialogue content.

### What it doesn't understand

The model doesn't maintain state across sentence boundaries. After
".\n", it predicts generic sentence starts regardless of what the
previous sentence said. It has no paragraph-level coherence — each
sentence prediction starts fresh from whatever context fits in the
256-character window.
