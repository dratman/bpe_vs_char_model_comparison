# Diary Entry 029: Position 0 — The Model's Pure Bigram Baseline
## What does the model know with zero context?

### Position 0 = pure character knowledge

At position 0, the model has seen exactly ONE character and must predict
the next. There's no context — the prediction is based entirely on the
single character embedding plus the position-0 embedding. This is the
model's BASELINE knowledge about English character statistics.

### Every character has a sensible bigram prediction

| Input | Top prediction | Comment |
|-------|---------------|---------|
| 'a' | n (0.22) | "an" — most common bigram starting with 'a' |
| 'b' | e (0.36) | "be" — most common 'b' word |
| 'h' | e (0.49) | "he" — strongest single prediction |
| 'q' | u (0.96) | "qu" — near certain! |
| 't' | h (0.33) | "th" — most common bigram in English |
| 'v' | e (0.67) | "ve" — very strong |
| 'y' | SPACE (0.67) | "y" as last letter of a word |
| '.' | \n (0.85) | Period → newline (sentence boundary) |

These are all correct English bigram statistics. The model has learned
the statistical structure of English character pairs entirely from
the character embedding + position-0 embedding + 6 layers of processing
(with no prior context to attend to).

### The 'qu' prediction at position 0

q → u at 0.96 is remarkable. Even with NO prior context, the model
knows that 'u' follows 'q' with near-certainty. This is the same qu
test from our success metrics (Entry 005), but now tested at the most
information-poor position possible. The model's qu knowledge is built
into its WEIGHTS, not computed from context.

### Position matters: pos 0 vs pos 100

| Char | Pos 0 top | Pos 100 top |
|------|-----------|-------------|
| 't' | h (0.33) | SPACE (0.23) |
| 'h' | e (0.49) | SPACE (0.17) |
| 'e' | SPACE (0.34) | SPACE (0.22) |

At position 0, the model makes BOLD predictions: h→e (0.49), t→h (0.33).
At position 100 (after 100 'a' characters), the same letters produce
much flatter distributions with SPACE dominating.

Why? At position 100, the context of 100 'a' characters is nonsensical.
The model is confused by this unnatural input and defaults to safe
predictions (space, comma). At position 0, there's no confusing context
— just a clean character — so the model makes its strongest bigram bets.

### Position 0 as the model's "clean slate"

This explains why Layer 1 broadcasts position 0 to all positions. At
position 0, the model has its purest, most confident character knowledge.
There's no noise from conflicting context, no confusion from unnatural
sequences. The position-0 representation is the model's CLEANEST signal
about what character is there and what typically follows it.

By broadcasting this clean signal to all positions, Layer 1 gives every
position access to the model's strongest bigram knowledge. Other positions
may have richer context, but position 0 has the most reliable base
prediction.

### The 'y' → SPACE (0.67) prediction

This is interesting: the model knows that 'y' at the start of a context
window is most likely the last letter of a word (67% space follows).
Words ending in 'y': they, every, only, away, money, very...
This means the model treats position 0 not as "the start of a word"
but as "a random position in the text" — because that's what continuous
mode training does. Position 0 could be the start of a word, the middle,
or the end. The model's predictions reflect this ambiguity.
