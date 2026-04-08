# Diary Entry 041: How the Model Recovers on "rhythm"
## A rare word recognized despite unusual character patterns

### The logit lens tells a dramatic story

**Position 6 ('y' → 't'):**
- L0-2: y=0.96-1.00 (character identity)
- L3: y drops to 0.27, many alternatives emerge
- L4: e=0.50 (WRONG — it thinks 'ye...' might be coming)
- L5: t=0.50 (CORRECT — "rhy" → "rhythm" recognized at L5)

**Position 7 ('t' → 'h'):**
- L0-3: t=0.44-1.00 (slowly fading identity)
- L4: t=0.44 (still the input character — Layer 4 fails)
- L5: h=0.98 (MASSIVE jump from 0.13 to 0.98!)

**Position 8 ('h' → 'm'):**
- L0-2: h=0.67-1.00 (identity)
- L3: e=0.48 (WRONG — bigram 'he' prediction)
- L4: e=0.48 (still wrong — Layer 4 can't fix it)
- L5: m=0.92 (CORRECT — another massive L5 jump)

### Layer 5 does ALL the work on "rhythm"

At every position, Layer 4 either gets it wrong or barely contributes.
Layer 5 then makes a massive correction:

| Position | P(true) at L4 | P(true) at L5 | L5 improvement |
|----------|-------------|-------------|----------------|
| y→t | 0.086 | 0.505 | +0.42 |
| t→h | 0.127 | 0.984 | +0.86 |
| h→m | 0.015 | 0.921 | +0.91 |

Layer 5 adds +0.42 to +0.91 probability in a single step. This is
even more dramatic than "through" (+0.59 at L5, Entry 017).

### Why Layer 4 fails completely on "rhythm"

"rhythm" has several unusual character transitions: r→h, h→y, y→t.
None of these are common English bigrams. Layer 3's bigram prediction
is confused. Layer 4 tries to recover but doesn't have enough depth
to recognize the full pattern.

Only Layer 5, with access to the full depth of the residual stream
(5 layers of accumulated processing), can recognize the COMPLETE
prefix "rhy" or "rhyt" and make the correct prediction.

### The model knows "rhythm" is a word

Despite the unusual character sequence, the model reaches 0.50, 0.98,
and 0.92 confidence at the final layer. It has seen "rhythm" enough
times in the Gutenberg corpus to learn its character pattern. But it
needs ALL SIX layers to recognize it — no shortcut through early layers.

### This supports the "recognition depth" theory

My earlier finding (Entry 035): Layer 5 is the primary word recognition
layer, handling 8/10 words. "rhythm" takes this further: it's a word
that CANNOT be recognized by any layer except L5. The unusual character
transitions mean there's no partial recognition at L3 or L4 — the
word is invisible until the final layer.

This suggests a hierarchy of word difficulty:
- **Easy words** (would, old): recognizable at L4 from common prefixes
- **Normal words** (said, church): recognizable at L5
- **Hard words** (rhythm): recognizable at L5, but requiring the full
  depth of processing

The 7-layer model might handle hard words BETTER if it adds another
level of processing depth. This is what I predicted in Entry 039.
