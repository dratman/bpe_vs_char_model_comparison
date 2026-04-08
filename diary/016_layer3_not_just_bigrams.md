# Diary Entry 016: Layer 3 Is Smarter Than I Thought
## Correcting my earlier oversimplification

### My earlier claim was wrong

In Entry 012, I said "Layer 3 makes bigram predictions." I showed that
after 'i' in "said", Layer 3 predicts 'n' (0.58), and I called this a
bigram prediction (i→n is common in English).

But testing other words reveals Layer 3 does NOT predict 'n' for every 'i':

| Context | Layer 3 top prediction after 'i' |
|---------|----------------------------------|
| "he said" (sai-) | n=0.58 |
| "he did" (di-)   | i=0.82 (still the input char!) |
| "he sin" (si-)   | i=0.68 |
| "he win" (wi-)   | o=0.21 |
| "he big" (bi-)   | i=0.39, c=0.22 |
| "he fit" (fi-)   | i=0.43 |

The predictions are DIFFERENT for each context. In "did", Layer 3 still
thinks the answer is 'i' itself (0.82) — it hasn't started predicting
the next character at all. In "said", it confidently says 'n' (0.58).
In "win", it says 'o' (0.21).

### Layer 3 also reads previous word characters

Layer 3's attention at position 5 ('i' in "said"):
- H2: reads 's' at 0.80 (first letter of word)
- H3: reads 'a' at 0.84 (previous letter)
- H1: reads 's' at 0.45

This is the SAME word-reading pattern I attributed to Layer 4! Layer 3
ALSO reads back to the word's characters. The difference is that Layer 3
has weaker word recognition — it makes "sai"→'n' instead of "sai"→'d'.

### The corrected picture: Layers 3 AND 4 both do word processing

Layer 3 is not a pure bigram layer. It reads the word's characters and
makes a context-dependent prediction. But it's often wrong. Layer 4 does
the same thing with more accuracy and overrides Layer 3.

### Different words get recognized at different layers

**"through" at 'u' position:**
- L3: space=0.22, s=0.22, t=0.20 — very uncertain, hasn't recognized it
- L4: c=0.60 — thinks "ch" is coming? WRONG
- L5: g=1.00 — suddenly perfect

"through" is recognized at Layer 5, not Layer 4! The model needs all six
layers for this word. This makes sense: "throu" requires recognizing a
5-character prefix, which needs more processing depth.

**"did" at 'i' position:**
- L3: i=0.82 — still thinks it's 'i', no prediction yet
- L4: n=0.34 — starts making a (wrong) prediction
- L5: s=0.42 — predicts 's' (WRONG — true is 'd')

The model FAILS on "did"! P(d) is only 0.067 after all layers. It
predicts 's' (0.42) instead. "did" is a common word but the model
can't recognize it. Why?

Because 'di' is ambiguous: did, dig, dim, dip, direction, different,
dinner... There are too many words starting with 'di'. Unlike "sai"
(almost certainly "said") or "throu" (almost certainly "through"),
"di" doesn't narrow the possibilities enough.

**"win" at 'i' position:**
- L3: o=0.21, n=0.19 — hedging between 'o' (window?) and 'n' (win)
- L4: t=0.49 — predicts 't' (winter?)
- L5: l=0.28, n=0.27 — still uncertain

Again uncertain. "wi" could be "win", "with", "will", "winter", "wide"...

### When does word recognition succeed vs fail?

The model succeeds at word recognition when the character prefix is
DISTINCTIVE — when few common words share that prefix:
- "sai" → only "said" → 'd' at 0.70 ✓
- "throu" → only "through" → 'g' at 1.00 ✓
- "stran" (from "strange") → 'g' then 'e' at 0.99 ✓

The model fails when the prefix is AMBIGUOUS:
- "di" → many words → 'd' at 0.07 ✗
- "wi" → many words → 'n' at 0.27 ✗

This is exactly what you'd expect from a character-level model that
processes left-to-right. It can only narrow down the word space based
on characters it has already seen. Words with common prefixes remain
ambiguous until a disambiguating character appears.

### The layer at which recognition happens depends on word length

Short, distinctive words: recognized at Layer 4 (e.g., "said" after 3 chars)
Long, distinctive words: recognized at Layer 5 (e.g., "through" after 5 chars)
Ambiguous prefixes: never fully recognized (e.g., "did" after 2 chars)

This makes sense with the logit lens progression: each successive layer
can integrate more context and recognize longer patterns. Layer 3 handles
2-character patterns, Layer 4 handles 3-character patterns, Layer 5
handles 4-5 character patterns. The deeper the layer, the longer the
prefix it can process.

### Revising the pipeline

My earlier description was:
- L0-2: character identity
- L3: bigram prediction
- L4: word-level override
- L5: calibration

A more accurate description:
- L0: character detection (MLP signature neurons)
- L1: global context distribution (position-0 broadcast)
- L2: begin reading previous characters (self+prev attention emerges)
- L3: short-range word processing (reads 2 chars back, handles common
  bigrams and some trigrams, often tentative/wrong)
- L4: medium-range word processing (reads 3+ chars back with very
  focused attention, handles distinctive 3-char prefixes)
- L5: long-range word processing (reads 4-5 chars back, handles long
  words) + context reading (quotation marks, sentence structure)

Each layer extends the word-reading range by roughly one character.
