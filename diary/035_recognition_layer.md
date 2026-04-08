# Diary Entry 035: Which Layer Recognizes Which Word?
## Systematic test across 10 words

### Results

Testing 10 words at the position 2 chars before word end (where
recognition should be happening):

| Word | Recognized at | L4 P(true) | L5 P(true) | Word length |
|------|-------------|-----------|-----------|-------------|
| would | L4 | 0.70 | 1.00 | 5 |
| old | L4 | 0.55 | 0.90 | 3 |
| said | L5 | 0.19 | 0.70 | 4 |
| through | L5 | 0.41 | 1.00 | 7 |
| church | L5 | 0.44 | 0.55 | 6 |
| beautiful | L5 | 0.05 | 0.99 | 9 |
| was | L5 | 0.14 | 0.97 | 3 |
| him | L5 | 0.11 | 0.57 | 3 |
| had | L5 | 0.06 | 0.99 | 3 |
| went | L5 | 0.12 | 0.89 | 4 |

### Layer 4 recognizes only 2 of 10 words

Only "would" (0.70) and "old" (0.55) are recognized at Layer 4 with
>50% confidence. All others need Layer 5.

### My earlier claim was wrong again

In Entry 012, I said "Layer 4 handles 3-character patterns like 'said'."
But the systematic test shows Layer 4 only reaches 0.19 for "said" —
not >0.50. Layer 5 is where "said" actually gets recognized (0.70).

In Entry 016, I revised to say "Layer 4 handles distinctive 3-char
prefixes." But even that's too generous. Layer 4 handles "would" and
"old" but not "said", "was", "him", or "had" — all of which are
3-4 character words.

### What makes "would" and "old" special?

**"would"**: At the 'l' position, the prefix is 'woul'. This is
extremely distinctive — almost no other common word starts with 'woul'.
Layer 4 can recognize it confidently.

**"old"**: At the 'd' position, the prefix is 'ol'. Fewer words start
with 'ol' than with 'sa' or 'wa'. And the measurement is at the last
character, where the model only needs to predict space.

**"said"**: At the 'i' position, the prefix is 'sai'. While distinctive,
Layer 4 only gets to 0.19. The SAME word reaches 0.70 at Layer 5.

### Layer 5 is the primary word recognition layer

8 of 10 words are first recognized at Layer 5. Layer 4 handles only
the easiest cases. This corrects my earlier overestimate of Layer 4's
importance.

The corrected pipeline:
- **Layers 0-2**: Character identity + context mixing
- **Layer 3**: Erases character identity, installs tentative prediction
- **Layer 4**: Partial word recognition (only very distinctive prefixes)
- **Layer 5**: Primary word recognition (handles most words)

### Layer 5 is remarkably capable

- "through" at 'g': L5 jumps from 0.41 to 1.00 (adds +0.59)
- "beautiful" at 'u': L5 jumps from 0.05 to 0.99 (adds +0.94!)
- "had" at 'd': L5 jumps from 0.06 to 0.99 (adds +0.93)

Layer 5 can take a near-zero prediction and push it to near-certainty
in a single step. It's doing the heavy lifting for word recognition.

### Why did I overestimate Layer 4?

Because my original test case ("said") happened to be partially
recognized at Layer 4 (0.19 — above zero but below the threshold).
I interpreted "Layer 4 pushes d above n" as "Layer 4 recognizes said."
But 0.19 is not recognition — it's a hint. Real recognition (>0.50)
happens at Layer 5.

### The model is deeper than I thought

The effective computation is more serial than I described:
- Layers 0-3: Setup (character detection, context mixing, identity erasure)
- Layer 4: Pre-recognition (provides hints, handles easiest cases)
- Layer 5: Full recognition (handles most words with high confidence)

The model concentrates its word-recognition capability in the FINAL
layer. This makes sense: Layer 5 has access to the most processed
residual stream, containing all the information accumulated by the
previous five layers. It's the layer with the most context to work with.
