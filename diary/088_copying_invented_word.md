# Diary 088 — How a Character Model Copies an Invented Word

Date: 2026-04-27

## The observation

During free generation at iteration 20,000 of the 152M character model
training on the high-quality filtered corpus, the model produced:

    illustration: appalpittidax and marsilla heights by says regeneration
    of books (the early principal appalpittidax) had the great sensations
    for the ti

The word "appalpittidax" — a 13-character string that does not exist in
the training corpus or anywhere else — was generated twice. The model
invented a word and then reproduced it exactly, character by character,
86 positions later.

## The question

How did the model reproduce a word it has never seen in training? The
only source for "appalpittidax" is the model's own earlier output within
the same generation pass. The copying mechanism must operate through
attention to earlier positions in the context window.

## The analysis

Fed the exact generated text into the iter-20000 checkpoint and captured
attention weights at all 12 layers (8 heads each) for every character
position of the second "appalpittidax" (positions 100-112). The first
occurrence spans positions 14-27.

## Findings

### Layer 9, Head 3: the region-locking head

L9H3 sends 46-92% of its total attention to the first occurrence region
(positions 14-27) at every character of the second occurrence:

- Position 100 ("a"): 18.5% to first occurrence
- Position 101 ("p"): 56.2% — top target is position 16 ("p")
- Position 102 ("p"): 76.2% — top target is position 17 ("a")
- Position 103 ("a"): 79.7% — top target is position 18 ("l")
- Position 105 ("p"): 88.1% — top target is position 20 ("i")
- Position 106 ("i"): 90.7% — top target is position 21 ("t")
- Position 107 ("t"): 87.3% — top target is position 22 ("t")
- Position 108 ("t"): 83.3% — top target is position 23 ("i")
- Position 109 ("i"): 83.4% — top target is position 24 ("d")
- Position 110 ("d"): 92.2% — top target is position 25 ("a")
- Position 111 ("a"): 67.1% — top target is position 26 ("x")
- Position 112 ("x"): 46.0% — top target is position 27 (" ")

Critical observation: L9H3 does NOT attend to the corresponding
character. It attends to a character 2-3 positions AHEAD of the
corresponding position in the first occurrence. When generating "p"
(position 101, character #1), it looks at position 16 which is "p" —
but that is the first "p" in the first occurrence, which is character #1
there too. Wait — let me recount.

First occurrence: positions 14-27 = "appalpittidax" (14=a, 15=p, 16=p,
17=a, 18=l, 19=p, 20=i, 21=t, 22=t, 23=i, 24=d, 25=a, 26=x, 27=" ")

The pattern: L9H3 attends to positions that are 2-3 characters AHEAD
of the current character index in the first occurrence. When generating
character #3 ("a"), it looks at position 18 ("l", character #4). When
generating character #7 ("t"), it looks at position 22 ("t", character #8).
It is reading slightly ahead — as if it is gathering information about
what comes NEXT in the first occurrence, to predict what the model should
generate next in the second occurrence.

This makes sense: to predict the next character at position N, the model
needs to know what character follows the current position in the source.
L9H3 is reading "what comes next in the first occurrence" to inform the
prediction of "what should come next in the second occurrence."

### Layer 11: precise character matching

L11H0 shows the highest attention to the CORRESPONDING character:

- Position 108 ("t"): 22.8% to corresponding position in first occurrence
- Position 111 ("a"): 22.4% to corresponding position (25, which is "a")
- Position 112 ("x"): 16.9% to corresponding position (26, which is "x")

L11H0 also shows very high total attention to the first occurrence:
- Position 107 ("t"): 74.6%
- Position 108 ("t"): 81.2%
- Position 110 ("d"): 73.3%
- Position 111 ("a"): 60.4%

### Layer 10: intermediate processing

L10H1 and L10H4 show moderate attention to the first occurrence,
particularly to position 26 ("x") — the last character before the
space. This may be the model tracking "how far through the word am I?"

### Layer 0: uniform background

All 8 heads in layer 0 show ~13% attention to the first occurrence.
This is not meaningful — the first occurrence is 14 of 100+ positions,
so uniform attention gives about 14%. Layer 0 is not participating in
the copying.

### Layers 2-5: not involved

These layers show essentially zero attention to the first occurrence.
They are doing something else (local character processing, perhaps).

## The mechanism

The copying is a two-stage process distributed across layers:

1. **Stage 1 (Layer 9, Head 3): Region identification and read-ahead.**
   A single head locks onto the first occurrence as a region and reads
   slightly ahead of the current position. It provides information about
   "what comes next in the source word" to the residual stream.

2. **Stage 2 (Layer 11, Head 0): Character-level confirmation.**
   A later head attends to the specific corresponding character in the
   first occurrence, perhaps verifying or refining the prediction that
   L9H3 initiated.

The model does not do simple character-by-character copying via one head
attending to one position. It uses a layered strategy where different
layers contribute different aspects of the copying operation.

## Significance

1. **The attention mechanism can copy arbitrary novel sequences.**
   The word "appalpittidax" was never in training. The copying mechanism
   is general — it works on any character sequence in the context window.

2. **Copying is not a simple lookup.** It involves multiple layers with
   different roles (region identification vs. character matching).

3. **The read-ahead pattern in L9H3 is consistent with next-character
   prediction.** The model's task is to predict the next character. To
   copy, it needs to know what the next character in the source is.
   Attending one position ahead in the source gives exactly that signal.

4. **This is happening at iteration 20,000 (epoch 0.13).** The model
   has seen only 13% of the corpus once. The copying mechanism developed
   very early in training — it is one of the first capabilities to emerge.

## Connection to earlier diary entries

- Entry 014: "Layer 4 attention gathers word characters, Layer 4 MLP
  interprets." Here, Layer 9 gathers the source word characters.
- Entry 015: "Head H2 reads first letter at 95-99%." Here, L9H3 reads
  the source region at 46-92%.
- Entry 036: "Attention reads predictions from previous positions."
  Here, the model reads its own earlier output to inform predictions.

## Model details

- Architecture: 12 layers, 8 heads, n_embd=1024, block_size=512
- Parameters: 152M
- Tokenizer: character-level, vocab=52
- Checkpoint: iter 20,000 (val loss 1.037)
- Corpus: corpus_high_quality_2026_04_26.txt (1.42 GB, 4,430 books)
