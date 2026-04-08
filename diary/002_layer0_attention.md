# Diary Entry 002: Layer 0 Attention
## Stepping through Layer 0 of the dim=128 Gutenberg model

Input: "he said" = [h, e, SPACE, s, a, i, d]

### Layer Norm amplifies but preserves direction

Layer norm takes each 128-dim vector and normalizes it to have zero mean
and unit variance, then applies learned scale/shift. The result: norms
jump from ~0.9-1.2 to ~9.0-9.4, but the cosine similarity between before
and after is 0.984-0.988. So layer norm makes the vectors ~10x larger
while barely changing their direction. It's essentially scaling everything
up to a consistent magnitude.

### Q, K, V norms reveal head personality

Each head gets a 32-dimensional slice of the 128-dim Q, K, V projections.
The norms vary a lot between heads:

- **Head 1** has the largest Q and K norms (Q up to 4.04, K up to 4.13).
  This head is shouting — its attention scores will be large, making its
  softmax peakier. It's a decisive head.
- **Head 2** has the smallest norms (Q: 0.89-1.59, K: 0.91-1.83).
  Its attention will be more diffuse.
- **Head 1** also has the smallest V norms (0.70-1.20). Decisive about
  WHERE to look, but contributes small values. Interesting tradeoff.
- **Head 2** has the largest V norms (1.17-2.11). Diffuse attention but
  each attended value carries more weight.

### The attention matrices — each head has a different strategy

**Head 0 (diffuse, recent-biased):**
The attention is fairly spread out. No position gets more than 0.41 from
any query. The head tends to attend to recent characters — 'e' attends
mostly to itself (0.74), and later characters spread attention across
everything with a slight bias toward recent positions. This is a
"local context" head.

**Head 1 (previous-word biased):**
After the space, 's' attends most to itself (0.39) and to SPACE (0.35).
'a' and 'i' also attend most to SPACE and 's'. This head seems to care
about the word boundary — it's attending to the space and the beginning
of the current word.

**Head 2 (first-position fixated):**
After position 1, EVERY query's maximum attention is at position 0 ('h').
By position 6, 'h' still gets 0.17 (nearly uniform would be 0.14).
This is the first-position attention pattern we saw in the aggregate
probes. Even in continuous mode it develops, just not as strongly as
in sentence mode.

**Head 3 (strong first-position + local):**
'e' attends 0.61 to 'h' (position 0) — very strong first-position pull.
SPACE attends 0.55 to 'h'. But then 's' attends 0.33 to itself,
breaking the pattern. After the word boundary, this head seems to
"reset" and start attending locally again. Then 'd' attends 0.19 to
itself — the strongest self-attention in the row.

### Key insight: Head 3 detects word boundaries

Head 3's behavior changes at the space character. Before the space,
everything points at position 0. After the space, the head starts
attending to the current word instead. This could be how the model
marks "a new word has started" — the attention pattern literally
shifts from global to local at the word boundary.

### What this means for prediction

At this stage (Layer 0 output), the model has:
- Mixed information from nearby characters (Head 0)
- Word-boundary awareness (Head 1)
- Global position 0 context (Head 2)
- Word-boundary-sensitive local/global switching (Head 3)

None of these heads are doing character-specific computation yet.
They're providing CONTEXT — telling later layers "where are we in the
sequence" and "did a word just start." The actual character-level
prediction presumably happens in the MLP and in later layers.

Next: I want to see what the attention output actually looks like
(the V-weighted sum), and what the MLP does with it.
