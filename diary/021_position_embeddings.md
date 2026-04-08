# Diary Entry 021: Position Embeddings — Three Regimes
## The position embedding structure across 256 positions

### Three distinct regimes

The position embedding norms reveal a U-shaped curve with three zones:

**Zone 1: Position 0 (norm 0.612)**
Position 0 has an embedding 3-4× larger than all others. It's a unique
position — the only one the model can't look "before." Layer 1 broadcasts
position 0's representation to all other positions, so position 0 serves
as a global context anchor. Its large embedding ensures it has a strong
initial signal even before any attention processing.

**Zone 2: Positions 1-200 (norm 0.14-0.19)**
The "middle" of the context window. Position embeddings are small and
similar. Adjacent positions have cosine similarity 0.94-0.97. The model
barely distinguishes position 50 from position 51 — it relies on
attention patterns (self, prev) rather than absolute position for
local processing.

**Zone 3: Positions 200-255 (norm rising to 0.24)**
Norms increase toward the end of the window. The model gives slightly
more positional weight to the END of the context. Why? Possibly because
in continuous mode, the prediction at position 255 is the "final" one,
and the model wants more precise position information near the edge
of its context window.

### Adjacent positions are nearly identical

For positions 32+, adjacent cosine similarities are 0.94-0.97. This
means the position embeddings change very smoothly — they're essentially
a continuous function discretized into 256 steps. The model doesn't need
sharp position boundaries; it just needs a slowly-varying signal that
lets attention know roughly "how far back" to look.

The exception: position 0↔1 has cosine only 0.065. Position 0 is
radically different from position 1. The model treats position 0 as
a special signal, not as "the first of a smooth sequence."

### Position 0 has negative cosine with distant positions

pos 0 ↔ pos 64: -0.29
pos 0 ↔ pos 128: -0.35

Position 0's embedding is anti-correlated with positions in the middle
and second half of the window. When Layer 1 reads position 0 and
broadcasts its value, the anti-correlation with the local position
embedding creates a CONTRAST — the model can tell "this information
came from position 0, not from here" because the directions differ.

### The position embedding uses low-frequency structure

The FFT of position norms shows dominant frequencies at period 256 and
128 — very low-frequency oscillations. There's no high-frequency
periodic structure (no sinusoidal encoding was designed in — it's
learned). The model has settled on smooth, slowly-varying position
signals, which matches how it uses positions: for rough "how far back
to look" rather than exact position identification.

### Comparison with the original transformer

The original transformer used fixed sinusoidal position encodings with
many frequencies. This model learns its own, and discovers that only
the lowest frequencies matter. This makes sense for a character-level
model: the relevant positional information is "am I near the start of
the word?" and "am I near the start of the context?" — both low-frequency
questions. Character-level predictions don't benefit from knowing "I am
at exactly position 137 vs 138."

### Implications for the model's architecture

The position embedding's three-regime structure creates a natural
hierarchy:
1. Position 0 is "the beginning" — a unique global signal
2. Positions 1-200 are "the middle" — interchangeable, processed locally
3. Positions 200+ are "near the end" — slightly more position-aware

This hierarchy complements the attention patterns: Layer 1 reads the
unique position 0 signal, Layers 3-4 do local processing in the uniform
middle zone, and Layer 5 reads back to specific positions for context.
The position embeddings don't need to be precise because the attention
mechanism handles fine-grained position selection.
