# Diary 087 — Holographic Framing of Transformer Internals

Date: 2026-04-26

## Context

This entry records a conversation exploring the analogy between
holographic storage and what happens inside a transformer. The
conversation grew out of the imitator experiment (diary 086) and
the observation that free-running generation is much more coherent
than forced next-token prediction.

## The holographic analogy

The weights of a transformer store information in a distributed,
superimposed manner. Many patterns are recorded on the same set of
parameters, just as many images can be recorded on a single holographic
film. Retrieval happens when the right input activates the relevant
stored patterns, just as the right reference beam reconstructs the
right image from a hologram.

This is not a new observation (see references below), but it has not
been developed into a unified account specific to language.

## Literature found

- Plate, "Holographic Reduced Representations" (1991, book 2003) —
  circular convolution for binding, addition for superposition,
  correlation for retrieval. Explicit holographic operations.
- Ramsauer et al., "Hopfield Networks is All You Need" (2020) —
  transformer attention is mathematically equivalent to modern Hopfield
  network update rule (associative memory retrieval).
- Elhage et al., "Toy Models of Superposition" (2022, Anthropic) —
  neural networks store more features than they have dimensions by
  packing them into near-orthogonal directions. Features interfere;
  sparsity makes interference tolerable.
- "Hrrformer" (2022/2023) — replaces attention with explicit HRR
  operations. 370x faster to train but quality not yet competitive.
- "Energy Transformer" (2023) — replaces transformer blocks with a
  single associative memory energy minimization.

The total body of work is small — maybe a dozen papers in 35 years.
Not a crowded field.

## Step-by-step holographic interpretation of a single layer

1. **Input signal:** The sequence of vectors along the text axis is a
   discrete signal in 2048-dimensional space.

2. **Attention as autocorrelation:** The query at position i is compared
   (dot product) with keys at all earlier positions. This is a
   correlation of the signal with itself. The dot product is the
   nonlinear interference measurement. The Wiener-Khinchin theorem
   relates autocorrelation to the power spectrum — so attention is
   implicitly operating in the frequency domain.

3. **Weighted sum as reconstruction:** The attention-weighted sum of
   value vectors is the holographic reconstruction — a new signal
   created from the interference measurements.

4. **Residual connection as superposition:** The attention output is
   ADDED to the original vector. The original information is preserved;
   new information (the reconstruction) is superimposed on top.

5. **MLP as nonlinear recording:** The accumulated signal is projected
   to 4x higher dimension, passed through GELU (nonlinearity), and
   projected back. This is analogous to the nonlinear recording step
   in holography — the squared-magnitude response that captures phase
   information. The MLP output is added to the residual (another
   superposition).

6. **Repeat:** The next layer correlates the enriched signal with itself,
   extracting structure that was not visible in the raw input.

After N layers, the residual stream is the sum of 1 + 2N contributions:
the original embedding, plus one attention output and one MLP output per
layer. All superimposed.

## Frequency hierarchy hypothesis (unverified)

Each layer may extract different frequency components of the signal:

- Early layers: high-frequency (local, character-level patterns)
- Middle layers: medium-frequency (word boundaries, cross-word patterns)
- Late layers: low-frequency (word identity, syntactic role, prediction)

This matches the empirical findings from diary entries 001-076 for the
character model. But the mechanism by which attention and MLP act as
frequency filters is not yet clear. Attention is data-dependent (adaptive
filtering), not a fixed filter. The MLP filters along the feature
dimension, not the position dimension. Whether their combination
produces something equivalent to successive Fourier component extraction
is an open question.

## Connection to the imitator experiment

The imitator (diary 086) predicts the next vector in the residual stream
at layer 8 — that is, it predicts the accumulated interference pattern
at the midpoint of the processing pipeline. The high cosine similarity
(0.948) means the interference pattern is highly predictable from one
position to the next. The poor token-level match (15-20%) means the
subtle details of the interference pattern — the details that determine
which specific token the back half decodes — are in the remaining 5%
of directional error.

## Connection to the nonlinearity in holographic recording

Ralph noted that holographic recording requires a nonlinear step — the
photographic film records intensity (squared magnitude of the field),
not amplitude. This squaring is what captures phase information.

In the transformer, the analogous nonlinear steps are:
- Softmax in attention (converts dot products to weights)
- GELU in the MLP (selectively passes or suppresses components)
- The dot product itself (bilinear — two vectors produce a scalar)

Without these nonlinearities, the entire transformer would collapse to
a single linear transformation and could not store or retrieve anything
interesting.

## Open questions

1. Can the layer-by-layer process be described rigorously as successive
   Fourier component extraction? What is the mechanism of frequency
   separation?

2. Could one build a language model using explicit holographic operations
   (circular convolution, superposition, correlation) that produces
   coherent text? The HRR framework provides the operations; nobody has
   demonstrated competitive language generation.

3. What is the relationship between the Wiener-Khinchin theorem (linking
   autocorrelation to power spectrum) and the attention mechanism?

4. Is there a way to simplify the training process by designing the
   interference patterns explicitly rather than discovering them through
   gradient descent?

## References to own diary entries

- 002: Layer 0 attention — four specialized heads with different
  correlation ranges (high and low frequency)
- 030: Different words activate near-orthogonal neuron ensembles —
  analogous to different reference beam angles
- 042: 368 weak neurons matter more than 13 strong ones — distributed
  holographic signal, not sparse coding
- 052: Per-component decomposition of residual stream — the individual
  "waves" in the interference sum
- 065: Two trainings produce different weights but same predictions —
  knowledge is in the interference pattern, not the specific weights
