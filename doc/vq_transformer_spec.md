# VQ Bottleneck Transformer: Experiment Spec

## Overview

This experiment replaces the feedforward (MLP) sublayer in each transformer block
with a vector-quantization (VQ) codebook bottleneck. The hypothesis is that
discretization provides sufficient nonlinearity to replace GELU, while also
forcing representations at each layer to converge onto a shared discrete vocabulary
of equivalence classes — potentially encouraging more structured internal
representations than a continuous residual stream.

This is a first exploratory test. The goal is to verify that the architecture
trains at all and produces coherent text generation.

---

## Base Code

Start from the existing `model.py`. The architecture to modify is the `Block` class
and `GPTConfig`. Do not modify the attention mechanism. Do not modify LayerNorm,
DimensionAdapter, or the GPT class except as minimally required to support the new
config flags described below.

---

## What to Remove

Remove the MLP sublayer entirely from each `Block`. Specifically:

- Remove `self.ln_2` (the LayerNorm that precedes the MLP)
- Remove `self.mlp`
- Remove the line `x = x + self.mlp(self.ln_2(x))` from `Block.forward()`

---

## What to Add

### 1. VQBottleneck module

Add a new `nn.Module` class called `VQBottleneck`. It takes the following constructor
arguments:

- `codebook_size: int` — number of discrete entries (default 40)
- `dim: int` — embedding dimension (must match residual stream width)

Internally it holds:
- `self.codebook = nn.Embedding(codebook_size, dim)` — the learned discrete entries

The `forward(x)` method:

1. Input `x` has shape `[B, T, dim]`
2. Reshape to `[B*T, dim]`
3. Compute L2 distances between each token vector and every codebook entry.
   Use the standard expansion: `||x - e||^2 = ||x||^2 + ||e||^2 - 2*x·eᵀ`
4. Find the nearest codebook entry index for each token: `indices = argmin(distances)`
5. Look up the quantized vectors: `x_q = codebook(indices)`, shape `[B*T, dim]`
6. Apply straight-through estimator:
   `x_st = x_flat + (x_q - x_flat).detach()`
   This passes gradients through as if the operation were identity, while the
   forward pass uses the quantized values.
7. Reshape back to `[B, T, dim]`
8. Return `x_st` and `indices` (indices shape `[B*T]`, useful for monitoring)

### 2. Commitment loss

The VQ bottleneck needs an auxiliary commitment loss to encourage the continuous
representations to stay close to their nearest codebook entries. This is the
standard VQ-VAE loss term:

`commitment_loss = mean((x_flat.detach() - x_q) ** 2)`

Include a `commitment_loss` scalar in the return value of `VQBottleneck.forward()`,
so the caller can accumulate it across layers.

Return signature: `(x_st, indices, commitment_loss)`

### 3. Modified Block

The new `Block.forward()` should be:

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x_vq, indices, commit_loss = self.vq(x)
    x = x_vq
    x = self.adapter(x)
    return x, commit_loss
```

Note: the VQ replaces the residual-add pattern. The quantized output directly
replaces `x` rather than being added to it. This is intentional — the codebook
entry IS the new representation, not a delta.

### 4. GPT.forward() accumulation

Modify `GPT.forward()` to accumulate commitment losses across all layers and add
them to the main cross-entropy loss with a small weight:

```python
total_commit_loss = sum of commit_loss from each block
loss = ce_loss + commit_weight * total_commit_loss
```

`commit_weight` should be a config parameter defaulting to `0.25`.

---

## Config Changes

Add the following fields to `GPTConfig`:

```python
use_vq: bool = False           # Enable VQ bottleneck (replaces MLP)
vq_codebook_size: int = 40     # Number of codebook entries per layer
vq_commit_weight: float = 0.25 # Weight for commitment loss term
```

When `use_vq=False`, the block should behave exactly as before (MLP present,
no VQ). This allows direct comparison between the two architectures using the
same config infrastructure.

---

## Key Design Decisions (do not change without discussion)

- **One codebook per layer**: Each `Block` has its own independent `VQBottleneck`
  instance. Codebooks are not shared across layers.
- **Codebook size = 40**: Matching the lowercase character vocabulary size.
  This is intentional and meaningful for interpretability.
- **No MLP**: The feedforward sublayer is entirely absent. The VQ bottleneck
  provides the only nonlinearity beyond attention.
- **No GELU**: Removed along with the MLP. The discretization is hypothesized
  to provide sufficient nonlinearity.
- **Straight-through estimator**: Standard VQ-VAE gradient trick. Do not use
  Gumbel-softmax or other alternatives for this first test.
- **VQ replaces, not adds**: The quantized vector replaces `x` outright rather
  than being added as a residual. This is the natural semantics for a
  re-tokenization operation.

---

## Small Test Config

Use a small fast config for initial testing. Something like:

```python
GPTConfig(
    vocab_size=40,        # lowercase character model
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=False,
    use_vq=True,
    vq_codebook_size=40,
    vq_commit_weight=0.25,
    tie_weights=False,    # weight tying may not make sense with VQ replacing MLP
)
```

---

## Monitoring

Add logging for the following quantities during training (in addition to the
normal loss):

- `commit_loss` (the raw unweighted commitment loss, averaged across layers)
- Codebook utilization per layer: what fraction of the 40 codebook entries are
  actually being used across a batch. A healthy VQ should use most entries;
  codebook collapse (most tokens mapping to one entry) is the main failure mode
  to watch for.

Codebook utilization can be computed cheaply by counting unique indices in each
layer's `indices` tensor.

---

## Expected Failure Modes

1. **Codebook collapse**: All tokens map to one or a few codebook entries.
   Symptom: utilization near 0. If this happens, try reducing `commit_weight`
   or adding a small amount of noise to codebook entries during training.

2. **Loss fails to decrease**: The straight-through estimator may not provide
   sufficient gradient signal early in training. If loss is stuck, try
   initializing codebook entries from the embedding layer weights.

3. **Incoherent generation**: Expected at very small scale; the question is
   whether loss decreases at all, not whether output is fluent.

---

## Files to Modify

- `model.py`: Add `VQBottleneck`, modify `Block`, modify `GPTConfig`,
  modify `GPT.forward()`
- `train.py`: Add logging for `commit_loss` and codebook utilization.
  No other changes expected.

Do not create a separate model file. Keep everything in `model.py` for
consistency with the existing codebase.
