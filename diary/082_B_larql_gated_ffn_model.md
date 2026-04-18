# Diary 082: LARQL and a Gated-FFN Model for Queryable Internals

Date: 2026-04-14

## What is LARQL?

LARQL (Lazarus Query Language) is a Rust tool by chrishayuk that decompiles
transformer weights into queryable vector indices called "vindexes." It
treats the model as a database: you can walk through layers, describe what
features each layer has learned, query gate vectors via KNN, edit knowledge
with lightweight JSON patches, and even recompile. The query language (LQL)
has SQL-like operations: WALK, DESCRIBE, INFER, INSERT, DELETE, BEGIN PATCH.

Repository: https://github.com/chrishayuk/larql

Key performance claims: gate KNN operations in 0.008ms per layer, full
34-layer walks in 0.3ms, inference on Gemma 3 4B at 517ms on Apple Silicon.
Supports Gemma, Llama, Mistral, Mixtral, Qwen, Phi, DeepSeek.

Found via a YouTube presentation. Could be valuable for our research goal
of understanding and interpreting small transformer internals.

## Why our existing models can't use LARQL

LARQL's entire feature-extraction system is built around **gated FFN layers**
(LLaMA/Gemma SwiGLU style). The vindex extraction reads the gate projection
matrix from each layer and indexes its rows as queryable features.

Our models use standard GPT-2 FFN:

    output = GELU(x @ W_fc) @ W_proj        # two matrices, no gate

LARQL expects gated FFN (SwiGLU):

    output = SiLU(x @ W_gate) * (x @ W_up) @ W_down   # three matrices

Without a gate projection matrix, LARQL's extraction silently produces
an empty vindex — no features, nothing to query. This is not a format
issue; it's a fundamental architectural mismatch.

Additional obstacles (all solvable):
- Format: our .pt checkpoints need conversion to SafeTensors + config.json
- Tensor naming: our key names (c_fc, c_proj, c_attn) differ from HuggingFace
- Tokenizer: needs export to HuggingFace tokenizer.json format

## Plan: train a third model with gated FFN

Add a GatedMLP option to model.py and train a new model:

### Architecture for the gated-FFN model
- 16 layers, 16 heads, 2048 embedding dim — same as Models B and BPE
- Softmax attention — same CausalSelfAttention, unchanged
- **Gated FFN (SwiGLU)** — the one change
- LayerNorm with bias — same as existing models (not RMSNorm)
- Learned position embeddings — same as existing models
- BPE tokenizer, vocab 8,192 — same as current BPE model
- Same corpus

### Parameter budget

Standard FFN has 2 weight matrices per layer:
    c_fc:   2048 × 8192 = 16.8M
    c_proj: 8192 × 2048 = 16.8M
    Total: 33.6M per layer

Gated FFN has 3 weight matrices. To match parameter count, set
intermediate_size = (2/3) × 8192 = 5461, rounded to 5504:
    gate_proj: 2048 × 5504 = 11.3M
    up_proj:   2048 × 5504 = 11.3M
    down_proj: 5504 × 2048 = 11.3M
    Total: 33.8M per layer — nearly identical

### Design decisions

**Keep LayerNorm (not RMSNorm):** LARQL defaults to RMSNorm, but the
difference is minor (RMSNorm omits mean-centering). Keeping LayerNorm
means one fewer variable when comparing against our existing models.
Handled on the LARQL export side via architecture config.

**Keep bias=True:** LLaMA-style models drop all bias vectors. Our existing
models have bias=True throughout. Keeping bias maintains comparability —
one variable at a time. The bias vectors are ignored during LARQL vindex
extraction, so no compatibility issue.

**CAUTION:** If LARQL does not work as expected with this model, the
LayerNorm and bias decisions should be revisited as potential causes.
LARQL was designed for LLaMA/Gemma-style models which use RMSNorm and
no bias. While these differences should be handled by the architecture
config and should not affect vindex extraction (which only reads weight
matrices), untested combinations can have unexpected interactions.
Specifically:
- LayerNorm bias terms could shift activation distributions in ways that
  affect how gate vectors activate, potentially making LARQL's feature
  labels less interpretable
- If LARQL's inference path (not just extraction) is used, norm type
  mismatch would produce wrong results unless correctly configured

## What this enables

Three directly comparable models:

| Model | Tokenizer | FFN type | LARQL? | Status |
|-------|-----------|----------|--------|--------|
| Model B (char) | char, vocab 52 | standard | no | trained |
| BPE model | BPE, vocab 8192 | standard | no | training |
| Gated-FFN model | BPE, vocab 8192 | gated SwiGLU | yes | planned |

Comparing BPE-standard vs BPE-gated isolates the effect of the FFN
architecture with tokenization held constant. Persistent homology,
activation analysis, and dark subspace measurements can be applied to
all three. LARQL queries can be applied to the gated model.

## Connection to existing research

- **Diary 074 (persistent homology):** The build→peak→simplify→compress
  pattern across layers should be testable via LARQL's WALK command on
  the gated model. Do gate vector KNN clusters show the same arc?

- **Diary 055 (dark subspace):** With BPE vocab 8192 > dim 2048, there
  is no structural dark subspace. But LARQL's gate vectors might reveal
  a de facto dark subspace — features that activate but don't surface
  in predictions.

- **Diary 076 (layer-by-layer analysis):** LARQL's DESCRIBE could show
  what concepts each layer's gate vectors represent, complementing the
  attention-vs-MLP analysis. Does the gate learn the same consolidation
  → differentiation → segmentation pipeline?

## Implementation steps

1. Add GatedMLP class to model.py (config flag: use_gated_mlp)
2. Add intermediate_size to GPTConfig (default: 5504 for dim 2048)
3. Create training launch script
4. Train on same corpus as BPE model
5. Write SafeTensors + config.json export script
6. Write tokenizer.json export script
7. Install LARQL, extract vindex, test queries

## See also

- Diary 081: BPE vs character-level comparison setup
- Diary 074: Persistent homology / topological crystallization
- Diary 055: Dark subspace universality
- Diary 076: Layer-by-layer activation analysis
