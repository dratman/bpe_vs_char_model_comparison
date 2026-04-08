# Diary 081: BPE vs Character-Level Model Comparison — First Results

Date: 2026-04-07

## The experiment

Trained a BPE model with identical architecture to the character-level
Model B, to compare how tokenization affects learning at the same scale.

### Model B (character-level) — reference model
- Architecture: 16L / 16H / 2048 embd, softmax attention, tied weights
- Tokenizer: char, vocab size 52
- Corpus: gutenberg_corpus_MODERN_CLEAN.txt
- Reached iter 53,500, best val loss 0.8322
- Checkpoint: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous.pt`
- Metadata: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous_meta.pkl`
- Parameters: ~822M

### BPE model — new training run
- Architecture: 16L / 16H / 2048 embd, softmax attention, tied weights
- Tokenizer: BPE, vocab size 8,192
- Corpus: corpus_of_gutenberg_novels_cleaned_shuffled_2026_03_03_B.txt
  (8.3 billion characters → 2.03 billion BPE tokens)
- max_iters set to 200,000 with plan to stop early
- Training started: 2026-04-05 17:54
- Checkpoint: `bpe_vs_char_model_comparison/pt/bpe_16L16H.pt`
- Training log: `bpe_vs_char_model_comparison/terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt`
- Parameters: ~822M (same architecture, slightly different due to vocab size)

## Setup issues resolved (April 5)

1. HuggingFace tokenizers Rust panic on 8.3B character string — fixed by
   chunking encode() into 100 MB pieces in tokenizer.py.
2. Log and checkpoint naming — modified train.sh to use --output basename
   for the log filename instead of the input corpus name.
3. Progress messages — added timestamps for tokenization, tensor
   conversion, and train/val split phases in train.py.

## Loss comparison — why BPE loss looks higher

BPE per-token loss cannot be compared directly to character per-token loss
because BPE predicts over 8,192 choices while the character model predicts
over 52.

To normalize: divide BPE per-token loss by the average characters per token.
The corpus has 8.3B characters / 2.03B tokens = ~4.09 characters per token.

| Metric | Model B (char) | BPE model (at 51h) |
|---|---|---|
| Per-token val loss | 0.8322 | 3.36 |
| Per-character equivalent | 0.8322 | 3.36 / 4.09 ≈ 0.82 |

The per-character losses are remarkably close, suggesting both models
extract similar information per character despite very different
tokenization.

## Training progress as of April 7

- Iter ~35,600, epoch 0.32 (seen 32% of data once)
- Loss: ~3.32 per token (≈ 0.81 per character equivalent)
- Speed: 0.20 iters/sec (~5 sec/iter), steady
- Training time elapsed: ~51 hours

Power-law extrapolation (R² = 0.998) predicts loss of 3.305 in 24 hours.
The curve is in deep diminishing returns numerically, but small loss
decrements at this stage can produce noticeable sample quality improvements
because they reflect better ranking among the top candidate tokens.

## Key structural advantage of BPE

With 1024-token context and ~4.09 characters per token, the BPE model
sees ~4,000 characters of context vs 1,024 for the character model.
This 4x longer effective context should help with coherence over longer
passages — sentences, paragraphs, narrative flow.

The BPE model also gets word-level structure for free from the tokenizer.
As predicted in diary 074, the character model must spend network capacity
building words from characters, while the BPE model can use those layers
for higher-level linguistic structure.

## Connection to prior diary entries

- **Diary 074**: Predicted that tokenization acts as a topological
  operation, eliminating dimensions the character model must build from
  scratch. This BPE run enables the planned comparison.
- **Diary 055**: Character models have a dark subspace (vocab 52 << dim
  2048). BPE model (vocab 8192 > dim 2048) does not — an important
  structural difference.
- **Diary 080**: Layer crystallization and thermodynamics of training.
  The BPE model's intermediate checkpoints (iter 10k, 20k, 30k) could
  enable the same layer stability analysis on a BPE model.

## Open questions

1. Will the BPE model produce noticeably better samples than Model B
   despite similar per-character loss? The 4x context advantage suggests
   it should.
2. Do the same topological patterns (consolidation → differentiation →
   segmentation) appear in BPE model layers, or does the tokenizer
   eliminate some of these phases?
3. Does the BPE model show the same lower-layers-freeze-first pattern
   as the character model (diary 080)?
4. How far should we let the BPE model train? It has only seen 32% of
   the corpus once. Model B reached 53,500 iters — the BPE model is at
   35,600 and still improving.
