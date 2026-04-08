# BPE vs Character-Level Model Comparison

## What this project is

This project trains and compares two transformer language models with
identical architecture but different tokenization: one using BPE (byte pair
encoding) and one using character-level tokens. The goal is to understand
how tokenization affects what a model learns internally, building on 80
diary entries of research into small character-level transformer internals.

Ralph's broader research program: understanding the internals of small
GPT-type language models so humans can learn to interpret and improve them.

## The two models

### Model B — Character-level (reference, training complete)
- Architecture: 16 layers, 16 heads, 2048 embedding dim, softmax attention
- Tokenizer: character-level, vocab size 52
- Parameters: ~822M
- Trained to iter 53,500, best val loss: 0.8322
- Corpus: gutenberg_corpus_MODERN_CLEAN.txt
- Checkpoint: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous.pt`
- Metadata: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous_meta.pkl`

### BPE model (currently training as of 2026-04-07)
- Architecture: 16 layers, 16 heads, 2048 embedding dim, softmax attention
- Tokenizer: BPE, vocab size 8,192
- Parameters: ~822M
- Corpus: corpus_of_gutenberg_novels_cleaned_shuffled_2026_03_03_B.txt
  (8.3 billion characters, 2.03 billion BPE tokens)
- max_iters: 200,000 (plan to stop early based on quality)
- Checkpoint: `pt/bpe_16L16H.pt`
- Intermediate checkpoints: `pt/bpe_16L16H_iter{N}.pt`
- Training log: `terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt`
- Launch script: `sh/train_bpe_16L16H.sh`
- Training speed: ~0.20 iters/sec (~700 iters/hour)

## Comparing losses between the two models

BPE per-token loss CANNOT be compared directly to character per-token loss.
The BPE model predicts 1 of 8,192 tokens; the character model predicts 1
of 52 characters. To compare them, normalize to per-character loss:

    BPE per-character loss = BPE per-token loss / (chars per token)

The corpus has 8.3B chars / 2.03B tokens = ~4.09 characters per BPE token.

Example: BPE loss 3.36 per token ≈ 0.82 per character, comparable to
Model B's 0.8322 per character.

To convert loss to probability of correct prediction:
    probability = e^(-loss)
So loss 0.82 per character means ~44% chance of predicting each character
correctly, and loss 3.36 per BPE token means ~3.5% chance of predicting
each token correctly out of 8,192 possibilities.

## BPE model's structural advantage

With 1024-token context and ~4.09 chars/token, the BPE model sees ~4,000
characters of context vs 1,024 for the character model. This 4x longer
effective context should improve coherence over longer passages.

The BPE model also gets word-level structure from the tokenizer for free,
while the character model must spend network capacity learning to assemble
characters into words. See diary entries 074 and 055 for the theoretical
implications.

## Key diary entries for context

The `diary/` directory contains 78 research entries (001-076, 080-081).
Most relevant to the BPE comparison:

- **081**: This experiment's setup, results, and open questions
- **080**: Thermodynamics of training — layer crystallization order
- **074**: Hypothesis that tokenization is a topological operation
- **055**: Dark subspace — exists in char models (vocab << dim), absent
  in BPE models (vocab > dim)
- **076**: Layer-by-layer activation analysis of a character model
- **049**: Synthesis of character model findings

## Project structure

- `py/` — Python source: model.py, train.py, tokenizer.py, sample.py
- `sh/` — Shell scripts: train.sh (wrapper), train_bpe_16L16H.sh
- `diary/` — Research diary entries 001-081
- `pt/` — Checkpoints (gitignored, large files)
- `txt_local/` — Training corpus (gitignored, ~8 GB)
- `terminal_logs/` — Training output logs (gitignored)
- `plots/` — Generated plots (gitignored)
- `plot_loss.py` — Plot training loss vs time
- `plot_loss_extrapolated.py` — Plot with power-law extrapolation

## Technical notes

- The BPE tokenizer crashes on strings >~1GB. The encode() method in
  tokenizer.py chunks large texts into 100 MB pieces, splitting on
  newline boundaries. This was fixed on 2026-04-05.
- train.sh uses the --output filename (not --input) for log naming.
- Training uses python -u for unbuffered output to logs.
- Progress messages during tokenization, tensor conversion, and
  train/val split were added to train.py for visibility.
