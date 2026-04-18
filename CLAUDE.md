# BPE vs Character-Level Model Comparison

## FIRST: Read HANDOFF.md

At the start of every session, read `HANDOFF.md` in the project root.
It contains the current state of the project and pending tasks. Verify
each item marked TODO before proceeding with any work.

During your session, update HANDOFF.md whenever you:
- Write a diary entry
- Complete or abandon a task
- Make a decision that a future instance would need to know
- Discover something unexpected about the state of the project

Before your session ends, make sure HANDOFF.md is current and committed.

## What this project is

This project trains and compares transformer language models with
different tokenization: BPE (byte pair encoding) vs character-level
tokens. The goal is to understand how tokenization affects what a model
learns internally, building on 80+ diary entries of research into small
character-level transformer internals.

Ralph's broader research program: understanding the internals of small
GPT-type language models so humans can learn to interpret and improve them.

## Models

### Model B — Character-level (reference, training complete)
- Architecture: 16 layers, 16 heads, 2048 embedding dim, softmax attention
- Tokenizer: character-level, vocab size 52
- Parameters: ~822M
- Trained to iter 53,500, best val loss: 0.8322
- Corpus: gutenberg_corpus_MODERN_CLEAN.txt
- Checkpoint: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous.pt`
- Metadata: `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous_meta.pkl`

### BPE model — see HANDOFF.md for current training status

## Comparing losses between BPE and character models

BPE per-token loss CANNOT be compared directly to character per-token loss.
The BPE model predicts 1 of 8,192 tokens; the character model predicts 1
of 52 characters. To compare them, normalize to per-character loss:

    BPE per-character loss = BPE per-token loss / (chars per token)

To convert loss to probability of correct prediction:
    probability = e^(-loss)

## Key diary entries

The `diary/` directory contains research entries 001-085.
Most relevant:

- **085**: Corpus rebuild — document-level shuffling (this is the most recent)
- **084_B**: Corpus filtering and cleanup pipeline
- **082_B**: Gated-FFN (SwiGLU) model design for LARQL compatibility
- **081**: BPE experiment setup, results, and open questions
- **080**: Thermodynamics of training — layer crystallization order
- **074**: Hypothesis that tokenization is a topological operation
- **055**: Dark subspace — exists in char models, absent in BPE models
- **076**: Layer-by-layer activation analysis of a character model
- **049**: Synthesis of character model findings

## Project structure

- `py/` — Python source: model.py, train.py, tokenizer.py, sample.py,
  rebuild_corpus.py, filter_paragraphs.py, plot_train_val_loss.py, etc.
- `sh/` — Shell scripts for launching training runs
- `diary/` — Research diary entries
- `pt/` — Current checkpoints (gitignored, large files)
- `txt_local/` — Training corpora (gitignored, large files)
- `terminal_logs/` — Training output logs (committed for history)
- `plots/` — Generated plots (gitignored)
- `HANDOFF.md` — Current state and pending tasks (READ THIS FIRST)

## Technical notes

- The BPE tokenizer crashes on strings >~1GB. The encode() method in
  tokenizer.py chunks large texts into 100 MB pieces, splitting on
  newline boundaries.
- train.sh uses the --output filename (not --input) for log naming.
- Training uses python -u for unbuffered output to logs.
- Python environment: `/Users/RalphDratman/miniforge3/bin/python3`
  (Python 3.12). Do NOT use `/usr/bin/python3` (system Python 3.9,
  broken numpy). LaunchAgents must use the full miniforge path.
- GitHub repo: `dratman/small_transformer_research` (cloned locally
  at `../small_transformer_research/`). Auth via `gh auth login`.

## Preserved checkpoints

- `old_8_GB_corpus_pt/` — old 8 GB corpus run (iters 10K-160K)
- `unshuffled_corpus_pt/` — unshuffled 2.5 GB corpus run (iters 10K-50K)
- `../valuable_checkpoints/bpe_16L16H_old_corpus_iter160k_Excellent_from_8GB_corpus.pt`
