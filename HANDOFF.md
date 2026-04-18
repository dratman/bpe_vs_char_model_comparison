# Handoff Document

Last updated: 2026-04-18 16:30 by Claude Opus (Mac Studio session)

## Current State

### Training
- **No training is currently running.**
- The previous run (on unshuffled corpus) was stopped at iter 52,880,
  loss 2.88. Checkpoints preserved in `unshuffled_corpus_pt/`.

### Corpus
- A new corpus is ready but has NOT been used for training yet:
  `txt_local/corpus_books_shuffled_2026_04_18.txt` (2.14 GB, 6,496 books)
- This corpus uses document-level shuffling (books intact, book order
  randomized) with `<|endoftext|>` separators between books. See diary
  085 for full details.
- The tokenizer (`py/tokenizer.py`) has been updated to include
  `<|endoftext|>` as a special token.

### What needs to happen before training starts
- **TODO: Discuss hyperparameters with Ralph before starting any training.**
  Key parameters to decide:
  - `block_size` — was 1024, should be larger (see diary 085 for why)
  - `batch_size` — was 16, may need to decrease if block_size increases
    (memory constraint)
  - `max_iters` — was 200,000
  - `learning_rate` — was 0.0003
  - All other parameters may stay the same but should be confirmed.
- **TODO: Verify that train.py handles the `<|endoftext|>` token correctly.**
  The tokenizer will encode it as a single special token. Make sure the
  training loop doesn't do anything unexpected with it (e.g., treating
  it as a stop token or masking loss at document boundaries).

### Corpus files on disk (txt_local/, gitignored)
- `corpus_books_shuffled_2026_04_18.txt` — NEW, ready for training (2.14 GB)
- `corpus_cleaned_shuffled_2026_04_18.txt` — paragraph-shuffled version (not recommended, 2.16 GB)
- `corpus_final_2026_04_18.txt` — paragraph-filtered, single-newline format (2.16 GB)
- `corpus_cleaned_2026_04_15.txt` — UNSHUFFLED, the one that caused the problem (2.5 GB)
- `corpus_of_gutenberg_novels_cleaned_shuffled_2026_03_03_B.txt` — original 8 GB corpus

### LaunchAgents running
- `com.ralph.thermal-monitor` — checks thermal throttling every 15 min,
  plays Crystals alert if throttling detected
- `com.ralph.loss-plot-updater` — updates loss comparison plot every 15 min
  (will error when no training is running; harmless)

### GitHub
- Repo: `dratman/small_transformer_research`
- Local clone: `../small_transformer_research/`
- Auth: via `gh auth login` (configured 2026-04-15)
- Last pushed: French corpus instructions (2026-04-15)
- This repo (bpe_vs_char_model_comparison) is NOT pushed to GitHub.
  It is a separate local repo.

## Recent Decisions

- **Document-level shuffling** replaces paragraph-level shuffling.
  Based on research showing all major LLMs shuffle at document level.
  See diary 085. (2026-04-18)
- **English-only corpus** — non-English books filtered by function-word
  frequency test. French phrases within English novels are kept.
  (2026-04-18)
- **Dialect books removed** — 139 books with heavy dialect (>2% dialect
  markers) removed entirely rather than filtering individual paragraphs.
  (2026-04-18)
- **No French corpus expansion for now** — the 1.2% incidental French in
  the earlier corpus produced surprisingly good French output, but Ralph
  decided the current English-only corpus is sufficient. The idea is saved
  in memory for potential future work. (2026-04-15)

## Open Questions

- What block_size to use with the new document-shuffled corpus?
  Larger is better (fewer cross-document transitions per block) but
  increases memory usage. Need to test what fits in 192 GB RAM with
  the current model size.
- Should train.py mask loss at `<|endoftext|>` boundaries? Standard
  practice varies — GPT-2/3 don't mask, LLaMA 3 does. For our model
  size and context length, probably doesn't matter much.
