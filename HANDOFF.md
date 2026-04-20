# Handoff Document

Last updated: 2026-04-19 12:30 by Claude Opus (Mac Studio session)

## Current State

### Training
- **Training IS running** — launched 2026-04-19 11:24
- Launch script: `sh/train_bpe_32k_bf16.sh`
- Log: `terminal_logs/terminal_log_for_bpe_16L16H_2026_04_19_1124.txt`
- Corpus: `txt_local/corpus_books_shuffled_2026_04_18.txt` (2.14 GB, 6,496 books)
- Key hyperparameters:
  - vocab_size: 32000 (BPE) — increased from 8192
  - precision: bfloat16 — changed from float32
  - block_size: 2048 — increased from 1024
  - batch_size: 4
  - learning_rate: 0.00015 (sqrt-scaled for batch=4)
  - warmup_iters: 500
  - max_iters: 400000
- At iter 11,000 loss is ~4.0, trending down. Memory stable at ~54 GB free.
- Loss is noisier than previous runs due to small batch size. This is
  expected and acceptable — the trend is clearly downward.

### IMPORTANT LESSONS FROM THIS SESSION
- **Batch size 16 with 32K vocab and block=2048 crashes** from OOM.
  Batch=8 runs but memory is tight. Batch=4 is safe.
- **Learning rate must be scaled with batch size.** Batch=4 with
  lr=0.0003 (the old batch=16 rate) causes loss to plateau at 6.0.
  The fix: lr=0.00015 (sqrt scaling: lr *= sqrt(batch/old_batch)).
- **bfloat16 DOES work on MPS** despite train.py previously having a
  fallback that said it didn't. The fallback was removed.

### Saved checkpoints from prior runs
- `old_8_GB_corpus_pt/` — old 8 GB corpus run (iters 10K-160K)
- `unshuffled_corpus_pt/` — unshuffled 2.5 GB corpus run (iters 10K-50K)
- `doc_shuffled_batch4_pt/` — failed batch=4 lr=0.0003 run (iters 10K-20K)
- `../valuable_checkpoints/bpe_16L16H_old_corpus_iter160k_Excellent_from_8GB_corpus.pt`

### LaunchAgents running
- `com.ralph.training-monitor` — checks training, memory, thermal every
  5 min. Logs to `~/training_monitor_log.txt`. Plays Crystals alert if
  problems detected.
- `com.ralph.loss-plot-updater` — updates loss comparison plot every 15 min
  at `plots/train_val_loss_old_vs_new.png`.
- `com.ralph.storage-monitor` — weekly storage check (Sundays 9 AM).
- Note: thermal monitor was folded into training-monitor. The separate
  thermal-monitor LaunchAgent was unloaded.

### Corpus files on disk (txt_local/, gitignored)
- `corpus_books_shuffled_2026_04_18.txt` — CURRENT, in use (2.14 GB)
- `corpus_cleaned_shuffled_2026_04_18.txt` — paragraph-shuffled (not recommended)
- `corpus_final_2026_04_18.txt` — paragraph-filtered version
- `corpus_cleaned_2026_04_15.txt` — UNSHUFFLED (the one that caused problems)
- `corpus_of_gutenberg_novels_cleaned_shuffled_2026_03_03_B.txt` — original 8 GB corpus

### GitHub
- Repo: `dratman/small_transformer_research`
- Local clone: `../small_transformer_research/`
- Auth: via `gh auth login` (configured 2026-04-15)
- This repo (bpe_vs_char_model_comparison) is NOT pushed to GitHub.

## Recent Decisions

- **32K vocabulary** replaces 8K. Better information density per token,
  standard for 1B-class models. (2026-04-19)
- **bfloat16 precision** replaces float32. Halves memory usage with
  no quality impact. (2026-04-19)
- **Learning rate sqrt-scaling** for small batch sizes. lr=0.00015
  for batch=4 (scaled from lr=0.0003 at batch=16). (2026-04-19)
- **Document-level shuffling** replaces paragraph-level shuffling.
  See diary 085. (2026-04-18)
- **English-only corpus** — non-English books filtered out. (2026-04-18)
- **Dialect books removed** — 139 books with heavy dialect. (2026-04-18)
- **LARQL/gated-FFN** planned for the run after this one. (2026-04-18)

## Future Training Ideas (see memory for details)
- Stop-words-only training (syntax skeleton model)
- Bilingual French+English corpus
- LARQL/gated-FFN (SwiGLU) model — next after current run

## Open Questions
- Will 32K vocab produce better sample quality than 8K at equivalent loss?
- Should train.py mask loss at `<|endoftext|>` boundaries?
