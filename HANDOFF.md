# Handoff Document

Last updated: 2026-04-26 09:23 by Claude Code Opus (Mac Studio session)

## Current State

### BPE Model Training (Mac Studio)
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
- At iter ~176,000, val loss ~3.57. Sample quality is reasonable prose.

### Imitator Experiment (NEW — 2026-04-25/26)
- **Concept:** Train a small transformer to predict the next residual-stream
  vector at layer 8 of the frozen BPE model. Feed predicted vectors through
  the frozen model's back half (layers 8-15) to decode into English.
- **Origin:** Ralph's idea, developed in conversation with regular Claude,
  then implemented in Claude Code.
- **Files created:**
  - `py/imitator_model.py` — 21M-param vector-to-vector transformer
    (d_model=512, n_layer=6, n_head=8, no embeddings)
  - `py/small_model_split.py` — splits frozen GPT at layer N
  - `py/train_imitator.py` — training loop (cosine + MSE loss)
  - `py/sample_imitator.py` — compare, rollout, and stats evaluation modes
  - `sh/train_imitator.sh` — logging wrapper
  - `sh/train_imitator_L8_first_run.sh` — first run launch script
  - `sh/sample_imitator_compare.sh` — compare mode launch script
- **First run completed on M3 Pro (64 GB):**
  - 5000 iterations, ~70 minutes
  - Frozen model: `pt/bpe_16L16H_iter170000.pt` (875M params, bfloat16)
  - Best val loss: 0.1277, val cos_sim: 0.9484 (iter 4400)
  - Checkpoints: `pt/imitator_L8.pt` (best), `pt/imitator_L8_final.pt`
  - Log: `terminal_logs/terminal_log_for_imitator_L8_2026_04_25_2346.txt`
  - Token cache: `txt_local/corpus_tokens_32k.pt` (speeds up reruns)
- **Key result:** Vector-level cosine similarity is high (0.948) but
  token-level match is poor (14-20% top-1 match with frozen model).
  Decoded output collapses to high-frequency tokens (commas, "the", "of").
  The 512-dim bottleneck and/or cosine loss may discard the subtle features
  the back half needs for sharp token decisions.
- **Second run (2048-dim, stopped early):** Started a 302M-param imitator
  at full d_model=2048 (no projection bottleneck). Started at cos_sim=0.84
  due to residual connections acting as identity. Stopped after ~90 iters
  to explore other questions. Launch script: `sh/train_imitator_L8_full_dim.sh`
- **Key insight from this session:** Free-running generation produces much
  more coherent text than forced next-token prediction. The frozen model
  scores 27% top-1 on a War and Peace passage, but the actual token is in
  the top 5 about 65% of the time (median rank 3). The model knows English
  well; it just cannot predict the exact word an author chose. This reframes
  the imitator experiment — rollout mode (free generation in vector space)
  may be more informative than compare mode (forced prediction).
- **Evaluation bug found and fixed:** The original compare script had a
  causal context error — the back half was missing position 0's context.
  Fixed by decoding the full sequence through the back half.
- **Next steps to consider:**
  1. Imitator rollout mode — free generation in vector space, decoded through back half
  2. Full-dim (2048) imitator training to completion
  3. Downstream KL loss (optimize token distribution match, not vector match)
  4. Different split layers (2, 4, 12, 14)
  5. Retokenization: cluster predicted vectors into discrete mid-layer tokens
  6. Try the imitator experiment on a stronger open-source base model (Llama 3B)
- **M3 repo location:** `/Volumes/RalphDratman/0-Home-Working-on-M3-Pro/bpe_vs_char_model_comparison/`
  (cloned from Mac Studio, contains all source + checkpoint + corpus)
- **Additional shell scripts on M3:**
  - `sh/sample_imitator_compare.sh` — compare mode evaluation
  - `sh/sample_imitator_war_and_peace.sh` — compare on specific Tolstoy passage
  - `sh/sample_frozen_war_and_peace.sh` — frozen model baseline on Tolstoy
  - `sh/sample_frozen_detailed.sh` — per-position rank/probability analysis
  - `sh/sample_frozen_continue.sh` — free generation from Tolstoy prompt
  - `sh/train_imitator_L8_full_dim.sh` — 302M imitator at full dimension

### IMPORTANT LESSONS FROM THIS SESSION
- **Batch size 16 with 32K vocab and block=2048 crashes** from OOM.
  Batch=8 runs but memory is tight. Batch=4 is safe.
- **Learning rate must be scaled with batch size.** Batch=4 with
  lr=0.0003 (the old batch=16 rate) causes loss to plateau at 6.0.
  The fix: lr=0.00015 (sqrt scaling: lr *= sqrt(batch/old_batch)).
- **bfloat16 DOES work on MPS** despite train.py previously having a
  fallback that said it didn't. The fallback was removed.
- **Cannot run code on M3 from Mac Studio.** Shell commands execute on
  the Mac Studio even when pointed at the share. Must give Ralph shell
  scripts to run on the M3 directly.
- **High cosine similarity does not mean good decoded output.** 94.8%
  cosine similarity in 2048-dim space still produces 80%+ wrong tokens
  when decoded through the frozen model's back half.

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

- **Imitator experiment started.** Predict mid-layer residual stream
  vectors, decode through frozen model's back half. (2026-04-25)
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
- Can downstream KL loss or a larger imitator close the vector→token gap?
- What do imitators at different split layers reveal about layer-by-layer processing?
- Can the predicted vectors be retokenized into a discrete mid-layer vocabulary?
