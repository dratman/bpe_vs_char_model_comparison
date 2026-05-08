# Handoff Document

Last updated: 2026-05-08 by Claude Code Opus (Mac Studio session)

## Current State

### Character Model Training (M3 Pro) — COMPLETED 2026-05-07
- **Training complete** — launched 2026-04-26 20:44, finished 2026-05-07 23:58
  (11 days 3 hours)
- Script: `sh/train_char_high_quality.sh` (lived on M3; brought into Studio
  pt/ and terminal_logs/ on 2026-05-08 from `../bpe_vs_char_model_comparison_M3_2026_05_08/`)
- Logs:
  - `terminal_logs/terminal_log_for_char_high_quality_2026_04_26_2044.txt` (initial launch)
  - `terminal_logs/terminal_log_from_2026-04-26_through_2026_05_08.txt` (full run, ends with "Training complete!")
- Corpus: `txt_local/corpus_high_quality_2026_04_26.txt` (1.42 GB, 4,430 books)
  - Filtered from the 2.05 GB corpus using Claude Haiku API to judge quality
  - Removed archaic language, cookbooks, scientific texts, index entries, etc.
  - 1,768 books removed (29%), 4,430 kept
- Architecture: 152M params, n_layer=12, n_head=8, n_embd=1024, block_size=512
- Tokenizer: character-level, vocab=52
- batch_size=16, learning_rate=3e-4, max_iters=500,000
- Final: iter 500,000 (epoch 3.21), best validation loss **0.8225**
- Checkpoints in `pt/` (copied from M3 2026-05-08):
  - `char_high_quality.pt`
  - `char_high_quality_final.pt`
  - `char_high_quality_iter480000.pt`
  - `char_high_quality_meta.pkl`
- Final samples in the run log are coherent multi-sentence 19th-century-style
  prose (see tail of `terminal_log_from_2026-04-26_through_2026_05_08.txt`)

### BPE Model Training (Mac Studio) — STOPPED
- **Training stopped** 2026-04-27 at iter ~235,000 (epoch 4.48)
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
- Final: iter ~235,000, best val loss 3.35 (iter 229,000). Val loss
  was plateau-ing with only marginal improvement.
- Best checkpoint: `pt/bpe_16L16H.pt` (val loss 3.35)
- Also kept: `pt/bpe_16L16H_iter170000.pt` (imitator experiments)
- Saved to: `../valuable_checkpoints/bpe_16L16H_32k_vocab_best_val3.35_iter229000.pt`
- Intermediate checkpoints deleted (freed ~215 GB)

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
- **Layer-10 imitator completed on Mac Studio (2026-04-27):**
  - 302M params, d_model=2048, split_layer=10, 5000 iters
  - Val cos_sim: 0.932 (lower than layer-8's 0.948 — harder to predict)
  - Best val loss: 0.359
  - Checkpoint: `pt/imitator_L10_full.pt`
  - Compare and rollout not yet run
- **Copying mechanism discovered (diary 088, 2026-04-27):**
  - Character model (152M, iter 20K) invented "appalpittidax" and
    reproduced it exactly 86 characters later
  - Layer 9 Head 3 sends 46-92% attention to first occurrence
  - Logit lens: correct character appears at layer 9 (jumps from ~5%
    to ~80-95% probability in one layer)
  - Copying is a two-stage process: L9 identifies region and reads
    ahead, L11 does precise character matching
- **Per-position prediction analysis (2026-04-26):**
  - BPE model (875M, iter 170K): 27% top-1, median rank 3, 65% in top 5
  - Character model B (808M, iter 99K): 77% top-1, median rank 1, 94% in top 5
  - Key insight: free-running generation is much more coherent than forced
    prediction because the model always continues from its own output
- **Corpus quality filtering (2026-04-26):**
  - Used Claude Haiku API to judge 6,198 book samples
  - Removed 1,768 books (archaic, recipes, scientific, indexes, etc.)
  - New corpus: `txt_local/corpus_high_quality_2026_04_26.txt` (1.42 GB, 4,430 books)
  - Quality decisions saved in `doc/book_quality_decisions.json`
- **War and Peace character model (2026-04-26):**
  - 19M params, trained on War and Peace only (3.2M chars)
  - Reached val loss 1.157 at iter 6000, then started overfitting
  - Free generation collapsed quickly — too small and too little data
  - Checkpoint on M3: `pt/char_war_and_peace.pt`
- **Character models on Mac Studio:**
  - `../valuable_checkpoints/B_9GB/gutenberg_corpus_MODERN_CLEAN_continuous.pt`
    — 808M, val loss 0.832, iter 53500 (Model B)
  - `../study_corpus_and_training_2a/pt/gutenberg_corpus_MODERN_CLEAN_continuous.pt`
    — 808M, val loss 0.779, iter 99000 (best character model, same arch as B)
  - `../study_corpus_and_training_2c_char/pt/...continuous.pt`
    — 51M (n_embd=512), val loss 0.826, iter 1.4M
  - Copies on M3 as `pt/char_model_B_best.pt` and `pt/char_model_2c_clean.pt`
- **M3 work consolidated to Studio (2026-05-08).** A snapshot of the M3
  working folder was placed at `../bpe_vs_char_model_comparison_M3_2026_05_08/`
  and merged into Studio in three commits (`a3d7e3f`, `bb15329`, `354d67a`):
  - char_high_quality `.pt` checkpoints copied into `pt/`
  - 29 unique M3 terminal logs imported into `terminal_logs/`
  - `RUN_ON_M3.txt` (imitator setup notes) preserved at repo root
  - `claude_code_sessions/SESSION_2026_04_20_2245.raw.txt` preserved on
    disk (gitignored, like all session logs now)
  - HANDOFF.md updated to mark training complete
  - All `sh/sample_*` and `sh/train_*` scripts that previously existed
    only on M3 are now committed to this repo (happened over April–May)
  - The M3 snapshot folder was then deleted (~14 GB freed)
- **M3 machine itself** still holds a working clone at
  `/Volumes/RalphDratman/0-Home-Working-on-M3-Pro/bpe_vs_char_model_comparison/`,
  not currently active.

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
- Sibling repo for diary entries and shared notes: `dratman/small_transformer_research`
  (local clone at `../small_transformer_research/`)
- This repo: `dratman/bpe_vs_char_model_comparison`
  (origin = https://github.com/dratman/bpe_vs_char_model_comparison.git, master pushed)
- Auth: via `gh auth login` (configured 2026-04-15)

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

## TODO

Review this list at the start of every session. Mark items DONE when complete.

### When training pauses
- [ ] Consolidate this repo into `small_transformer_research` as a subdirectory.
  See memory file `project_repo_consolidation.md` for the plan.

### Now that char_high_quality training is done (2026-05-07) — unblocked
- [ ] Run per-position prediction analysis (War and Peace passage) on
  `pt/char_high_quality.pt` — compare to the BPE model's 27% and Model B's 77%
- [ ] Run free generation from the War and Peace prompt — compare to
  earlier models that collapsed into junk
- [ ] Try the "appalpittidax" copying analysis on the new model
- [ ] Run compare and rollout on the L10 imitator (`pt/imitator_L10_full.pt`,
  still not done since training in 2026-04-27)

### Experiments to try
- [ ] Imitator rollout with a stronger base model (download Llama 3B or similar)
- [ ] Whole-word tokenizer implementation (diary 089 — 100K word vocab)
- [ ] Spectral analysis of layer contributions (holographic framing, diary 087)
- [ ] Imitator with downstream KL loss — fix generalization gap
- [ ] Imitator at different split layers (4, 6, 12, 14) for comparison
- [ ] Retokenization: cluster predicted vectors into discrete mid-layer tokens
