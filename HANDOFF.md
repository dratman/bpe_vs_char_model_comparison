# Handoff Document

Last updated: 2026-05-20 08:18 by Claude Code Opus (Mac Studio session) —
M3 BPE training stopped early

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

### Character Model Training (Mac Studio) — STARTED 2026-05-09
- **Training in progress** — launched 2026-05-09 00:38 EDT, projected
  ~24 days (500K iters at 4.18 sec/iter measured on the speedtest)
- Launch script: `sh/train_char_uppercase_16L_1280.sh` (committed)
- Log: `terminal_logs/terminal_log_for_char_uppercase_16L_1280_2026_05_09_0038.txt`
- Corpus: `txt_local/corpus_high_quality_uppercase_2026_05_08.txt` (1.27 GB,
  3,979 books, 78-char vocab, **case preserved**, document-shuffled, seed=42)
- Architecture: **320M params** (n_layer=16, n_head=8, n_embd=1280, block_size=4096)
- Tokenizer: character-level, vocab=78
- batch_size=4, learning_rate=1.5e-4, max_iters=500,000, **bfloat16 on MPS**
- Output: `pt/char_uppercase_16L_1280.pt` (best) + `pt/char_uppercase_16L_1280_final.pt`
- Tokens-per-iter: 16,384 (vs char_high_quality's 8,192). Total at 500K iters:
  8.2 B chars (~6.4 epochs of the 1.27 B-char corpus, ~26 tokens/param)
- Speed-tested at 4.18 sec/iter steady-state, MFU ~3.5% on M3 Studio MPS
- Earlier on 2026-05-09 also speed-tested 12L/8H/1024 (151M, 1.92 sec/iter).
  Logs: `terminal_log_for_char_4096_speedtest_2026_05_08_2346.txt` and
  `terminal_log_for_char_4096_16L_1280_speedtest_2026_05_09_0006.txt`.
- **Things to monitor during the run**: train/val gap (should stay <0.05
  through epoch 4 ≈ iter 310K), loss trajectory (4.5→2.0 by iter ~5K
  expected), no NaN, disk space (25 intermediate checkpoints × 3.84 GB ≈
  96 GB; 256 GB free at launch).
- **Progress as of 2026-05-20 07:37** (11d 7h elapsed, ~46 % of the run):
  iter 231,800 / 500,000 (epoch 3.33). Best val 0.7774 at iter 230,000.
  Train/val gap remains tight (~0.015). LR has decayed from 1.50e-4 to
  9.14e-5 along the cosine schedule. Speed steady at 4.18 sec/iter.
  No incidents.
- **Sample at iter 184,000 (2026-05-19):** model produces credible
  19th-century literary prose with fewer invented words than at iter
  154K. Register more consistently held across each sample. See
  diary 093.

### BPE Model Training (M3 laptop) — STOPPED 2026-05-20
- **Training stopped early** 2026-05-20 08:15 EDT at iter 145,100 /
  220,000 (epoch 4.60). Val loss had plateaued for 13 consecutive
  evaluations past the iter-132K best (no improvement from iter 133K
  through 145K). Remaining 75K iters would have been ~5.5 more days of
  M3 compute past the point where val kept improving.
- Launch script: `sh/train_bpe_uppercase_16L_1280_b2.sh` (committed)
- Log on M3: `terminal_logs/terminal_log_for_bpe_uppercase_16L_1280_b2_2026_05_09_0926.txt`
  (mirror on Studio refreshed by `sh/plot_m3_bpe_snapshot.sh`)
- Corpus: same as the Studio char run
  (`txt_local/corpus_high_quality_uppercase_2026_05_08.txt`), rsynced to
  the M3's `txt_local/` at setup time. MD5 verified matching.
- Architecture: ~360M params (n_layer=16, n_head=8, n_embd=1280,
  block_size=4096, vocab=32,000 BPE)
- Tokenizer: BPE, vocab=32,000 (HuggingFace `tokenizers`)
- batch_size=2 (started at batch=4 but restarted at batch=2 the same
  morning when M3 memory pressure was tight), learning_rate=1.06e-4
  (sqrt-scaled from batch=4's 1.5e-4), max_iters=220,000,
  bfloat16 on MPS
- Speed: ~6.6 sec/iter steady-state; ran 10d 22h total
- **Best checkpoint: `pt/bpe_uppercase_16L_1280_b2.pt` on M3, val loss
  3.3657 at iter 132,000 (epoch 4.19), 2026-05-19 08:48.** Per-character
  loss ~0.77, slightly ahead of the Studio char run at matched corpus
  exposure. Tokens-per-iter: 8,192 BPE tokens (= ~36,864 chars at ~4.5
  chars/token). Reached ~5.4 corpus epochs of character budget when
  stopped (132K iters × 36,864 chars / 1.27 GB corpus).
- After iter 132K, 13 evals over iters 133K-145K never beat 3.3657.
  Mean val over that window: ~3.48; latest (iter 145K) was 3.4433.
  Train loss steady at 3.20-3.30, so train/val gap widened from ~0.15
  (early run) to 0.25-0.40 — classic overfitting onset.
- 28 intermediate checkpoints on M3 (iter 5K, 10K, ..., 145K at every
  5K, ~112 GB total at 4 GB each). See TODO for cleanup decision.
- **Sample at iter 95K (2026-05-17):** the model produces fluent
  19th-century-style prose with multi-paragraph plot coherence.
  See diary 093. Final samples at iter 145K in the tail of the
  training log are similarly coherent.
- **M3 SSH note** (still relevant): when the M3's display is asleep,
  network sleep engages and SSH from the Studio times out. Wake the
  M3 locally if you need to inspect or sample from it. Since 2026-05-20
  the snapshot script (`sh/plot_m3_bpe_snapshot.sh`) resolves the M3
  by mDNS name (`MacBookProM3Max.local`) so it works whether the M3
  is on Ethernet (was 192.168.1.177) or wifi (was 192.168.1.185 on
  2026-05-20).

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

- **Working-tree cleanup (2026-05-08).** Studio's working tree had
  accumulated a long list of untracked files and ~240 GB of preserved
  checkpoint dirs. Resolved in four commits (`2d65d47`, `de2d593`,
  `ef5494d`, `60cd051`):
  - `.gitignore` extended: `*_pt/` (covers `old_8_GB_corpus_pt/`,
    `unshuffled_corpus_pt/`, `doc_shuffled_batch4_pt/`) and intercom
    runtime state (`channel.txt`, `cursor_*.pos`, `instance_*.id`).
  - Committed: 4 py utility scripts, `sh/claude.sh` (session-logging
    wrapper), intercom code, `stop_words.txt`, 7 doc/ chat transcripts
    and analyses, 12 prior-session terminal logs, plus a small
    `sh/sample_imitator_L10_compare.sh` tweak.
  - Deleted: `word_counts_clean_corpus_OLD.txt` and a misplaced
    near-duplicate `terminal_log_for_char_full_corpus_*.txt` that was
    sitting under `doc/`.
  - Moved into `txt_local/`: `reduced_corpus.txt` (1.59 GB),
    `word_counts_for_corpus_books_shuffled_2026_04_18.txt`,
    `stop_words_from_corpus_books_shuffled_2026_04_18.txt`.
  - Renamed: `doc/imitator_training_begun_20236_04_28.txt.txt` →
    `doc/imitator_training_begun_2026_04_28.txt` (typo fix).
  - `git status` is now clean.

- **`comments_on_prior_diaries/` folder added (2026-05-08, commit
  `72738fd`).** Sibling to `diary/`. Holds second-pass commentary on
  existing diary entries, written by a Claude instance different from
  the one that authored the original entry. Naming convention:
  `comments_on_NNN.md` where NNN is the diary number being commented
  on. Header records the date and which Claude instance wrote the
  comments. First entry: `comments_on_090.md` (Claude Code's response
  to browser-Claude's diary 090 on tokenization as a learned function).

- **Trained-model inventory generated (2026-05-08, commit `247d64c`).**
  One-shot snapshot of every `.pt` file under `..` (one row per
  training run, picking the best/final/highest-iter representative).
  179 runs total, ~637 GB of weights spread across ~38 sibling
  directories. Each row has: directory, run name, representative file,
  size, mtime, architecture (n_layer/n_head/n_embd/block_size/vocab_size),
  tokenizer type (extracted from each run's `_meta.pkl`), iter,
  best val loss, number of checkpoints in the run, and a generated
  text sample. Files in `doc/`:
  - `model_inventory_2026_05_08.md` — markdown source
  - `model_inventory_2026_05_08.csv` — for Numbers
  - `model_inventory_2026_05_08.pdf` — landscape A4
  - `model_inventory_2026_05_08_sortable.html` — click-to-sort in browser
  - `model_inventory_2026_05_08_samples.txt` — full untruncated samples
  Sample column is populated for 11 rows (3 in this project's `pt/`,
  3 in `*_pt/` saved corpora, 4 in `../valuable_checkpoints/`, 1 in
  `../valuable_checkpoints/B_9GB/` = Model B). The other 168 rows show
  `N/A` because their training code is in sibling repos (tiny_transformer,
  nanogpt_*, etc.) whose `model.py` / tokenizer setup diverges from this
  repo's `py/sample.py` and would need per-codebase samplers to evaluate.
  Sample params: prompt=`"the old man"`, max_tokens=200, temperature=0.8,
  seed=42. Generation scripts are in `/tmp/` (not committed); the
  inventory file artifacts are committed.

- **Corpus rebuilt with case preserved (2026-05-09, commits `a9ce02e`,
  `ac81d1e`).** New corpus
  `txt_local/corpus_high_quality_uppercase_2026_05_08.txt` (1.27 GB,
  3,979 books, 78-char vocab) is the case-preserved equivalent of
  `corpus_high_quality_2026_04_26.txt` (1.42 GB, 4,430 books, 52-char
  vocab, lowercase). Lost ~10% of books because the matcher's
  conservative policy drops books that landed in PG-boilerplate-only
  signatures. Files added at repo root:
  `book_index_to_filename.json`, `corpus_haiku_keep.txt`,
  `corpus_manifest_2026_05_08.tsv`, `unmatched_books.txt`. The new
  corpus is gitignored under `txt_local/`.
- **Build-script fixes (2026-05-09, commit `a9ce02e`).** Patched five
  scripts in `py/` that participate in the corpus pipeline:
  - `rebuild_corpus.py`: added `--preserve_case` flag, transliteration
    map for non-decomposable Latin chars (ß, þ, ð, æ, œ, ø, ł, plus
    uppercase variants — previously silently dropped, e.g. straße →
    strae), `i` added to English function-word list, case-insensitive
    is_english, `--manifest` output for auditability.
  - `clean_and_combine_corpus.py`: added `--preserve_case`, character
    set aligned with `rebuild_corpus.py` (previously allowed `/`,
    creating vocab mismatch).
  - `filter_corpus.py`: removed dead `KEEP_SHELVES` block.
  - `scan_corpus_quality.py`: docstring fixed (10-min not 10-sec timeout).
  - `match_book_samples.py` **NEW**: recovers Haiku-decisions
    index↔filename mapping by Aho-Corasick fuzzy match of sample
    excerpts against source files. Strips PG headers/footers and
    demotes multi-match files. Output: `corpus_haiku_keep.txt`.
- **Training-pipeline tooling improvements (2026-05-09, commit
  `be52633`).**
  - `py/train.py` startup banner now prints every CLI arg plus
    hardcoded weight_decay/betas/grad_clip values, so any training
    log is self-describing regardless of launch mechanism.
  - `py/sample.py` auto-detects case-preserved tokenizers. When
    detected: skip the default prompt-lowercasing AND skip the
    post-hoc `capitalize_sentences` pass (which would overwrite the
    model's correct case). `--no_lowercase` flag still honored as
    manual override. Header line reports "case-preserved" status.
- **`book_quality_decisions.json` indexing scheme** (a non-obvious thing
  to know): the Haiku quality-filter step that produced this file is
  NOT in the repository. Its keys are integer indices (`'0'`-`'6197'`)
  into `doc/all_book_samples.txt`. Mapping back to source filenames
  required `py/match_book_samples.py` (text matching). 96.3% of KEEP
  verdicts (4,282 of 4,430) had clean 1-to-1 matches; 199 KEEP
  verdicts were demoted as multi-match collisions and 148 had no
  matching source file (file deleted or sample defective).

- **`del text` fix in `train.py` (2026-05-09, commit `974cc43`).** `main()`
  loads the full corpus into `text = f.read()` and uses it only through
  the data-prep step. Previously the variable stayed in scope for the
  entire (multi-day) run, holding ~1.27 GB of Python heap that the
  compressor and swap had to manage. Added `del text` right after the
  data-prep conditional. Applies to every future run; the two in-flight
  runs (Studio char and M3 BPE) have their own private copies and
  cannot benefit unless restarted.

- **SSH key authentication Studio → M3 (2026-05-09).** The Studio's
  `~/.ssh/id_ed25519.pub` is in the M3's `~/.ssh/authorized_keys`. The
  M3 is at `192.168.1.177` on the Ethernet LAN. From the Studio, run
  `ssh RalphDratman@192.168.1.177 "..."` for any read-only inspection
  command, or `rsync` for moving files. The M3 may appear unreachable
  if its display is asleep and macOS network sleep has engaged — wake
  it up locally if so. `dratman@github.com` SSH auth is also configured
  on the M3 (used to clone the repo there at setup time).

- **Plot snapshot infrastructure (2026-05-09, commits `e331c30`,
  `1abaf52`, `2858a33`).** Two launcher scripts in `sh/` produce
  refreshed loss plots:
  - `sh/plot_m3_bpe_snapshot.sh` — rsyncs the M3 BPE training log to
    Studio's `terminal_logs/`, then plots. Output:
    `plots/bpe_uppercase_16L_1280_b2_loss.png`.
  - `sh/plot_studio_char_snapshot.sh` — plots the Studio char log
    directly (no rsync). Output:
    `plots/char_uppercase_16L_1280_loss.png`.
  Both call `py/plot_current_run.py` with `--log` and `--out`. The
  plotting script's smoothing window scales with data length so the
  smoothed curve is visible at any stage of training. Plots are
  gitignored under `plots/`; PNGs are named to mirror their checkpoint
  (`pt/X.pt` ↔ `plots/X_loss.png`).

- **training-monitor false-alarm fix (2026-05-15).** `~/bin/training_monitor.py`'s
  `check_memory()` used to trigger an alert whenever `Pages free` from
  `vm_stat` fell below 2 GB. On a hot-running Mac Studio with 192 GB
  unified memory, "Pages free" is almost always near zero because macOS
  fills every spare page with file cache and other reclaimable uses.
  Changed the threshold to `available_gb < 4` (where available = free
  + inactive, the actually-reclaimable memory). This is the same notion
  Activity Monitor uses for its "Memory Pressure" indicator. The
  script lives at `~/bin/training_monitor.py` outside the repository
  (consistent with the storage-monitor convention noted in CLAUDE.md);
  the fix is on the Studio only. If the M3 ever gets the same monitor,
  copy the script over.

- **Plot auto-refresh wired up (2026-05-16, commits `042fd2c`, `3b17645`,
  `a08da01`, `a7cf741`).** The pre-existing `com.ralph.loss-plot-updater`
  LaunchAgent runs every 15 minutes; its script `~/bin/update_plots.sh`
  now invokes `sh/plot_m3_bpe_snapshot.sh` and
  `sh/plot_studio_char_snapshot.sh` plus the historical comparison plot.
  Plots are written to:
    plots/char_uppercase_16L_1280_loss.png
    plots/bpe_uppercase_16L_1280_b2_loss.png
  Each plot shows a "Refreshed YYYY-MM-DD HH:MM" timestamp (lower-right)
  and an "Iteration X of Y -- Epoch Z" label (lower-left). The M3 BPE
  snapshot script tolerates rsync failure (M3 asleep) and falls back to
  the cached log so the plot is still refreshed.

- **Sample scripts for both runs (2026-05-16/17, commits `de5dd63`,
  `e6f6a8b`).**
  - `sh/sample_char_uppercase_16L_1280.sh` — runs on the Studio against
    the local char checkpoint.
  - `sh/sample_bpe_uppercase_16L_1280_b2.sh` — **runs on the Studio**
    too: rsyncs the M3's best-val checkpoint and tokenizer metadata
    over (skipped if unchanged; mtime preserved), then samples
    locally. Originally ran on the M3 but M3 memory was tight under
    training load. Rsync skips the ~4.3 GB transfer when the M3 has
    not saved a new checkpoint.
  Both scripts pass `$@` through to `py/sample.py`, so prompt /
  temperature / num_samples / max_tokens can be overridden per call.
  Both tee output to a timestamped log under `terminal_logs/`.

- **`py/sample.py` case-preservation detection (2026-05-09, commit
  `be52633`).** Auto-detects whether the loaded tokenizer is case-
  preserved (any uppercase letter in itos for char, any in vocab keys
  for BPE) and disables prompt-lowercasing + capitalize_sentences
  accordingly. Both current models trigger this path.

- **Diary 092 (2026-05-15, commit `e5c4ac7`).** Two-machine
  operational workflow, macOS memory accounting (wired/available/swap),
  the `del text` fix, plot infrastructure.

- **Diary 093 (2026-05-17, commit `55e585f`).** Tokenization fixes the
  level at which the model improvises. Char-level models improvise
  *words* because their atoms are letters; BPE models improvise
  *sentences* because their atoms are subwords/words. The choice of
  tokenization is the choice of where the model's freedom and
  imperfection should live. Connects to diaries 014/015/035 (the
  layer-machinery a char model spends on building word recognition)
  and 074 (topological framing).

- **Pending: requesting Claude.ai and ChatGPT exports (2026-05-19).**
  Ralph plans to request data exports from both services so a future
  fine-tuning or specialized-model experiment has a real conversation
  corpus to work with. Diary entries alone are 358 KB cleaned — far
  too small. Once the exports arrive (typically 1–2 days), the next
  session should:
    - Inspect each zip's structure
    - Write JSON-to-text converter into the 78-character vocabulary
    - Report total size and decide whether to mix into a future
      fine-tuning pass or skip the idea.
  Catastrophic-forgetting mitigation discussed: rehearsal (mix
  Gutenberg with the new data at maybe 4:1) + low learning rate
  (~10x to 100x below the original) + short fine-tuning run.
  LoRA-style adapters would be the cleaner approach if we ever want
  guaranteed no-forgetting, but require code changes.

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
- **macOS background-launched processes inherit SIGINT=SIG_IGN.** The
  `train.sh` wrapper launches python in the background (`python ... &`)
  and bash/zsh job-control semantics set SIGINT to ignored for
  background jobs. So `kill -INT $PYTHON_PID` from outside the wrapper is silently
  dropped — the only signal that worked when stopping the M3 BPE run
  on 2026-05-20 was SIGTERM (`kill -TERM`). train.py has no SIGTERM
  handler, so SIGTERM exits immediately; checkpoint state is safe
  because the save policy only writes on val improvement.

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

### While trainings are in progress
- [x] **DONE 2026-05-20.** M3 BPE run stopped early at iter 145,100.
  Best (iter 132K, val 3.3657) preserved as
  `pt/bpe_uppercase_16L_1280_b2.pt`.
- [ ] Watch Studio char train/val divergence around epoch 4 (~iter 310K).
  Currently at epoch 3.33, gap still tight (~0.015). If the gap stays
  small all the way through, consider extending max_iters; if it widens,
  plan to stop.
- [ ] Decide what to do with intermediate checkpoints from the Studio
  char run (~96 GB; clean periodically or keep all 25 for
  layer-stability analysis per diary 080).
- [ ] **Decide what to do with M3 BPE intermediate checkpoints**
  (~112 GB across 28 files, iter 5K-145K at every 5K). The best
  (iter 132K) is already separately saved as
  `pt/bpe_uppercase_16L_1280_b2.pt`. Layer-stability analysis would
  benefit from keeping early checkpoints; routine cleanup would free
  ~100 GB. Either rsync them to `../valuable_checkpoints/` first or
  delete in place on the M3.

### Diary + conversation corpus (planned post-training)
- [ ] **Request Claude.ai data export** (Settings → Privacy → Request
  data export). Expect email with download link in 1–2 days.
- [ ] **Request ChatGPT data export** (Settings → Data Controls →
  Export Data). Same timing.
- [ ] When exports arrive: inspect JSON structure, write converter to
  the 78-character vocabulary, report total size, decide whether to
  do a fine-tuning pass. See HANDOFF "Pending" bullet above.

### Analysis on completed char_high_quality.pt (still relevant; lower priority)
- [ ] Run per-position prediction analysis (War and Peace passage) on
  `pt/char_high_quality.pt` — compare to the BPE model's 27% and Model B's 77%
- [ ] Run free generation from the War and Peace prompt — compare to
  earlier models that collapsed into junk
- [ ] Try the "appalpittidax" copying analysis on `pt/char_high_quality.pt`
  (and later on the case-preserved model once trained)
- [ ] Run compare and rollout on the L10 imitator (`pt/imitator_L10_full.pt`,
  still not done since training in 2026-04-27)

### Corpus options if/when we want to scale past ~500M params
- [ ] Recover the 449 books lost in the matcher via multi-excerpt
  disambiguation (lifts corpus 10–15%)
- [ ] Add Wikipedia biographies via `clean_and_combine_corpus.py` (already
  patched for case-preservation; would roughly double the corpus to ~2.5 GB
  and unlock 800M-1B param models without data starvation)

### Experiments to try
- [ ] Imitator rollout with a stronger base model (download Llama 3B or similar)
- [ ] Whole-word tokenizer implementation (diary 089 — 100K word vocab)
- [ ] Spectral analysis of layer contributions (holographic framing, diary 087)
- [ ] Imitator with downstream KL loss — fix generalization gap
- [ ] Imitator at different split layers (4, 6, 12, 14) for comparison
- [ ] Retokenization: cluster predicted vectors into discrete mid-layer tokens
