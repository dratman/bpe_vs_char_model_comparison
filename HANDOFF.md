# Handoff Document

Last updated: 2026-06-06 by Claude Code Opus 4.8 (Linux/CUDA workstation,
SSH-from-Mac session) — repo housekeeping only, no model work. (1) Finished
the GitHub operation a parallel session left open: fast-forward-merged
branch `memorization-probe-and-handoff-fix` into master (HEAD now
`fe4b766`), pushed to origin, then deleted that branch (local + remote)
and removed the working-tree coordination note `temp-message.txt`. master
now contains `py/memorization_probe.py` + `sh/memorization_probe_bpe.sh`.
(2) RESOLVED the "test training" question. Mid-session Ralph pushed 3
commits to origin/master that I integrated by rebasing this handoff commit
on top (clean, no conflict): `159d0d5` (diary 095), `4360cac` (train.py
`--no_bias` flag + two no-GELU trial scripts), `410b79a` (matched-LR trial
script). The "test training" is the **no-GELU MLP ablation** on the Studio
char model (16L/8H/1280, block 4096, same `corpus_high_quality_uppercase`
corpus). With GELU off, each MLP collapses to a single rank-1280 linear map
(params unchanged at 320M). A first 10K-iter trial
(`sh/train_char_uppercase_16L_1280_no_gelu_trial.sh`) HAS ALREADY RUN on
Studio MPS (~12h) — per the matched-LR script's own notes it reached val
~1.84 at iter 8K vs baseline ~1.06, BUT that comparison is confounded by a
cosine-LR-schedule mismatch (trial max_iters=10K → LR decayed fast; baseline
used 500K → near-constant LR over its first 10K). Two follow-up scripts are
now committed but NOT confirmed launched: `..._no_gelu_matched_LR_trial.sh`
(max_iters=500K so LR matches baseline; stop after ~10K via SIGTERM, or let
it run ~24d for a full val floor) and `..._no_gelu_no_bias_trial.sh` (also
sets `--no_bias`, LLaMA/PaLM convention, ~226K fewer params). **OPEN for
next instance:** is the matched-LR run currently live on the Studio? Git
can't tell — SSH into the Studio and check `pgrep -af train`,
`ls -laht pt/*no_gelu*`, `ls -laht terminal_logs | head`. To stop a Studio
run use SIGTERM (not SIGINT — train.sh backgrounds python which inherits
SIGINT=SIG_IGN). **Studio address (Ralph-confirmed 2026-06-06):
`studio.local` = `192.168.1.233`.** From this Linux box: it IS routable
over the wifi LAN (this box is `192.168.1.224/24` on `wlp4s0`, alongside
the `10.99.99.x` link to the Mac), and its SSH host key is now cached —
BUT this box is not yet authorized: SSH returns `Permission denied
(publickey)`. To enable passwordless access, authorize this box's key once
with `ssh-copy-id RalphDratman@192.168.1.233` (run from `owner@linux-a6000`;
pubkey `~/.ssh/id_ed25519.pub` = "ralph.dratman@gmail.com (linux-a6000)";
enter the Studio password once). Mac username assumed `RalphDratman` (per
`/Users/RalphDratman`); confirm. NOTE mDNS `.local` does NOT resolve on this
Linux box (no avahi/nss-mdns daemon) — use the IP `192.168.1.233`, not the
name. Also still: the memorization probe is committed but **ready to run,
not yet run**.
(3) Side note (not project state): a separate on-screen claude session had
hung its TUI on a background gnome-terminal tab — looked like a "locked
keyboard"; resolved by `kill`-ing that claude PID and resuming with
`claude --continue`. Hardware was fine throughout.

Prior update: 2026-06-06 by Claude Code Opus 4.8 (Linux/CUDA workstation
session) — text/code-only work this session (no model weights or corpora
on the Linux box): implemented `py/memorization_probe.py` + wrapper
`sh/memorization_probe_bpe.sh` (overtraining experiment, ready to run on
the Studio/M3 — see the BPE-resumed section); fixed a stale, self-
contradictory line in the char cross-comparison bullet that claimed
diary 094 didn't incorporate the BPE-resumed best (it does, since commit
`abaa52c`). Still pending and NOT doable from Linux: char checkpoint
backup to Expansion, M3 wrapper-process cleanup (PIDs 26584/26585).

Earlier update: 2026-06-02 by Claude Code Opus 4.7 (Mac Studio session)
— **Both trainings are now COMPLETE.** Studio char: completed
2026-06-02 08:54 EDT at iter 500,000 (epoch 7.19), best val 0.7152
per-char, 24d 8h wall time. M3 BPE-resumed: completed 2026-06-01
15:34 EDT at iter 220,000 (epoch 6.99), best val 3.2652 BPE-loss
(~0.725 per-char) at iter 168K, no improvement in last 52K iters.
Char wins the per-char comparison by 0.010 (1.4 %) — a much
narrower margin than the original BPE-stop number (0.748) would
have suggested. Diary 094 has the analysis (revised to use the
resumed-run best, not the original-run number). Backup of final +
best char checkpoints to Expansion still pending (TCC permission
for Terminal.app — fix or run from a shell with FDA). M3 cleanup
pending: the resumed-run wrapper scripts (PID 26584, 26585) are
still alive on the M3 holding `tail -f` on a static log; needs
`kill -TERM` to clean up.

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

### Character Model Training (Mac Studio) — COMPLETED 2026-06-02
- **Training complete** — launched 2026-05-09 00:38 EDT, finished
  2026-06-02 08:54 EDT (24 days 8 hours 14 minutes wall time).
  Final iter: 500,000 (epoch 7.19). **Best validation loss: 0.7152**
  (≈ 48.9 % per-character probability of correct prediction).
  Per-step train loss at the end was bouncing in the 0.50-0.60 range
  on single-batch eval; the noise-floor of single-batch loss isn't
  meaningful for the model's actual quality, which is captured by val.
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
- **Progress as of 2026-05-20 09:40** (11d 9h elapsed, ~46 % of the run):
  iter 232,000 / 500,000 (epoch 3.34). **Best val 0.7720 at iter 226,000**
  (full-history scan of the log; latest eval at iter 232K showed val
  0.7881, not a new best). Train/val gap was 0.007 at the best-val
  state and has widened to ~0.028 at iter 232K, still well within
  the <0.05 target through epoch 4. LR has decayed from 1.50e-4 to
  9.06e-5 along the cosine schedule. Speed steady at 4.18 sec/iter.
  No incidents.
- **Progress as of 2026-05-23 13:13** (14d 12h 33m elapsed, ~59.6 % of
  the run): iter 298,200 / 500,000 (epoch 4.29). **Best val 0.7533 at
  iter 288,000** (set 2026-05-23 01:19, ~12 h before this update). Five
  new best-val checkpoints between iter 232K and 288K: 0.7689 (234K),
  0.7578 (244K), 0.7558 (270K), 0.7556 (280K), 0.7533 (288K). Train/val
  gap at the latest eval (iter 298K) was 0.058 — a single-eval reading
  above the <0.05 watch-threshold, but **mean gap over iters 260K-298K
  is ~0.031**, still under threshold. Four individual evals in this
  window crossed 0.05 (272K, 276K, 286K, 298K); eval-to-eval variance
  is large (20-batch evals), so the mean trend matters more than any
  single reading. Speed steady at 4.15 sec/iter. ETA ~2026-06-02
  (~10 more days). No incidents. **Decision: keep running.** If no
  new best-val by iter ~320K AND mean gap stays >0.04, that is the
  signal to consider stopping.
- **Progress as of 2026-05-26 10:32** (17d 9h 52m elapsed, ~71.5 % of
  the run): iter 357,500 / 500,000 (epoch 5.14). **Best val 0.7284 at
  iter 342,000** (set 2026-05-25 16:27, ~18 h before this update).
  Three new best-val checkpoints since the previous update: 0.7467
  (306K), 0.7342 (320K), 0.7284 (342K) — so the iter-320K
  stopping-criterion did NOT fire; training continued finding new
  bests. **Now 14 evals (iters 344K-356K) past the 342K best with
  no improvement** (val range 0.7426-0.7677). This is right around
  the BPE run's stopping pattern (13 non-improving evals), but the
  earlier inter-best gaps were 10-20K iters, so another best is
  plausible. Train/val gap mean over iters 330K-356K is ~0.045 —
  under the 0.05 threshold but borderline. Last six evals' gaps:
  0.057, 0.043, 0.047, 0.039, 0.037, 0.064. Speed steady at 4.16
  sec/iter, MFU ~3.5 %. ETA ~2026-06-02 (~7 more days). Process
  alive (PID 36141 on Studio). **Decision: keep running.** Reassess
  if (a) no new best by iter ~380K, or (b) mean gap pushes
  consistently above 0.05.
- **Progress as of 2026-05-27 20:44** (18d 20h elapsed, ~77.4 % of
  the run): iter 386,800 / 500,000 (epoch 5.56). **Still no new best
  since 0.7284 at iter 342K** — now 22 evals plateau-ed (~2d 3h).
  Recent val floor sits ~1-2 % above 0.7284 (best non-improving evals:
  0.7382 at 368K, 0.7394 at 378K, 0.7426 at 354K). Mean train/val gap
  over the 22-eval plateau is 0.0512; last-10-eval mean 0.0537;
  last-5-eval mean 0.0496. Slow upward drift in the 10-eval mean
  (0.0501 → 0.0511 → 0.0585 across the plateau). **Reassessment
  decision: keep running.** Reasoning: mean val loss is still
  trending down across the plateau (0.7568 → 0.7468 over the most
  recent 13 evals vs the immediately-post-best 9 evals), so this
  reads as noise-floor stagnation with a slow gap drift rather than
  classic overfitting onset. Next reassessment trigger: gap mean
  cleanly exceeds 0.06, OR val mean starts trending up, OR no new
  best by ~iter 430K.
- **Progress as of 2026-05-29 05:47** (20d 5h elapsed, ~83 % of the
  run): iter 415,100 / 500,000 (epoch 5.97). **New best val 0.7186
  at iter 390,000** (saved 2026-05-28 00:29, 4.5 h after the prior
  reassessment) — beats the 0.7284 plateau set 2026-05-25 by 1.3 %.
  This vindicates the 2026-05-27 keep-running decision. Since the
  new best, 12 evals plateau-ed (iters 392K-414K, all val > 0.7186,
  min 0.7240 at 394K, next 0.7268 at 410K). Mean train/val gap over
  those 12 evals: 0.0554 — same range as before the new best,
  neither clearly improving nor worsening. None of the reassessment
  triggers (gap mean >0.06, val mean trending up, no new best by
  ~iter 430K) have fired yet. Speed steady 4.17 sec/iter. ETA
  ~2026-06-02 morning.
- **Final state (2026-06-02 08:54 EDT, training complete):** iter
  500,000 / 500,000 (epoch 7.19). **Best val 0.7152 (per-char), set
  at iter 482,000 (epoch 6.93, 2026-06-01 11:54).** Two more bests
  were found after the 2026-05-29 update: 0.7152 at iter 482K
  (the one captured here) plus the iter 390K one already recorded
  above. Total improvement past the apparent 0.7284 plateau (iter
  342K, 2026-05-25): Δ 0.0132 over the final 140K iters. No further
  best in the last 18K iters (482K-500K). Total wall time: 24d 8h
  14m on Studio MPS at ~4.18 sec/iter throughout. No NaN, no
  crashes, no resume incidents. Best checkpoint at
  `pt/char_uppercase_16L_1280.pt`; final at
  `pt/char_uppercase_16L_1280_final.pt` (3.6 GB each). Final samples
  in the log are coherent multi-sentence 19th-century prose with
  proper dialogue structure and held register.
- **Cross-comparison with the M3 BPE run** (per-character loss, since
  BPE-token loss and char-token loss aren't directly comparable):
    - BPE best (original run): 3.3657 BPE-loss at iter 132K, epoch 4.19
      → ~0.748 per-char
    - BPE best (resumed run, see below): 3.2652 BPE-loss at iter 168K
      → ~0.725 per-char
    - char best: 0.7152 per-char at iter 482K, epoch 6.93
  At matched epoch (~4.2), char was already at ~0.73 per-char,
  i.e., already ahead of the original BPE run. Char continued
  improving for ~3 more epochs while the original BPE run had
  begun overfitting at epoch 4.6. The BPE-resumed run found a
  better val (3.2652) past that earlier plateau, narrowing the
  gap to char but still trailing. **Char surpasses BPE on
  per-character loss given the full training budget**, even
  accounting for the resumed BPE run's recovery. See diary 094
  for the analysis (revised 2026-06-02 in commit `abaa52c` to use
  the BPE-resumed best of 0.725 as the proper apples-to-apples
  reference — the 094 table and thesis already reflect this).
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
- Intermediate checkpoints cleaned up 2026-05-20 ~08:58: kept only
  iter 145000 (last save) on the M3 alongside the best-val file.
  Deleted 28 files (iter 5K-140K at every 5K), freed ~113 GB of M3
  storage (M3 went from 213 GB → 326 GB free, 89% → 83% used).
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

### BPE Model Resumed Training (M3 laptop) — COMPLETED 2026-06-01
- **Training complete** — launched 2026-05-26 11:59:12 EDT, finished
  2026-06-01 15:34 EDT (6 days 3 hours 28 minutes wall time). Final
  iter: 220,000 (epoch 6.99). **Best val: 3.2652 BPE-loss at iter
  168,000** (saved 2026-05-28 09:08, epoch ~5.34). No further best
  in the last 52K iters (168K → 220K); val bounced in the 3.40-3.51
  range. Final val (iter 219K): 3.5062. Per-character equivalent of
  the best: 3.2652 / ~4.5 chars-per-token ≈ **0.725 per-char**.
  Final samples are coherent multi-sentence 19th-century prose —
  qualitatively similar to the char model's final samples.
- **Cross-comparison with the Studio char run, updated:** char best
  per-char 0.7152 vs BPE-resumed best per-char 0.725. Char still
  wins by ~0.010 (~1.4 %) but the margin is much narrower than the
  comparison to BPE-original (0.748) would have suggested. The BPE
  resume found ~36K iters of genuine val improvement past the
  original stop at 132K. See diary 094 for the analysis.
- **M3 cleanup note:** as of 2026-06-02 09:30, the training python
  exited cleanly but the wrapper scripts (`/bin/zsh sh/train.sh ...`,
  PID 26584/26585 on the M3) are still alive — they were holding
  `tail -f` on the now-static log. Same SIGINT-vs-SIGTERM pattern
  noted in the original BPE stop. To clean up: SSH to M3 and
  `kill -TERM 26584 26585`.
- **Resume launched** 2026-05-26 11:59:12 EDT to complete the originally-
  planned 220,000 iterations. The earlier run (above) stopped at iter
  145,100 on val-loss plateau; this resume continues from
  `pt/bpe_uppercase_16L_1280_b2_iter145000.pt` to test whether the
  overtraining hypothesis (samples grow more memorized as training
  proceeds past the val-loss minimum) holds quantitatively.
- Launch script: `sh/train_bpe_uppercase_16L_1280_b2_resumed.sh`
  (commit `d56bf22`). Identical hyperparameters to the original launch
  (batch_size=2, lr=1.06e-4 peak, warmup=500, block=4096, eval_interval
  =1000, save_interval=5000) — only `--output` and `--resume` differ,
  so original-run artifacts are preserved untouched.
- Log: `terminal_logs/terminal_log_for_bpe_uppercase_16L_1280_b2_resumed_2026_05_26_1159.txt`
- Output base: `pt/bpe_uppercase_16L_1280_b2_resumed.pt` (best val of
  resumed run). Sibling files: `_resumed_rolling.pt`, `_resumed_tokens.pt`,
  `_resumed_iter<N>.pt` (every 5K iters), `_resumed_final.pt`.
- **First eval at iter 146000** (2026-05-26 14:03 EDT, 1h 56m after
  launch): train loss 3.1089, val loss 3.5406, LR 3.49e-5, epoch 4.64.
  All values within the predicted band; LR matches the cosine
  continuation within rounding. Resume verified faithful — same model
  weights, same hyperparameters, same LR schedule, drifting only
  through the stochastic RNG path because RNG state isn't preserved
  in the checkpoint.
- **Per-iter speed:** ~6.5-7 sec/iter, slightly slower than the 6.6
  sec/iter projection. MFU drifted from 1.47% at startup down to 1.19%
  by iter 146600 — possibly thermal throttling as the M3 warms up.
  ETA at this speed: ~2026-06-01 evening.
- **One artifact in the log:** at the iter-145000 resume-eval, the
  log claims `best val loss: 3.6013` — but this is **not** a real
  improvement over the original run's true best (3.3657 at iter 132K).
  It's because `train.py` treats the first resume-eval as the resumed
  run's initial best. Any val below 3.6013 in this resumed run (including
  the original's plateau range of ~3.44) will be saved as a "new best".
  When comparing best-val numbers across the original-vs-resumed
  records, compare against the original's true 3.3657, not against this
  artifact.
- **Comparison plan when done:** sample from each of {132K best-val,
  145K stopped, 150K-220K resumed-intermediates} on a fixed prompt set
  and probe each sample for verbatim-memorization fraction. If
  memorization fraction grows monotonically past iter 132K, the
  overtraining hypothesis is quantitatively confirmed.
- **`py/memorization_probe.py` IMPLEMENTED (2026-06-06, Linux workstation
  session).** No longer a sketch. Measures two cheap, char-comparable
  signals per checkpoint over a fixed set of corpus passages:
  (1) *extractable memorization* — exact-match prefix length (in chars)
  of a GREEDY free continuation vs the true continuation (Carlini-style;
  the primary signal); (2) *teacher-forced greedy accuracy* — per-token
  argmax==truth in one forward pass (within-tokenizer trend only). Pass
  several checkpoints with `--models` for a single iter-sorted comparison
  table. Reuses sample.py's `generate_local`/`detect_case_preservation`
  and the same meta-path derivation, so behavior matches the sampling
  path. Helpers unit-tested on Linux; **not yet run against real weights**
  (none on the Linux box — needs the Studio/M3). Wrapper
  `sh/memorization_probe_bpe.sh` stages the three BPE contrast points
  {132K=`bpe_uppercase_16L_1280_b2.pt`, 168K=`..._b2_resumed.pt`,
  220K=`..._b2_resumed_final.pt`} from the M3 (same rsync + meta-rename
  trick as the BPE sample scripts) and runs the probe on the Studio.
  Per the 2026-05-29 reframing, {132K, 168K, 220K-final} are the right
  contrast points — 168K is the true val minimum, not 132K.
- **Sample-on-Studio workflow for the resumed run is live (2026-05-27,
  commits `19508a0`, `8ab2d8c`, `244b77b`, `ecd4d5b`).** New script
  `sh/sample_bpe_uppercase_16L_1280_b2_resumed.sh`, modeled on the
  earlier `sh/sample_bpe_uppercase_16L_1280_b2.sh`. Two non-obvious
  quirks worth knowing for any future variant:
  - **Bonjour hostname, not hardcoded IP.** Both BPE sample scripts now
    use `RalphDratman@MacBookProM3Max.local` for the rsync source. The
    original `192.168.1.177` IP was stale — on 2026-05-27 the M3 was at
    `192.168.1.185` (DHCP reassignment, likely because the M3 was on
    wifi rather than Ethernet — same situation noted in
    `sh/backup_checkpoints.sh` and `sh/plot_m3_bpe_snapshot.sh`
    comments). The `.local` name resolves correctly across interface
    and DHCP-lease changes.
  - **Meta files keep the pre-resume name.** When training resumed, the
    tokenizer wasn't re-saved, so the meta files on the M3 are still
    `pt/bpe_uppercase_16L_1280_b2_meta.{pkl,json}` (no `_resumed_`).
    But `py/sample.py` derives the meta path from the model path, so
    on the Studio it expects `pt/bpe_uppercase_16L_1280_b2_resumed_meta.*`.
    The script's rsync invocation pulls from the base name and writes
    to the resumed name in a single step (explicit destination filename
    rather than a directory destination). If you ever create a `_v3`
    or further-resumed run from the same tokenizer, do the same trick.
- **First successful sample from the resumed run (2026-05-27 20:39):**
  iter 157,000, val loss 3.3556 (BPE per-token, ≈0.75 per-character
  at this corpus's ~4.5 chars/token — consistent with diary 093's
  0.77 reading at iter 95K). Five samples generated; four were
  coherent multi-paragraph 19th-century prose; one (sample 4) degenerated
  into a `"IIIll."` repetition loop after a short `D.C. D.C. D.C.`
  header preamble — the kind of low-content high-probability cluster
  `top_k=40` + `temperature=0.8` occasionally falls into. Lower `top_k`
  (e.g. 20) or adding `--top_p 0.9` would suppress it if it recurs.
  Log: `terminal_logs/sample_bpe_uppercase_16L_1280_b2_resumed_2026_05_27_2039.txt`
  on the Studio.
- **Progress as of 2026-05-29 05:47** (2d 17h 41m elapsed since the
  resume launch on 2026-05-26 11:59, ~44 % of the 75K-iter resume
  budget): iter 178,000 / 220,000 (epoch 5.65). **New best val 3.2652
  at iter 168,000** (saved 2026-05-28 09:08). This is the genuine
  new global minimum across the original + resumed run combined: it
  beats the original-run best of **3.3657 at iter 132K by 3.0 %**.
  Implication: the original-run "plateau" at iters 132K-145K that
  triggered the 2026-05-20 stop-early decision was a *local* plateau,
  not the true val-loss minimum. Val kept improving past iter 145K
  through ~iter 168K, with the LR cosine schedule continuing to
  decay into a productive low-LR regime. Since the iter-168K best,
  10 evals plateau-ed (iters 169K-178K, all val > 3.2652, range
  3.36-3.57). Latest eval (iter 178K) val 3.3842, train 3.1280,
  gap 0.26 (wider than the Studio char gap, as expected with the
  smaller batch and stronger gradient noise). Speed ~7 sec/iter
  steady. ETA ~2026-06-01 afternoon (42K iters remaining).
- **Reframing the overtraining experiment (2026-05-29).** The resume
  was launched to test "samples grow more memorized as training
  proceeds past the val-loss minimum." That premise assumed iter 132K
  was at or near the val-loss minimum. The 3.2652 at iter 168K
  shows it wasn't — there were ~36K more iters of genuine val
  improvement after the original stop point. The overtraining
  comparison plan (sample 132K-best vs 145K-stopped vs intermediates
  vs 220K-final, look for monotonic memorization growth) is still
  valid, but the *iter at which val truly stops improving* is now
  ≥ 168K rather than ~132K. Once training completes, the meaningful
  contrast points for the memorization probe should be {132K, 168K
  (new true best), 220K-final}, not the originally-planned brackets.

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

- **Diary 095 (2026-06-02).** Surprisal and pitch:
  speech, music, and the reading of poems. Cross-domain entry
  starting from a voice-memo idea — visualize per-word surprisal in
  prose, connect it to pitch changes in TTS and to key/melody
  changes in music. Covers the two TTS pitch-handling paradigms
  (explicit pitch prediction vs. neural-codec token LM), the Smooth
  Signal Redundancy / Uniform Information Density literature (Aylett
  & Turk, Jaeger, Bell/Jurafsky, Calhoun, Pate-Goldwater, Talman
  et al.), the music parallel (Pearce & Wiggins IDyOM, Huron *Sweet
  Anticipation*), joint text+audio language models (AudioPaLM,
  Spirit-LM-Expressive, Moshi, GPT-4o), an analysis of Stevens's
  *Peter Quince at the Clavier*, notes on how poets (Eliot, Stevens,
  Auden) read their own work flat versus how actors like Burton
  read Hardy theatrically, and a five-step ladder of next-step
  experiments (per-word surprisal visualizer at the easy end,
  Spirit-LM-Expressive-style small-scale joint training at the
  hard end). No code or model changes; entirely a research-direction
  entry.

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

- **`py/train.py` resume infrastructure improved (2026-05-21, commit
  `b874782`).** Two additions to make resume faster and safer:
  - **Tokens cache (`<output_base>_tokens.pt`).** Fresh training writes
    the full tokenized stream alongside the checkpoint; on `--resume`,
    `train.py` loads tokens from this cache instead of re-running
    `tokenizer.encode()`. Cache is invalidated when a fresh training
    starts (overwritten unconditionally) and also if the input corpus
    mtime is newer than the cache mtime. File sizes for the current
    runs: BPE ~2.2 GB (vocab 32K, dtype long), char ~10 GB (vocab 78,
    dtype long). Both are gitignored under `pt/`. Banner prints
    `token_cache: <path> (loaded|rebuilt)` so each training log
    self-documents.
  - **Rolling checkpoint (`<output_base>_rolling.pt`).** After every
    eval (every `eval_interval` iters = 1000 in both current runs),
    `train.py` writes the full checkpoint dict to
    `<output_base>_rolling.pt.tmp` and then `os.rename`s to the final
    path — atomic on the same filesystem, so the file is never
    partial after a crash. **This is the canonical `--resume` target
    after an involuntary break.** Worst-case loss on crash is now
    `eval_interval` iters (~70 min for the BPE run, ~70 min for the
    char run) instead of `save_interval` iters (5000, several hours).
    Banner prints the rolling path. Does not affect the two
    currently-running trainings (their in-memory copies of train.py
    predate the change); will take effect on the next fresh launch.
- **Resume troubleshooting context (2026-05-21).** Ralph reported
  two past resume incidents where "loss looked quite wrong and took
  hours to get back" — he doesn't remember the iter numbers or
  checkpoint files used and can't reconstruct from transcripts (the
  fault that caused the involuntary stop also disrupted the session
  logs). Suspect list ranked by plausibility of *hours-of-regression*:
  (1) resumed from `<base>.pt` (best-val, which can be many iters
  behind the latest) instead of an `_iter*.pt` or now-`_rolling.pt`;
  (2) optimizer state corruption / not applied; (3) `iter_num` lost
  during load so LR schedule restarted from warmup; (4) `--val_split`
  changed between save and resume. The rolling-save change addresses
  (1) by making the "latest" target obvious; the others would need
  individual investigation if symptoms recur. RNG state is not saved
  and not detectable from the standard training log, so its omission
  is cosmetic rather than correctness-affecting.

- **M2 → Studio SSH key auth set up (2026-05-23).** The M2 MacBook Pro's
  `~/.ssh/id_ed25519.pub` is now in the Studio's `~/.ssh/authorized_keys`.
  Studio is reachable as `mac-studio.local` (resolves to 192.168.1.233
  on Ethernet LAN). Lets a Claude session running on the M2 query Studio
  state directly: `ssh RalphDratman@mac-studio.local "..."`. The
  Studio's project lives at
  `/Users/RalphDratman/0_Home_Folder_Working_Mac_Studio/bpe_vs_char_model_comparison/`.
  Note: `ssh-copy-id` from the M2 needed the flags
  `-o PreferredAuthentications=password -o PubkeyAuthentication=no`
  because the M2 has three keys (id_ed25519, id_rsa, id_rsa-2) and SSH
  was burning through its auth attempts on those before reaching the
  password prompt. Also, the interactive password prompt did not work
  through Claude Code's `!`-prefix shell; running `ssh-copy-id` from
  a regular Terminal app succeeded.

- **M3 → Studio SSH key auth confirmed (2026-05-26).** Passwordless SSH
  from the M3 to the Studio (`ssh RalphDratman@mac-studio.local`)
  already worked at the start of this session, even though only the
  reverse direction (Studio → M3) and (M2 → Studio) had been documented
  before. Set up at some prior unknown date; useful for any M3-side
  session that needs to inspect Studio state read-only or run a
  cross-machine bootstrap (see next bullet).

- **`claude_memory/` dotfiles-style sync installed (2026-05-26, commits
  `4f25465`, `9f8bafe`).** Claude Code's per-project memory dir used to
  live at `~/.claude/projects/<slug>/memory/`, separately on each
  machine (different slug per machine because the project path differs:
  `0-Home-Working-on-M3-Pro/` on M3 vs `0_Home_Folder_Working_Mac_Studio/`
  on Studio). Studio had accumulated 16 memory files since April; the
  M3 had only this session's 2 new feedback files. Memories were not
  shared across machines.
  - Reorganized: 17 memory files (Studio's 15 + this session's 2)
    plus a re-indexed `MEMORY.md` moved into the project repo at
    `claude_memory/`. Total: 18 files.
  - On both machines, `~/.claude/projects/<slug>/memory/` is now a
    symlink to the repo's `claude_memory/`. Same files serve both
    machines through the existing git workflow.
  - Going-forward workflow: when memory accumulates on one machine,
    commit + push from there; pull on the other machine when
    convenient. Same as the rest of the project — no new sync tool.
  - One-time bootstrap of the Studio side required a cross-SSH
    `git pull` from this M3 session, which is the kind of cross-
    machine git operation the `feedback_one_git_per_machine.md` rule
    normally forbids. Exemption clause added to that memory file:
    one-time bootstrap operations are allowed with explicit prior
    approval from Ralph; normal one-git-per-machine resumes after.
  - The memory dirs Claude Code creates for other projects on each
    machine (e.g. `study-corpus-and-training-2c-char` on M3,
    `study-corpus-and-training-2c-bpe` on Studio) are unaffected.
    Only this project's memory dir was migrated.

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
- `old_8_GB_corpus_pt/` — old 8 GB corpus run (iters 10K-160K). Studio only; NOT backed up to Expansion.
- `unshuffled_corpus_pt/` — unshuffled 2.5 GB corpus run (iters 10K-50K). Studio only; NOT backed up.
- `doc_shuffled_batch4_pt/` — failed batch=4 lr=0.0003 run (iters 10K-20K). Studio only; NOT backed up.
- `../valuable_checkpoints/bpe_16L16H_old_corpus_iter160k_Excellent_from_8GB_corpus.pt`
- **`../valuable_checkpoints/` BPE files backed up to Expansion 2026-05-20.** The four top-level .pt files
  (bpe_16L16H_32k_vocab_best_val3.35_iter229000.pt, bpe_16L16H_old_corpus_iter160k_Excellent_from_8GB_corpus.pt,
  bpe_32k_bf16_iter153k_almost_time_to_stop.pt, bpe_32k_bf16_iter163k_epoch3.1.pt — ~38 GB total) were rsynced
  to `/Volumes/Expansion/0_backups_Mac_Studio_Expansion/valuable_checkpoints/`. The `B_9GB/` (Model B) and `py/`
  subdirs were already there from an earlier backup.

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
- [ ] Watch Studio char train/val divergence. **Reassessed 2026-05-27
  at iter 386,800**: criterion borderline-fired; decision: keep
  running. **Resolved 2026-05-28 00:29** — new best val 0.7186
  arrived at iter 390K (4.5 h after the reassessment), beating the
  prior 0.7284 by 1.3 %. As of 2026-05-29 05:47 (iter 415K), 12
  evals plateau-ed since the new best. Active triggers for next
  reassessment: gap mean >0.06, OR val mean trending up, OR no new
  best by ~iter 460K (set 70K iters past the new best, same gap
  the prior trigger used). See progress bullets above for details.
- [ ] Decide what to do with intermediate checkpoints from the Studio
  char run (~96 GB; clean periodically or keep all 25 for
  layer-stability analysis per diary 080).
- [x] **DONE 2026-05-20.** M3 BPE intermediate checkpoints cleaned up.
  Kept best (iter 132K) and last (iter 145K); deleted the other 28
  iter_*.pt files. Freed ~113 GB. No layer-stability analysis planned
  on the BPE model; if the need arises later, the best/last pair plus
  the Studio backup is what remains.

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
