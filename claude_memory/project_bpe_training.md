---
name: BPE 16L16H training run status
description: Status and details of the ongoing BPE model training started 2026-04-05
type: project
---

BPE 16L16H model training in progress since 2026-04-05 17:54 EDT.

**Why:** Comparing BPE vs character-level tokenization on identical 822M-param architectures to understand how tokenization affects what a transformer learns internally.

**How to apply:** Check the training log at terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt for current progress. The model was at iter ~35,600 / loss ~3.32 as of 2026-04-07. Training runs at ~700 iters/hour. The plan is to stop early based on sample quality rather than running the full 200,000 iterations. Loss is in deep diminishing returns numerically, but small decrements may still yield noticeable sample quality improvements.
