#!/bin/bash
#
# train_char_uppercase_16L_1280_seed2_CUDA.sh
#
# SECOND-SEED REPLICATION of the Studio baseline char run (diary 099,
# step 2a). Runs on the Linux A6000 box (~0.52 sec/iter → ~3 days for
# the full 500K iters, vs 24 days on the Studio).
#
# Purpose: every headline result in this lab is n=1. This run puts a
# second, independently-seeded sample under the baseline so that
# (a) diary 094's char-vs-BPE margin (0.7152 vs ~0.725 per-char, a
#     ~1.4 % gap at the noise floor) gains an empirical error bar, and
# (b) diary 100's real-word-fraction curve gets a replication point.
#
# Hyperparameters are IDENTICAL to sh/train_char_uppercase_16L_1280.sh
# (the Studio baseline), including batch_size=4 — the A6000 could fit
# larger batches but that would change the gradient-noise/LR regime and
# break comparability. Differences:
#   - calls sh/train_cuda.sh (conda-env python, nohup, prints PID)
#   - --seed 2: documented replication seed (train.py --seed added
#     2026-06-11; the original baseline ran entropy-seeded, so ANY
#     fixed seed is a valid independent draw; 2 is the convention for
#     "second run")
#   - --output suffixed _seed2_cuda so artifacts stay distinct
#
# NOTE: CUDA kernels are not bitwise deterministic, so --seed 2 does not
# make THIS run replayable bit-for-bit; it controls init and batch order
# and documents the draw.
#
# Output: pt/char_uppercase_16L_1280_seed2_cuda.pt (best-val)
#         pt/char_uppercase_16L_1280_seed2_cuda_rolling.pt (latest, atomic)
#         pt/char_uppercase_16L_1280_seed2_cuda_tokens.pt (tokens cache)
#         pt/char_uppercase_16L_1280_seed2_cuda_final.pt (last iter)
#
# Disk note: save_interval=20000 → 25 intermediate checkpoints × 3.6 GB
# ≈ 90 GB in pt/ — check free space on the A6000 before launch.
# Stopping: kill -TERM <PID> (the wrapper prints the PID on launch).

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280_seed2_cuda.pt \
    --mode continuous \
    --tokenizer char \
    --precision bfloat16 \
    --n_layer 16 \
    --n_head 8 \
    --n_embd 1280 \
    --block_size 4096 \
    --batch_size 4 \
    --learning_rate 1.5e-4 \
    --warmup_iters 2000 \
    --max_iters 500000 \
    --eval_interval 2000 \
    --eval_iters 20 \
    --save_interval 20000 \
    --log_interval 100 \
    --sample_interval 10000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 2
