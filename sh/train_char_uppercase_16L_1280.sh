#!/bin/zsh
#
# train_char_uppercase_16L_1280.sh - Char-level training on the case-preserved
# corpus, scaling up from char_high_quality (152M, lowercase, block=512) to
# 320M params at block=4096.
#
# Architecture: 16 layers, 8 heads, n_embd=1280, block=4096
# Parameters:   320M (vs char_high_quality's 151M; 2.1x scale-up)
# Precision:    bfloat16 (confirmed working on MPS, halves memory vs fp32)
# Batch size:   4
# Learning rate: 1.5e-4, sqrt-scaled from 3e-4@batch=16; cosine to 1.5e-5
# Max iters:    500,000  (~6.4 epochs of 1.27 B-char corpus at 16,384 tok/iter)
#
# Speedtest (2026-05-09, 200 iters): 4.18 sec/iter steady-state on M3 Studio.
# Wall-time projection: 500K iters ≈ 24 days. Watching for train/val divergence
# from epoch 4 onward; willing to stop early if anything looks wrong.
#
# Corpus: txt_local/corpus_high_quality_uppercase_2026_05_08.txt
#   - 1.27 GB, 3,979 books, 78-char vocab (case preserved), document-shuffled
#   - <|endoftext|> separators between books, seed=42
#   - Recovered from Haiku decisions via py/match_book_samples.py
#
# Per-position logging is verbose: every iter banner now prints n_layer,
# n_head, n_embd, dropout, weight_decay, betas, grad_clip, etc. (extended in
# train.py 2026-05-09).

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280.pt \
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
    --dropout 0.0
