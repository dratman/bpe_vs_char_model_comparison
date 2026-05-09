#!/bin/zsh
#
# train_bpe_uppercase_16L_1280.sh - BPE-tokenized parallel of the
# Studio's char_uppercase_16L_1280 run. Intended for the M3 laptop
# (64 GB unified memory).
#
# Purpose: a controlled tokenization comparison. Same architecture,
# same corpus, same case-preservation, same batch / lr / precision /
# warmup-style as the char run on Studio. The only intentional
# variable is char tokenization vs BPE tokenization.
#
# Architecture: 16 layers, 8 heads, n_embd=1280, block_size=4096
#   — identical to sh/train_char_uppercase_16L_1280.sh.
# Param count: ~360M (vs char's 320M; the difference is the larger
#   32K-token embedding table, since lm_head is tied to wte).
# Tokenizer: BPE, vocab_size=32000 (matches train_bpe_32k_bf16.sh convention).
# Batch:    4
# Precision: bfloat16 (confirmed working on MPS for both Studio and M3).
# Max iters: 110000  - data-matched to the char run's ~6.4 corpus epochs.
#   Char sees 8.2 B chars over 500K iters; BPE sees ~1.8 B tokens over
#   110K iters; both correspond to ~6.4 passes through the 1.27 B-char
#   (282 M BPE-token) corpus.
# Warmup:   500 (matches the existing BPE-run convention; ~0.45% of max_iters).
#
# Speed projection: M3 laptop runs MPS workloads at ~60-70% of M3 Studio
# throughput, so expect ~6-7 sec/iter (vs Studio's 4.18 at the same
# arch). 110K iters → ~8-9 days wall time.
#
# First-invocation overhead: BPE tokenizer training takes ~15-30 min
# on this 1.27 GB corpus before training begins. The trained tokenizer
# is saved to pt/bpe_uppercase_16L_1280_meta.pkl + .json and reused on
# any restart.
#
# Corpus (must be present in txt_local/ before running):
#   txt_local/corpus_high_quality_uppercase_2026_05_08.txt
#   (1.27 GB, 3,979 books, case-preserved, document-shuffled, seed=42).
#   This file is gitignored. To copy from the Studio:
#     rsync -av --progress \
#       <Studio>/.../bpe_vs_char_model_comparison/txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
#       <M3>/.../bpe_vs_char_model_comparison/txt_local/
#
# Thermal note: M3 laptop has less cooling headroom than the Studio.
# For a multi-day run keep the lid open and the machine plugged in.
# The training-monitor LaunchAgent (if installed on the M3) will log
# thermal status every 5 minutes; watch for "thermal: throttling".

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/bpe_uppercase_16L_1280.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 32000 \
    --precision bfloat16 \
    --n_layer 16 \
    --n_head 8 \
    --n_embd 1280 \
    --block_size 4096 \
    --batch_size 4 \
    --learning_rate 1.5e-4 \
    --warmup_iters 500 \
    --max_iters 110000 \
    --eval_interval 1000 \
    --eval_iters 20 \
    --save_interval 5000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0
