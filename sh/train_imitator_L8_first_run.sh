#!/bin/zsh
#
# train_imitator_L8_first_run.sh - First imitator experiment
#
# Trains a 21M-param vector-to-vector transformer to predict the next
# residual-stream vector at layer 8 of the frozen 875M BPE model.
#
# This is a short exploratory run (5000 iters) to find out whether
# mid-layer trajectories are autoregressively learnable at all.
#
# What to look for:
#   - cos_sim should climb from ~0 toward 0.3+ within a few thousand iters
#   - If it stays flat near 0, the approach needs rethinking
#

sh/train_imitator.sh \
    --small_model pt/bpe_16L16H_iter170000.pt \
    --input txt_local/corpus_books_shuffled_2026_04_18.txt \
    --output pt/imitator_L8.pt \
    --token_cache txt_local/corpus_tokens_32k.pt \
    --split_layer 8 \
    --d_model 512 \
    --n_layer 6 \
    --n_head 8 \
    --block_size 512 \
    --batch_size 8 \
    --max_iters 5000 \
    --learning_rate 3e-4 \
    --warmup_iters 200 \
    --eval_interval 200 \
    --log_interval 20 \
    --save_interval 2000 \
    --sanity_check
