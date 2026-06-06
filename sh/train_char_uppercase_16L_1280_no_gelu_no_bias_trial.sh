#!/bin/zsh
#
# train_char_uppercase_16L_1280_no_gelu_no_bias_trial.sh
#
# Second short 10K-iter ablation trial in the no-GELU series. Same setup
# as train_char_uppercase_16L_1280_no_gelu_trial.sh, but additionally
# disables every bias term in the model (LayerNorm beta, Linear bias in
# attention QKV proj, attention output proj, MLP up-proj, MLP down-proj).
# This is the LLaMA/PaLM convention -- biases are essentially free
# parameters that modern large transformers omit without quality loss.
#
# Architectural state after both flags:
#   - n_layer=16, n_head=8, n_embd=1280, block_size=4096 (unchanged)
#   - GELU disabled inside every MLP block (--no_gelu) -- MLP becomes a
#     single rank-1280 linear map per block
#   - All biases removed (--no_bias) -- ~226K params dropped (~0.07%)
#   - Attention softmax and output log-softmax remain as the only
#     element-wise nonlinearities
#
# Purpose: probe whether the no-GELU + no-bias model trains coherently
# over the same 10K-iter window, for direct comparison against:
#   (a) the baseline 16L/1280 char run (all biases, GELU on) -- 500K iters
#   (b) the first ablation, no-GELU only, biases on -- 10K iters
#
# LR-schedule caveat carried over from the first trial: cosine decay
# endpoint = max_iters, so LR drops from 1.5e-4 to 1.5e-5 over iters
# 2K-10K. Not a clean iter-10K comparison to baseline, but a clean
# comparison to the no-GELU-only trial since both have the same
# schedule.
#
# Output: pt/char_uppercase_16L_1280_no_gelu_no_bias.pt (best-val)
#         pt/char_uppercase_16L_1280_no_gelu_no_bias_rolling.pt (latest)
#         pt/char_uppercase_16L_1280_no_gelu_no_bias_tokens.pt (cache)
#         pt/char_uppercase_16L_1280_no_gelu_no_bias_final.pt (last iter)

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280_no_gelu_no_bias.pt \
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
    --max_iters 10000 \
    --eval_interval 2000 \
    --eval_iters 20 \
    --save_interval 5000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --no_gelu \
    --no_bias
