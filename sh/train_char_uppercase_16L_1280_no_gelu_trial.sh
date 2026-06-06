#!/bin/zsh
#
# train_char_uppercase_16L_1280_no_gelu_trial.sh
#
# Short 10K-iter ablation trial: same architecture and corpus as the just-
# completed Studio char run (16L/8H/n_embd=1280, block=4096, char vocab=78,
# corpus_high_quality_uppercase_2026_05_08.txt), but with the GELU activation
# inside every MLP block disabled.
#
# With GELU off, each MLP sub-block becomes c_proj(c_fc(x)) -- two linear
# layers in series with no nonlinearity between them. Mathematically the
# whole MLP collapses to a single rank-1280 linear map. Parameter count
# is unchanged at 320M; what changes is that the MLP can only mix
# coordinates linearly, not bend them. Attention (with its softmax) is
# untouched.
#
# Purpose: probe whether the no-GELU model trains at all and what its
# loss trajectory looks like over the first 10K iters (~12 h at 4.18
# sec/iter on Studio MPS).
#
# LR-schedule caveat: cosine decay endpoint is tied to max_iters in
# train.py's get_lr(). With max_iters=10000 the LR decays from 1.5e-4 to
# 1.5e-5 over iters 2K-10K -- much faster than the baseline run's near-
# constant LR during its own iters 0-10K (baseline used max_iters=500000).
# So the trial's iter-10K val loss is NOT directly comparable to the
# baseline's iter-10K val loss. It is a "does this architecture train
# coherently?" probe, not a matched ablation. If the curve looks
# promising, a follow-up with a matched LR schedule (e.g., max_iters
# kept at 500000 and an external stop) will give a clean comparison.
#
# Output: pt/char_uppercase_16L_1280_no_gelu.pt (best-val)
#         pt/char_uppercase_16L_1280_no_gelu_rolling.pt (latest, atomic)
#         pt/char_uppercase_16L_1280_no_gelu_tokens.pt (tokens cache)
#         pt/char_uppercase_16L_1280_no_gelu_final.pt (last iter)

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280_no_gelu.pt \
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
    --no_gelu
