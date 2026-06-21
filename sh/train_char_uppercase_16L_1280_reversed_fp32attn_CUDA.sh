#!/bin/bash
#
# train_char_uppercase_16L_1280_reversed_fp32attn_CUDA.sh
#
# coupler-queue 0003 RE-RUN with the fixed recipe. The first 0003 attempt failed the
# big-config CUDA bf16 instability (val 1.20@12K -> ~2.4 plateau). The fp32 diagnostic
# proved precision was the cause (full-fp32 descended monotonically past 1.2 with no
# divergence). This run uses the editor's production recipe: **bf16 everywhere EXCEPT
# attention, which runs in fp32** (`--fp32_attention`) — keeps the fix's stability at
# close to bf16 speed (full fp32 was ~6x slower / ~19 days).
#
# Matched to the forward best char model in every modeling hyperparameter; the ONLY
# intended difference is data direction (reversed-within-split corpus). Reversed val
# bits-per-char is therefore directly comparable to the forward floor 0.7152 per-char.
#
# Idempotency guards.
if [ -f pt/char_uppercase_16L_1280_reversed_fp32attn_cuda_final.pt ]; then
    echo "reversed fp32attn run already completed (_final exists); skipping."
    exit 0
fi
if pgrep -f "py/train.py.*reversed_fp32attn_cuda" > /dev/null; then
    echo "reversed fp32attn run already training; aborting duplicate launch."
    exit 0
fi

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08_REVERSED_within_splits.txt \
    --output pt/char_uppercase_16L_1280_reversed_fp32attn_cuda.pt \
    --accept_overrides pt/char_uppercase_16L_1280_reversed_fp32attn_cuda_overrides.json \
    --mode continuous \
    --tokenizer char \
    --precision bfloat16 \
    --fp32_attention \
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
    --save_interval 100000 \
    --log_interval 100 \
    --sample_interval 10000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 1337 \
    --no_fused
