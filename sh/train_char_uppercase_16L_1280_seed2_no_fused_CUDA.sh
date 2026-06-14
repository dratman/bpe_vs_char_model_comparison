#!/bin/bash
#
# train_char_uppercase_16L_1280_seed2_no_fused_CUDA.sh
#
# STABILITY-FIX TRIAL 1 for the seed-2 CUDA replication failure
# (diary 103 follow-up, 2026-06-13). The original seed-2 run on this
# A6000 box trained unstably — val loss oscillated 1.3-2.7 and never
# tracked the Studio baseline's smooth descent to ~0.80 — even though
# the byte-identical config ran 500K iters cleanly on the Studio (MPS).
#
# The code-level diagnosis isolated the CUDA-vs-MPS difference to
# **fused AdamW** (CUDA gets fused=True via model.configure_optimizers;
# MPS gets the non-fused implementation). This trial forces the
# non-fused path on CUDA with `--no_fused` and changes NOTHING ELSE, so
# if it trains stably, fused AdamW was the culprit and hyperparameter
# matching with the baseline is preserved.
#
# Hyperparameters are byte-identical to
# sh/train_char_uppercase_16L_1280_seed2_CUDA.sh (lr 1.5e-4, batch 4,
# warmup 2000, max_iters 500000, --seed 2). max_iters is kept at 500000
# ON PURPOSE so the cosine-LR schedule matches the baseline — do NOT
# shorten it, or you reintroduce the fast-LR-decay confound that muddied
# the no-GELU trial. Plan: let it run to ~iter 6K-10K, confirm the val
# curve is smooth (no 1.3<->2.7 oscillation), then decide whether to let
# it run to the full floor or stop with `kill -TERM <PID>`.
#
# Only differences from the seed-2 script:
#   - --no_fused           (the fix under test)
#   - --output suffixed _no_fused_cuda so artifacts stay distinct
#
# Output: pt/char_uppercase_16L_1280_seed2_no_fused_cuda.pt (best-val)
#         + _rolling.pt / _tokens.pt / _iter<N>.pt / _final.pt siblings.
# Disk: save_interval=20000 -> 25 ckpts x 3.6 GB ~= 90 GB if run long.
# Stopping: kill -TERM <PID> (the wrapper prints the PID on launch).

# Double-launch guard. On 2026-06-14 two queue waiters ended up armed for
# this trial (the original 152043->1369813 chain plus a redundant waiter on
# the no-GELU PID). Whichever fires second must NOT start a duplicate run on
# the same checkpoint base (would corrupt checkpoints + contend for the GPU).
# If a no-fused trial is already training, abort cleanly.
if pgrep -f "py/train.py.*seed2_no_fused_cuda" > /dev/null; then
    echo "no-fused trial already running (pgrep matched); aborting duplicate launch."
    exit 0
fi

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280_seed2_no_fused_cuda.pt \
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
    --seed 2 \
    --no_fused
