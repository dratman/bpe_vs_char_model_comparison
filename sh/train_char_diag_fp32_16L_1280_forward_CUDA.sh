#!/bin/bash
#
# train_char_diag_fp32_16L_1280_forward_CUDA.sh
#
# DIAGNOSTIC for the coupler-queue 0003 re-plan (editor reply 2026-06-19): is the
# 16L/1280 big-config CUDA instability a bf16 problem?
#
# Runs the big config in FULL FP32 (--precision float32 => train.py uses
# nullcontext, NO autocast) on the FORWARD corpus, single variable vs the failed
# seed-2 run (bf16 -> fp32; everything else matched, incl. --no_fused). Grad
# clipping is already engaged (train.py hardcodes clip_grad_norm_ to 1.0 each step).
#
# max_iters is 500000 ON PURPOSE so the cosine LR stays near-constant (~1.5e-4) over
# the first ~20K iters, matching the known-good MPS forward run's 0-20K trajectory
# for a clean comparison (do NOT shorten it — that reintroduces the fast-LR-decay
# confound). The plan is to WATCH the first ~20K iters, then graceful-stop via
#   sh/stop_run_via_override.sh pt/char_diag_fp32_16L_1280_forward_cuda_overrides.json <n>
#
# Decision rule (editor): descends past ~1.2 and tracks the MPS forward curve, no
# divergence -> bf16 was the culprit; adopt fp32 recipe + launch the full reversed
# 500K run. Also diverges -> stop debugging CUDA, run the reversed job on Studio/MPS.
#
# Idempotency guard.
if [ -f pt/char_diag_fp32_16L_1280_forward_cuda_final.pt ]; then
    echo "fp32 diagnostic already completed (_final exists); skipping."
    exit 0
fi
if pgrep -f "py/train.py.*char_diag_fp32_16L_1280" > /dev/null; then
    echo "fp32 diagnostic already running; aborting duplicate launch."
    exit 0
fi

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_diag_fp32_16L_1280_forward_cuda.pt \
    --accept_overrides pt/char_diag_fp32_16L_1280_forward_cuda_overrides.json \
    --mode continuous \
    --tokenizer char \
    --precision float32 \
    --n_layer 16 \
    --n_head 8 \
    --n_embd 1280 \
    --block_size 4096 \
    --batch_size 4 \
    --learning_rate 1.5e-4 \
    --warmup_iters 2000 \
    --max_iters 500000 \
    --eval_interval 500 \
    --eval_iters 20 \
    --save_interval 5000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 1337 \
    --no_fused
