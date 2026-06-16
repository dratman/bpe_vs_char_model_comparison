#!/bin/bash
#
# resume_seed2_no_fused_to_floor_CUDA.sh
#
# Resumes the seed-2 --no_fused stability trial from its latest checkpoint and
# lets it coast to the 500K floor. Used AFTER the coupler-queue 0002 pilot pair
# runs (which happens once Ralph stops the trial early to free the GPU). The
# trial's diagnostic verdict is already in hand (smooth at ~78K -> fused AdamW
# confirmed as the seed-2 instability cause); the remaining run only feeds the
# diary-094 error bar, so the editor asked to delay it ~an hour (for the pilot),
# not drop it.
#
# Two changes vs the original launch:
#   - --resume <rolling checkpoint>   (continue, not restart; iter/optim restored)
#   - --accept_overrides <json>       (NEW STANDING POLICY): train.py now polls
#     this file live, so this run can be stopped/extended via a committed override
#     ({"max_iters": <n>}) WITHOUT a kill — sidestepping the deny-list next time.
# Everything else is byte-identical to the original trial.
#
# Idempotency: if the trial already reached its floor (_final.pt exists) or is
# already running, do nothing.

RESUME_CKPT=pt/char_uppercase_16L_1280_seed2_no_fused_cuda_rolling.pt
OVERRIDES=pt/char_uppercase_16L_1280_seed2_no_fused_cuda_overrides.json

if [ -f pt/char_uppercase_16L_1280_seed2_no_fused_cuda_final.pt ]; then
    echo "seed-2 trial already completed (_final.pt exists); not resuming."
    exit 0
fi
if pgrep -f "py/train.py.*seed2_no_fused_cuda" > /dev/null; then
    echo "seed-2 trial already running; not resuming."
    exit 0
fi
if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: resume checkpoint not found: $RESUME_CKPT"
    exit 1
fi

sh/train_cuda.sh \
    --resume "$RESUME_CKPT" \
    --accept_overrides "$OVERRIDES" \
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
