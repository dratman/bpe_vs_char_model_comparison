#!/bin/bash
#
# train_char_revtest_pilot_6L_768_forward_CUDA.sh
#
# coupler-queue item 0002 — reversed-text learnability pilot, FORWARD half.
#
# One of a MATCHED PAIR (forward + reversed) that differ in EXACTLY one thing:
# the direction of the character stream. This is the FORWARD baseline: it trains
# on the corpus as-is. Its partner,
# sh/train_char_revtest_pilot_6L_768_reversed_CUDA.sh, trains on the same corpus
# reversed character-by-character WITHIN each train/val split (built + verified
# by py/make_reversed_corpus.py) so each model is held out on the same source
# text read in its own direction.
#
# Small/rough PILOT (Ralph asked for a cheap first look, not a production run):
#   6L / 6H / 768 embd (~43M params), block 256, batch 64, 10K iters.
#   LR 5e-4 cosine-decaying to min_lr = lr/10 = 5e-5 over the FULL 10K iters
#   (train.py get_lr: min_lr=learning_rate/10, decay across max_iters), warmup 200.
#   --seed 1337 — IDENTICAL to the reversed run, so the only difference is data
#   direction (same init, same seed).
#
# Output: pt/char_revtest_pilot_6L_768_forward_cuda.pt (+ _rolling/_tokens/_final).
# Stop:   kill -TERM <PID> (auto-terminates at max_iters anyway).

# Idempotency / double-run guard: if this pilot already COMPLETED (train.py writes
# _final.pt only at max_iters), do nothing. This self-neutralizes the armed queue
# runner (queue_revtest_pilot_after_seed2.sh) so it cannot re-launch a finished
# pilot when the seed-2 trial eventually exits days from now.
if [ -f pt/char_revtest_pilot_6L_768_forward_cuda_final.pt ]; then
    echo "forward pilot already completed (_final.pt exists); skipping launch."
    exit 0
fi

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_revtest_pilot_6L_768_forward_cuda.pt \
    --mode continuous \
    --tokenizer char \
    --precision bfloat16 \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 768 \
    --block_size 256 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --warmup_iters 200 \
    --max_iters 10000 \
    --eval_interval 200 \
    --eval_iters 20 \
    --save_interval 2000 \
    --log_interval 50 \
    --sample_interval 2000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 1337
