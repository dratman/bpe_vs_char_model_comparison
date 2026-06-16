#!/bin/bash
#
# train_char_uppercase_16L_1280_reversed_CUDA.sh
#
# coupler-queue item 0003 — full-length REVERSED character training.
#
# Trains ONE full-size (16L/8H/1280, block 4096, ~320M param) character model on
# the within-split REVERSED corpus, matched in EVERY modeling hyperparameter to
# the best forward char model `char_uppercase_16L_1280` (best val 0.7152 per-char
# at iter 482K), so its validation cross-entropy / bits-per-character is directly
# comparable to that EXISTING forward checkpoint in pt/. No new forward run.
#
# The ONLY intended difference vs the forward run is the DIRECTION of the text:
#   --input ..._REVERSED_within_splits.txt
# which was built + verified by py/make_reversed_corpus.py (VAL_SPLIT=0.1 — the
# SAME positional split the forward run used, so reverse(train)/reverse(val) line
# up exactly and the char vocab/meta is identical). Report val loss in BITS PER
# CHARACTER = nats / ln(2); reverse generated samples back to normal reading order
# before eyeballing (py/extract_revtest_samples.py).
#
# Two operational (non-modeling) flags differ from the MPS forward launch:
#   --no_fused          : the seed-2 diagnosis isolated the CUDA training
#                         instability to fused AdamW; the MPS forward run used the
#                         plain AdamW path, so --no_fused matches it and avoids the
#                         diary-103 instability. (NOT a modeling change.)
#   --accept_overrides  : standing policy — train.py polls this JSON live (every
#                         1000 iters) so the run is stoppable/extendable via a
#                         committed override ({"max_iters": <n>}) WITHOUT a kill.
# eval_interval (2000) and sample_interval (10000) match the forward run exactly,
# so the val-loss curve and the periodic in-log generated samples line up. ONE
# operational (non-modeling) deviation: save_interval is 100000 (not the forward's
# 20000) to bound disk use on the shared A6000 — checkpoint frequency does not
# affect the trained model at all. Crash recovery uses the rolling checkpoint
# (rewritten every eval_interval); the spec's deliverables (best + final
# checkpoints, the bpc curve from the log, early/mid/late in-log samples) are
# fully preserved, and 4 intermediate _iter checkpoints (100K/200K/300K/400K)
# remain for downstream iter-specific probing.
#
# Output base: pt/char_uppercase_16L_1280_reversed_cuda.pt (+ _rolling/_tokens/
# _iter*/_final). ~0.52 s/iter on the A6000 -> ~3 days for 500K iters.
# Stop without kill: write {"max_iters": <n>} to the overrides JSON below.
#
# Idempotency / double-launch guard: if this run already COMPLETED (train.py
# writes _final.pt only at max_iters) or is already running, do nothing. This
# self-neutralizes the armed waiter (queue_0003_reversed_after_seed2.sh) so it
# cannot re-launch a finished/running job.

OUT=pt/char_uppercase_16L_1280_reversed_cuda.pt
OVERRIDES=pt/char_uppercase_16L_1280_reversed_cuda_overrides.json

if [ -f pt/char_uppercase_16L_1280_reversed_cuda_final.pt ]; then
    echo "0003 reversed run already completed (_final.pt exists); skipping launch."
    exit 0
fi
if pgrep -f "py/train.py.*char_uppercase_16L_1280_reversed_cuda" > /dev/null; then
    echo "0003 reversed run already running; skipping launch."
    exit 0
fi

# Start the override file empty so the live override mechanism is armed from iter 0.
echo '{}' > "$OVERRIDES"

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08_REVERSED_within_splits.txt \
    --output "$OUT" \
    --accept_overrides "$OVERRIDES" \
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
    --save_interval 100000 \
    --log_interval 100 \
    --sample_interval 10000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 1337 \
    --no_fused
