#!/bin/zsh
#
# train_bpe_uppercase_16L_1280_b2.sh - same as train_bpe_uppercase_16L_1280.sh
# but with batch_size=2 instead of 4. Use this if memory pressure on the
# M3 (64 GB unified) makes batch=4 unreliable.
#
# Three coupled changes from the batch=4 version:
#   --batch_size:    4 -> 2          (halves activation memory)
#   --learning_rate: 1.5e-4 -> 1.06e-4   (sqrt-scaled: 1.5e-4 * sqrt(2/4))
#   --max_iters:    110000 -> 220000     (preserves 8.2 B-char data
#                                         exposure to match Studio's char
#                                         run; tokens-per-iter halves so
#                                         iters double)
#
# Other knobs unchanged from the batch=4 version. Output filename and
# meta.pkl are different (suffix _b2) so this run's artifacts do not
# collide with the existing batch=4 run's files.
#
# Wall-time projection on M3 laptop: ~6-7 sec/iter unchanged
# (per-iter time mostly tracks block_size and depth, not batch); 220K
# iters → ~17-18 days wall time. Roughly double the batch=4 run.
#
# The comparison metric (per-character loss at fixed character budget)
# is preserved across this change; see the discussion at the time of
# this script's creation in the project session log. batch_size is not
# directly matchable across char↔BPE tokenization, so the change does
# not invalidate the cross-tokenization comparison.

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/bpe_uppercase_16L_1280_b2.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 32000 \
    --precision bfloat16 \
    --n_layer 16 \
    --n_head 8 \
    --n_embd 1280 \
    --block_size 4096 \
    --batch_size 2 \
    --learning_rate 1.06e-4 \
    --warmup_iters 500 \
    --max_iters 220000 \
    --eval_interval 1000 \
    --eval_iters 20 \
    --save_interval 5000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0
