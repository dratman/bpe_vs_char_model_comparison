#!/bin/bash
#
# train_wordpiece_uppercase_16L_1280_b2_no_gelu_CUDA.sh
#
# WordPiece + NO-GELU run — second half of the WordPiece/no-GELU pair
# (Ralph's experiment, 2026-06-11; diary 102).
#
# The question this run answers (with its control,
# sh/train_wordpiece_uppercase_16L_1280_b2_CUDA.sh): does the GELU's
# contribution shrink when the tokenizer already supplies the lexical
# inventory? At char level, removing the GELU cost 0.16-0.20 nats/char
# and ~4x slower lexical acquisition (diaries 098/100). With WordPiece,
# common words are atomic tokens — no spelling-out, no MLP key-value
# lookup needed for word identity. If the no-GELU penalty largely
# disappears here, the GELU's early-training role really is lexical
# retrieval; if it persists, the GELU does more than store the lexicon.
#
# Identical to the control in EVERYTHING except --no_gelu:
#   - same --seed 42 -> identical init and batch order
#   - --tokenizer_from loads the control's trained tokenizer, so both
#     runs see the exact same token stream (and this run skips the
#     ~1 h tokenizer training). REQUIRES the control run to have saved
#     pt/wordpiece_uppercase_16L_1280_b2_cuda_meta.pkl first.
#
# Runs on the A6000 via sh/train_cuda.sh (prints PID; stop with
# kill -TERM <PID>). Normally launched by sh/queue_wordpiece_pair_after_seed2.sh.

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/wordpiece_uppercase_16L_1280_b2_no_gelu_cuda.pt \
    --mode continuous \
    --tokenizer wordpiece \
    --vocab_size 32000 \
    --tokenizer_from pt/wordpiece_uppercase_16L_1280_b2_cuda_meta.pkl \
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
    --save_interval 10000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 42 \
    --no_gelu
