#!/bin/bash
#
# train_wordpiece_uppercase_16L_1280_b2_CUDA.sh
#
# WordPiece-tokenized control run (GELU ON) — first half of the
# WordPiece/no-GELU pair (Ralph's experiment, 2026-06-11; diary 102).
#
# WordPiece = BERT-style tokenization: whitespace pre-tokenization, bare
# spaceless word tokens, ## continuation pieces. The top ~10-20K word
# forms become single whole-word tokens, so the lexicon largely lives in
# the tokenizer rather than having to be learned by the MLPs. Newlines
# survive via a [NL] special token (see py/tokenizer.py).
#
# Hyperparameters match the M3 BPE run (sh/train_bpe_uppercase_16L_1280_b2.sh)
# EXACTLY — vocab 32000, batch 2, lr 1.06e-4, warmup 500, 220K iters —
# so {BPE, WordPiece, WordPiece-no-GELU} form one comparable family.
# Two deliberate exceptions:
#   - save_interval 10000 (BPE used 5000): two WordPiece runs at 5K
#     would need ~380 GB of checkpoints; 10K fits both in the A6000's
#     free disk alongside the seed-2 char run's artifacts.
#   - --seed 42: the no-GELU twin uses the SAME seed, so the pair gets
#     identical init and batch order — the only difference between the
#     two runs is the GELU itself.
#
# Runs on the A6000 via sh/train_cuda.sh (prints PID; stop with
# kill -TERM <PID>). Normally launched by sh/queue_wordpiece_pair_after_seed2.sh.

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/wordpiece_uppercase_16L_1280_b2_cuda.pt \
    --mode continuous \
    --tokenizer wordpiece \
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
    --save_interval 10000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --seed 42
