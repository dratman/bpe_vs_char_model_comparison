#!/bin/zsh
#
# train_bpe_16L16H_clean_corpus.sh - BPE training on cleaned 2.5 GB corpus
#
# Same hyperparameters as previous BPE run (16L/16H/2048embd/BPE 8192)
# but on the cleaned, filtered Gutenberg corpus (2026-04-15).
#

sh/train.sh \
    --input txt_local/corpus_cleaned_2026_04_15.txt \
    --output pt/bpe_16L16H.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 8192 \
    --n_layer 16 \
    --n_head 16 \
    --n_embd 2048 \
    --block_size 1024 \
    --batch_size 16 \
    --max_iters 200000 \
    --learning_rate .0003 \
    --warmup_iters 500 \
    --val_split 0.1 \
    --dropout 0.0
