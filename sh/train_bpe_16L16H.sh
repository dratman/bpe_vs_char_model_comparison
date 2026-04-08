#!/bin/zsh
#
# train_bpe_16L16H.sh - BPE training run matching Model B's architecture
#
# Same hyperparameters as the character-level Model B (16L/16H/2048embd)
# but with BPE tokenization (8192 vocab) instead of character-level (52 vocab)
#

sh/train.sh \
    --input txt_local/corpus_of_gutenberg_novels_cleaned_shuffled_2026_03_03_B.txt \
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
