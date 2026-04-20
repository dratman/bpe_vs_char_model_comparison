#!/bin/zsh
#
# train_bpe_32k_bf16.sh - BPE training with 32K vocab, bfloat16, block=2048
#
# Changes from previous runs:
#   - vocab_size 32000 (was 8192) — better information density per token
#   - precision bfloat16 (was float32) — halves memory, enables larger batch
#   - block_size 2048 (was 1024) — longer context for document-shuffled corpus
#   - batch_size 16 (restored from 4) — stable gradients at lr=0.0003
#   - max_iters 200000 (with 32K vocab, fewer tokens per char, ~same data coverage)
#
# Corpus: document-shuffled, English-only, <|endoftext|> separators
#

sh/train.sh \
    --input txt_local/corpus_books_shuffled_2026_04_18.txt \
    --output pt/bpe_16L16H.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 32000 \
    --n_layer 16 \
    --n_head 16 \
    --n_embd 2048 \
    --block_size 2048 \
    --batch_size 4 \
    --max_iters 400000 \
    --learning_rate .00015 \
    --warmup_iters 500 \
    --val_split 0.1 \
    --dropout 0.0 \
    --precision bfloat16
