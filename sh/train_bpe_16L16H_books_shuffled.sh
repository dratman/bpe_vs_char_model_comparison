#!/bin/zsh
#
# train_bpe_16L16H_books_shuffled.sh - BPE training on document-shuffled corpus
#
# Same architecture as previous runs (16L/16H/2048embd/BPE 8192)
# but with:
#   - Document-shuffled corpus (books intact, <|endoftext|> separators)
#   - block_size 2048 (doubled from 1024 for longer context)
#   - batch_size 4 (reduced from 16 to fit larger blocks in memory)
#   - max_iters 400000 (doubled to compensate for smaller batch)
#   - warmup_iters 1000 (doubled to match slower token throughput)
#

sh/train.sh \
    --input txt_local/corpus_books_shuffled_2026_04_18.txt \
    --output pt/bpe_16L16H.pt \
    --mode continuous \
    --tokenizer bpe \
    --vocab_size 8192 \
    --n_layer 16 \
    --n_head 16 \
    --n_embd 2048 \
    --block_size 2048 \
    --batch_size 4 \
    --max_iters 400000 \
    --learning_rate .0003 \
    --warmup_iters 1000 \
    --val_split 0.1 \
    --dropout 0.0
