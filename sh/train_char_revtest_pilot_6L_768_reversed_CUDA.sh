#!/bin/bash
#
# train_char_revtest_pilot_6L_768_reversed_CUDA.sh
#
# coupler-queue item 0002 — reversed-text learnability pilot, REVERSED half.
#
# Partner of sh/train_char_revtest_pilot_6L_768_forward_CUDA.sh. IDENTICAL config
# (6L/6H/768, block 256, batch 64, lr 5e-4 cosine->5e-5 over 10K iters, warmup
# 200, --seed 1337). The ONLY difference is --input: this trains on the corpus
# reversed character-by-character WITHIN each split.
#
# The reversed corpus was built and VERIFIED by py/make_reversed_corpus.py:
#   reversed_text = reverse(forward_train_part) ++ reverse(forward_val_part)
# so re-applying train.py's positional --val_split 0.1 reproduces exactly
# reverse(train) and reverse(val). The character set is identical to the forward
# corpus, so the char vocab/meta is identical and validation cross-entropy is
# directly comparable between the two runs. Report val loss in BITS PER CHARACTER
# = nats / ln(2). For samples, reverse the generated text back to normal reading
# order before eyeballing.
#
# Output: pt/char_revtest_pilot_6L_768_reversed_cuda.pt (+ _rolling/_tokens/_final).
# Stop:   kill -TERM <PID> (auto-terminates at max_iters anyway).

# Idempotency / double-run guard: if this pilot already COMPLETED (train.py writes
# _final.pt only at max_iters), do nothing. This self-neutralizes the armed queue
# runner (queue_revtest_pilot_after_seed2.sh) so it cannot re-launch a finished
# pilot when the seed-2 trial eventually exits days from now.
if [ -f pt/char_revtest_pilot_6L_768_reversed_cuda_final.pt ]; then
    echo "reversed pilot already completed (_final.pt exists); skipping launch."
    exit 0
fi

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08_REVERSED_within_splits.txt \
    --output pt/char_revtest_pilot_6L_768_reversed_cuda.pt \
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
