#!/bin/zsh
#
# train_bpe_uppercase_16L_1280_b2_resumed.sh - Resume the stopped BPE run
# from its iter-145000 intermediate checkpoint and continue to the
# originally-planned 220,000 iterations.
#
# Background: the original launch (sh/train_bpe_uppercase_16L_1280_b2.sh,
# 2026-05-09) ran to iter 145,100 and was stopped 2026-05-20 because
# validation loss had plateaued for 13 consecutive evals past the iter-132K
# best (val 3.3657). This resumed run continues the same training to its
# originally-planned end-of-cosine point at iter 220,000, to observe how
# the loss curve and sample quality evolve in the "overtrained" regime.
#
# All hyperparameters are identical to the original launch. Only --output
# and --resume differ:
#   --output:  pt/bpe_uppercase_16L_1280_b2_resumed.pt  (new base — all
#              new-run artifacts live under this prefix so the original
#              run's records are preserved untouched)
#   --resume:  pt/bpe_uppercase_16L_1280_b2_iter145000.pt  (source; read-
#              only during this run)
#
# Files the resumed run reads but does not modify:
#   pt/bpe_uppercase_16L_1280_b2_iter145000.pt          (model + optimizer state)
#   pt/bpe_uppercase_16L_1280_b2_meta.pkl               (tokenizer)
#
# Files the resumed run writes (all with the _resumed prefix):
#   pt/bpe_uppercase_16L_1280_b2_resumed.pt             (best-val of this run)
#   pt/bpe_uppercase_16L_1280_b2_resumed_rolling.pt     (atomic-rename rolling save)
#   pt/bpe_uppercase_16L_1280_b2_resumed_tokens.pt      (tokenized-corpus cache)
#   pt/bpe_uppercase_16L_1280_b2_resumed_iter<N>.pt     (intermediate, every 5000 iters)
#   pt/bpe_uppercase_16L_1280_b2_resumed_final.pt       (end-of-run)
#   terminal_logs/terminal_log_for_bpe_uppercase_16L_1280_b2_resumed_<timestamp>.txt
#
# LR continuation: with --max_iters 220000 unchanged, the cosine schedule
# resumed at iter 145K continues exactly the curve the original would have
# followed — LR ~3.3e-5 at iter 145K, decaying to floor 1.06e-5 at 220K.
#
# Wall-time projection: ~6.6 sec/iter × 75K iters ≈ 5.7 days.

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/bpe_uppercase_16L_1280_b2_resumed.pt \
    --resume pt/bpe_uppercase_16L_1280_b2_iter145000.pt \
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
