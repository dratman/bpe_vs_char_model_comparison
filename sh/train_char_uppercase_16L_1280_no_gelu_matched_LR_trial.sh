#!/bin/zsh
#
# train_char_uppercase_16L_1280_no_gelu_matched_LR_trial.sh
#
# Re-run of the no-GELU ablation with the cosine LR schedule matched
# to the baseline char run. The earlier trial
# (train_char_uppercase_16L_1280_no_gelu_trial.sh) set max_iters=10000,
# which made the cosine endpoint also 10K — so LR decayed from
# 1.5e-4 at iter 2K down to 3.48e-5 by iter 8K, while baseline kept
# its LR near 1.5e-4 throughout (baseline used max_iters=500K). That
# schedule mismatch confounded the iter-8K val-loss comparison: most
# of the 0.78 gap (no_gelu 1.84 vs baseline 1.06 at iter 8K) couldn't
# be attributed cleanly to "no GELU" vs "starved of LR".
#
# This run uses max_iters=500000, identical to baseline. Over the
# first 10K iters the LR will hold near-peak (1.5e-4) — same regime
# as the baseline trace at iters 2K-10K. After we have the comparison
# data (~10K iters, ~12 hours on Studio MPS), the run can be stopped
# early. If left running for the full 500K iters (~24 days), it will
# also tell us the no-GELU architecture's val-loss floor under a
# proper schedule, for comparison against baseline's 0.7152 final.
#
# Differences from the earlier no_gelu_trial.sh:
#   --max_iters: 10000 -> 500000
#   --output:    no_gelu.pt -> no_gelu_matched_lr.pt  (new path,
#                                so the iter-10K trial's checkpoints
#                                and tokens cache are preserved)
#
# Stopping: when ready, send SIGTERM to the python process (not
# SIGINT; train.sh backgrounds python, which inherits SIG_IGN for
# SIGINT). See HANDOFF "LESSONS FROM THIS SESSION" entry on
# "macOS background-launched processes inherit SIGINT=SIG_IGN".
# Atomic rolling-save means losing at most eval_interval=2000 iters
# of work to the stop.
#
# Output: pt/char_uppercase_16L_1280_no_gelu_matched_lr.pt (best-val)
#         pt/char_uppercase_16L_1280_no_gelu_matched_lr_rolling.pt (latest, atomic)
#         pt/char_uppercase_16L_1280_no_gelu_matched_lr_tokens.pt (tokens cache)
#         pt/char_uppercase_16L_1280_no_gelu_matched_lr_final.pt (last iter, only if completed)

sh/train.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280_no_gelu_matched_lr.pt \
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
    --save_interval 5000 \
    --log_interval 100 \
    --sample_interval 5000 \
    --val_split 0.1 \
    --dropout 0.0 \
    --no_gelu
