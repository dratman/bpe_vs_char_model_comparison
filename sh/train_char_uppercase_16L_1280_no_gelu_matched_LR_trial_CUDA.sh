#!/bin/bash
#
# train_char_uppercase_16L_1280_no_gelu_matched_LR_trial_CUDA.sh
#
# CUDA/Linux (RTX A6000) port of
# train_char_uppercase_16L_1280_no_gelu_matched_LR_trial.sh.
#
# Same experiment as the macOS version: the no-GELU MLP ablation with the
# cosine LR schedule matched to the baseline char run (max_iters=500000 so the
# LR holds near its 1.5e-4 peak over the first ~10K iters, instead of decaying
# fast as the original max_iters=10000 trial did). Over the first 10K iters this
# gives a clean no-GELU-vs-baseline comparison; left running it finds the
# no-GELU val-loss floor under a proper schedule.
#
# Differences from the macOS trial script:
#   - #!/bin/bash and calls sh/train_cuda.sh (not sh/train.sh) — runs the
#     bpe_char conda-env python on the A6000; py/train.py auto-selects CUDA.
#   - --output suffixed with _cuda so this box's checkpoints/logs stay distinct
#     from the (identically-configured) run live on the Mac Studio.
#   - Hyperparameters are otherwise IDENTICAL to the macOS trial (batch_size=4,
#     lr=1.5e-4, warmup 2000, block 4096, bf16) so the comparison stays valid.
#     NOTE: batch_size=4 was chosen for the M3/Studio MPS memory limit; the
#     A6000 (48 GB) could run larger batches, but that would change the LR
#     regime and break the matched comparison, so it is left at 4.
#
# Output: pt/char_uppercase_16L_1280_no_gelu_matched_lr_cuda.pt (best-val)
#         pt/char_uppercase_16L_1280_no_gelu_matched_lr_cuda_rolling.pt (latest, atomic)
#         pt/char_uppercase_16L_1280_no_gelu_matched_lr_cuda_tokens.pt (tokens cache)
#         pt/char_uppercase_16L_1280_no_gelu_matched_lr_cuda_final.pt (last iter, if completed)
#
# Stopping: kill -TERM <PID> (the wrapper prints the PID on launch).

sh/train_cuda.sh \
    --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
    --output pt/char_uppercase_16L_1280_no_gelu_matched_lr_cuda.pt \
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
