#!/bin/zsh
#
# train_char_nope_pilot_6L_768_mps.sh
#
# coupler-queue item 0004 — position-encoding (PE) vs no-position-encoding (NoPE)
# pilot, on the Mac Studio (MPS).
#
# A MATCHED PAIR that differs in EXACTLY one thing: whether the model has a
# positional embedding.
#   ARM A (pe)   : standard model, learned positional embedding (wpe) ON.
#   ARM B (nope) : --no_pos_emb — no wpe parameter, token embeddings only.
#                  The causal mask is kept, so any sense of position must be
#                  recovered implicitly from the mask.
# Same seed (1337) for both. With --no_pos_emb, py/model.py still draws wpe's
# init numbers before deleting it, so every other weight starts IDENTICAL to
# ARM A's and the batch order is the same.
#
# Config = the 0002 forward pilot (sh/train_char_revtest_pilot_6L_768_forward_CUDA.sh)
# ported to the Studio:
#   6L / 6H / 768 embd (~43M params), block 256, batch 64, 10K iters,
#   LR 5e-4 cosine to 5e-5 over the full run, warmup 200.
# One deliberate change from the template: --precision float32 instead of
# bfloat16, for BOTH arms. On this Studio's torch, bf16 autocast on MPS is a
# no-op (it silently runs fp32), so float32 is what actually runs either way;
# saying so explicitly keeps the log honest. The spec allows this fallback.
#
# The arms run ONE AFTER THE OTHER (A, then B) rather than at the same time, so
# each has the whole GPU. This script calls train.py directly instead of going
# through sh/train.sh, because sh/train.sh ends in a never-ending `tail -f`,
# which would block running the second arm after the first. The log file
# naming matches sh/train.sh (terminal_logs/terminal_log_for_<output>_<time>.txt).
#
# Launch (returns immediately):
#   nohup sh/train_char_nope_pilot_6L_768_mps.sh > terminal_logs/0004_driver.log 2>&1 &
# Stop:   kill -TERM <python PID>   (each arm auto-terminates at max_iters).

cd "$(dirname "$0")/.." || exit 1
PY=$HOME/miniforge3/bin/python3

run_arm() {
    local name=$1; shift
    local out=pt/char_nope_pilot_6L_768_${name}_mps.pt
    if [[ -f pt/char_nope_pilot_6L_768_${name}_mps_final.pt ]]; then
        echo "arm $name already completed (_final.pt exists); skipping."
        return 0
    fi
    local log=terminal_logs/terminal_log_for_char_nope_pilot_6L_768_${name}_mps_$(date +"%Y_%m_%d_%H%M").txt
    echo "Command: $PY -u py/train.py ... --output $out $@" > "$log"
    echo "Started: $(date)" >> "$log"
    echo "========================================" >> "$log"
    echo "arm $name -> $log"
    $PY -u py/train.py --checkpoints_to pt \
        --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
        --output "$out" \
        --mode continuous \
        --tokenizer char \
        --precision float32 \
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
        --seed 1337 \
        "$@" >> "$log" 2>&1
    echo "arm $name exited with status $? at $(date)"
}

run_arm pe
run_arm nope --no_pos_emb
echo "both arms finished at $(date)"
