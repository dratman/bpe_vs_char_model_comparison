#!/bin/bash
# coupler-queue 0012 step 4 checks: (a) snapshot-consistency runs with snapshots at 0/250/500;
# (b) data-order runs: 20 steps with different init/data seeds, compare batch hashes.
cd /home/owner/bpe_vs_char_model_comparison
PY=$PWD/.venv/bin/python3
echo "0,250,500" > pt/study0012/snaps_consistency.txt
common() { # tk root maxit snapfile init data interval
  tk=$1; if [ $tk = bpe ]; then TF=pt/study0012/bpe8192.pkl; BS=256; else TF=pt/study0012/char.pkl; BS=1024; fi
  $PY -u py/train.py --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt --mode continuous \
    --tokenizer $tk --tokenizer_from $TF --heldout_frac 0.01 --study_cache_dir pt/study0012 \
    --n_layer 8 --n_head 8 --n_embd 512 --block_size $BS --batch_size 32 \
    --max_iters $3 --warmup_iters 100 --learning_rate 0.001 \
    --eval_interval 1000 --eval_iters 2 --log_interval 100 --sample_interval 100000000 --save_interval 1000000 \
    --precision bfloat16 --fp32_attention --no_fused --init_seed $5 --data_seed $6 \
    --snapshot_steps $4 --probe_batch doc/0012_probe_batch.json --probe_interval $7 \
    --output $2/$tk/model.pt > $2_$tk.log 2>&1
}
for tk in bpe char; do
  common $tk pt/study0012/consist 500 pt/study0012/snaps_consistency.txt 1 1 250
  common $tk pt/study0012/order_i1d1 20 pt/study0012/snaps_consistency.txt 1 1 1000
  common $tk pt/study0012/order_i2d1 20 pt/study0012/snaps_consistency.txt 2 1 1000
  common $tk pt/study0012/order_i1d2 20 pt/study0012/snaps_consistency.txt 1 2 1000
done
