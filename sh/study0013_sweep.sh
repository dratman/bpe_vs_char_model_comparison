#!/bin/bash
# coupler-queue 0013 part 3: shared-grid learning-rate sweep, 20 runs one at a time.
# Re-runnable: a run whose run_summary.json exists is skipped. Touch pt/study0013/sweep/STOP to stop between runs.
cd /home/owner/bpe_vs_char_model_comparison
PY=$PWD/.venv/bin/python3
R=pt/study0013/sweep
mkdir -p $R
for tk in bpe char; do
  if [ $tk = bpe ]; then TF=pt/study0012/bpe8192.pkl; BS=256; else TF=pt/study0012/char.pkl; BS=1003; fi
  for len in 3050 12200; do
    for lr in 0.00025 0.0005 0.001 0.002 0.004; do
      [ -f $R/STOP ] && { echo "STOP file seen"; exit 0; }
      name=${tk}_${len}_${lr}; d=$R/$name
      [ -f $d/run_summary.json ] && continue
      mkdir -p $d
      t0=$(date +%s)
      $PY -u py/train.py --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt --mode continuous \
        --tokenizer $tk --tokenizer_from $TF --heldout_frac 0.01 --study_cache_dir pt/study0012 \
        --n_layer 8 --n_head 8 --n_embd 512 --block_size $BS --batch_size 32 \
        --max_iters $len --warmup_iters 610 --learning_rate $lr \
        --eval_interval 250 --eval_iters 50 --log_interval 100000000 --sample_interval 100000000 --save_interval 100000000 \
        --precision bfloat16 --fp32_attention --no_fused --init_seed 1 --data_seed 1 \
        --probe_batch doc/0013_probe_batch_v2.json --probe_interval 250 \
        --diverge_margin 0.5 --final_weights_only --accept_overrides $d/overrides.json \
        --output $d/model.pt > $R/$name.log 2>&1
      t1=$(date +%s)
      echo "$name wall_seconds=$((t1-t0)) exit=$?" >> $R/runs.log
    done
  done
done
echo SWEEP_DONE >> $R/runs.log
