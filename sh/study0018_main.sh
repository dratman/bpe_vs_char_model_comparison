#!/bin/bash
# coupler-queue 0018 part 2: 7 main runs, one at a time, outputs on /data. Re-runnable (skips finished, resumes interrupted).
# Touch /data/study/0018/STOP to stop between runs.
cd /home/owner/bpe_vs_char_model_comparison
PY=$PWD/.venv/bin/python3
R=/data/study/0018
mkdir -p $R
RUNS="B1:bpe:1:1 C1:char:1:1 B2:bpe:2:1 B3:bpe:3:1 C2:char:2:1 C3:char:3:1 C4:char:1:2"
for spec in $RUNS; do
  IFS=: read name tk iseed dseed <<< "$spec"
  if [ $tk = bpe ]; then TF=pt/study0012/bpe8192.pkl; BS=256; else TF=pt/study0012/char.pkl; BS=1003; fi
  d=$R/$name
  [ -f $d/run_summary.json ] && continue
  [ -f $R/STOP ] && { echo "STOP file seen"; exit 0; }
  free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ "$free" -lt 20 ] && { echo "DISK LOW ($free GB)" >> $R/runs.log; exit 1; }
  mkdir -p $d
  RES=""; [ -f $d/model_resume.pt ] && RES="--resume $d/model_resume.pt"
  t0=$(date +%s)
  $PY -u py/train.py --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt --mode continuous \
    --tokenizer $tk --tokenizer_from $TF --heldout_frac 0.01 --study_cache_dir pt/study0012 \
    --n_layer 8 --n_head 8 --n_embd 512 --block_size $BS --batch_size 32 \
    --max_iters 61000 --warmup_iters 610 --learning_rate 0.002 \
    --eval_interval 250 --eval_iters 50 --log_interval 1000 --sample_interval 100000000 --save_interval 2000 \
    --precision bfloat16 --fp32_attention --no_fused --init_seed $iseed --data_seed $dseed \
    --snapshot_steps doc/0012_snapshot_steps.txt --probe_batch doc/0018_probe_batch_v3.json --probe_interval 250 \
    --diverge_margin 0.5 --final_weights_only --accept_overrides $d/overrides.json $RES \
    --output $d/model.pt >> $d/train.log 2>&1
  rc=$?
  echo "$name wall_seconds=$(( $(date +%s)-t0 )) exit=$rc $(date -u +%FT%TZ)" >> $R/runs.log
  if [ $name = C1 ]; then
    if ! $PY py/study0018_tools.py gate $R > $R/gate.txt 2>&1; then echo "GATE FAILED" >> $R/runs.log; exit 2; fi
    echo "GATE PASSED" >> $R/runs.log
  fi
done
echo ALL_DONE >> $R/runs.log
