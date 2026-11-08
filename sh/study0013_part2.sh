#!/bin/bash
# coupler-queue 0013 part 2: 500-step checks of the fixes, both tokenizers (char context 1003, probe v2).
#   A = scheduled snapshots; C = snapshots at 0/250/500 (same seeds -> val-loss repeat + same-step reload check)
cd /home/owner/bpe_vs_char_model_comparison
PY=$PWD/.venv/bin/python3
R=pt/study0013/part2
mkdir -p $R
for tk in bpe char; do
  if [ $tk = bpe ]; then TF=pt/study0012/bpe8192.pkl; BS=256; else TF=pt/study0012/char.pkl; BS=1003; fi
  for run in A C; do
    if [ $run = A ]; then SN=doc/0012_snapshot_steps.txt; else SN=pt/study0012/snaps_consistency.txt; fi
    $PY -u py/train.py --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt --mode continuous \
      --tokenizer $tk --tokenizer_from $TF --heldout_frac 0.01 --study_cache_dir pt/study0012 \
      --n_layer 8 --n_head 8 --n_embd 512 --block_size $BS --batch_size 32 \
      --max_iters 500 --warmup_iters 100 --learning_rate 0.001 \
      --eval_interval 250 --eval_iters 50 --log_interval 100 --sample_interval 100000000 --save_interval 100000000 \
      --precision bfloat16 --fp32_attention --no_fused --init_seed 1 --data_seed 1 \
      --snapshot_steps $SN --probe_batch doc/0013_probe_batch_v2.json --probe_interval 250 \
      --final_weights_only --output $R/${run}_$tk/model.pt > $R/${run}_$tk.log 2>&1
  done
done
echo PART2_DONE > $R/DONE
