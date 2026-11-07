#!/bin/bash
# coupler-queue 0012 step 4: 500-step smoke test, full shape, both tokenizers.
cd /home/owner/bpe_vs_char_model_comparison
PY=$PWD/.venv/bin/python3
for tk in bpe char; do
  if [ $tk = bpe ]; then TF=pt/study0012/bpe8192.pkl; BS=256; else TF=pt/study0012/char.pkl; BS=1024; fi
  $PY -u py/train.py --input txt_local/corpus_high_quality_uppercase_2026_05_08.txt --mode continuous \
    --tokenizer $tk --tokenizer_from $TF --heldout_frac 0.01 --study_cache_dir pt/study0012 \
    --n_layer 8 --n_head 8 --n_embd 512 --block_size $BS --batch_size 32 \
    --max_iters ${MAXIT:-500} --warmup_iters 100 --learning_rate 0.001 \
    --eval_interval 250 --eval_iters 20 --log_interval 100 --sample_interval 100000000 --save_interval 250 \
    --precision bfloat16 --fp32_attention --no_fused --init_seed 1 --data_seed 1 \
    --snapshot_steps doc/0012_snapshot_steps.txt --probe_batch doc/0012_probe_batch.json --probe_interval 250 \
    --output ${OUTROOT:-pt/study0012/smoke}/$tk/model.pt > ${OUTROOT:-pt/study0012/smoke}_$tk.log 2>&1
done
