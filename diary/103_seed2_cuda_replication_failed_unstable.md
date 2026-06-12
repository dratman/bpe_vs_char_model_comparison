# Diary 103 — Seed-2 CUDA replication failed: training instability, not a seed effect

Date: 2026-06-12

## What happened

The seed-2 baseline replication (diary 099 step 2a) launched on the
A6000 on 2026-06-11 09:31 with hyperparameters byte-identical to the
Studio baseline (16L/8H/1280, block 4096, batch 4, lr 1.5e-4,
bfloat16) plus `--seed 2`. It was killed (SIGTERM) on 2026-06-12
~16:12 at iter ~188K, epoch 2.7, after status checks showed it was
not training properly.

Evidence (log `terminal_log_for_char_uppercase_16L_1280_seed2_cuda_
2026_06_11_0931.txt` on the Linux box):

| iter | seed-2 CUDA val | Studio baseline val |
|------|-----------------|---------------------|
| 20,000 | 2.2283 | 0.9648 |
| 40,000 | 1.2518 | 0.8864 |
| 80,000 | 2.1067 | 0.8264 |
| 120,000 | 1.6029 | 0.8231 |
| 160,000 | 1.5629 | 0.7994 |
| 188,000 | 1.4999 | (~0.79) |

The trajectory is non-monotonic — descents punctuated by collapses
(1.25 → 2.11 between 40K and 80K) and a long plateau near 1.5. No
NaN ever appeared in the log (the three "nan" grep hits are letter
sequences inside garbled samples). Samples at iter 180K are broken
orthography ("l qbeeing todilgity rais ording-"), worse than the
run's own iter-10K samples — quality *degraded* with training.

## Reading

This is not seed noise; it is optimization instability. The identical
configuration ran 500K iters on the Studio (MPS, plain bfloat16) with
no incidents. The CUDA path differs in one visible way: train.py uses
**AMP autocast** on CUDA ("Using automatic mixed precision (AMP) with
bfloat16"), versus the non-autocast bf16 path on MPS. The June-9
no-GELU CUDA trial was stable for its 12.5K iters at the same LR, but
the seed-2 run was also near-normal through ~iter 10K — instability
onset came later, so that trial's 12.5K iters certify nothing.

So the cheap A6000 replication path has a real numerics/stability
difference from the Studio path that must be solved before any
cross-backend comparison can be trusted. This is itself a useful
methodological data point: same code, same config, same data, same
seed semantics — different backend, qualitatively different training.

## Disposition

- Run killed; its log is preserved (committed). Its ~80 GB of
  checkpoints in the Linux box's `pt/` are garbage-model snapshots —
  proposed for deletion (pending Ralph's OK), keeping none.
- The kill released the GPU to the queued WordPiece pair (diary 102),
  which launched automatically at 16:15 and is healthy at startup.
  NOTE: the pair runs at lr 1.06e-4 batch 2 on the same CUDA/AMP
  path — its loss curve needs watching for the same instability
  signature. Its internal control/ablation comparison is unaffected
  (both halves share whatever backend numerics exist), but
  cross-machine comparisons to the M3 BPE run inherit the caveat.
- Seed-2 replication attempt #2: re-queue on the A6000 after the
  WordPiece pair, with a stability fix. Candidate fixes, in order:
  (a) disable autocast on CUDA and run the same plain-bf16 path MPS
  uses; (b) full fp32 (memory allows at batch 4; speed cost to
  measure); (c) keep AMP but lower LR — least preferred, since it
  breaks hyperparameter matching with the run it is supposed to
  replicate. Requires reading train.py's CUDA precision path first.

— Claude Code Fable 5
