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

## Follow-up (2026-06-13) — code-level diagnosis, read-only

Read train.py's full numerics path and model.py's optimizer setup. Two
corrections and a sharper suspect list.

**The instability is real, not an eval artifact.** The `Step` line's
*train* loss is also an `eval_iters=20` average (`estimate_loss_continuous`),
not a single-batch read — and it oscillates in lockstep with val
(1.31/1.33 → 2.74/2.73 → …). If this were sampling variance the two
would bounce independently; lockstep means the model's weights are
genuinely oscillating. Code is otherwise correct: a fresh random batch
is drawn every iteration (`train.py:1275`; the grad-accum refetch loop
is inert at the default `grad_accum_steps=1`), grad-clip 1.0 is applied,
and `scaler` is correctly `None` for bf16 (no GradScaler misuse).

**Correction to the "autocast" reading above.** Both backends use
`torch.amp.autocast` — `device_type='cuda'` (train.py:978) and
`device_type='mps'` (train.py:983). MPS is *not* a "plain, non-autocast
bf16 path." So autocast presence is **not** the CUDA-vs-MPS difference,
and "disable autocast to match MPS" (candidate (a) above) is based on a
misreading. The genuine code-level deltas between the two backends are:

1. **Fused AdamW (CUDA-only).** `model.py:configure_optimizers` sets
   `use_fused = fused_available and device_type == 'cuda'`, so the CUDA
   runs get `torch.optim.AdamW(..., fused=True)` (banner: "using fused
   AdamW: True") and the MPS baseline gets the non-fused implementation.
   PyTorch's fused AdamW has had numerical-correctness regressions across
   versions. **This is the prime suspect and the cheapest to test.**
2. **bf16 matmul accumulation on CUDA tensor cores** vs MPS bf16 —
   different rounding/accumulation under the same autocast wrapper.

**Triangulation across the three runs we have:**

| run | device | lr | AdamW | result |
|---|---|---|---|---|
| baseline char | MPS | 1.5e-4 | non-fused | stable |
| WordPiece (diary 102) | CUDA | 1.06e-4 | fused | stable |
| seed-2 char | CUDA | 1.5e-4 | fused | **unstable** |

It is *not* lr alone (MPS ran 1.5e-4 fine) and *not* CUDA alone
(WordPiece is stable on CUDA). The failure needs char-config + lr 1.5e-4
+ CUDA numerics together. All three of {fused AdamW, tensor-core bf16,
lr-is-marginal} are consistent with this table, so a test run is needed
to decide between them.

**Test ladder (each ~6–10K iters ≈ 1–2 h to reveal the signature;
keep `max_iters=500000` so the cosine-LR schedule matches the baseline —
do NOT shorten max_iters, or you reintroduce the LR-decay confound that
muddied the no-GELU trial):**
1. **`--no_fused`** (new flag, added this session; forces non-fused
   AdamW on CUDA), everything else byte-identical to seed-2. Stable →
   fused AdamW was the cause; cleanest outcome, preserves hyperparameter
   matching.
2. If still unstable: lr 1.06e-4 + warmup 500 (match WordPiece).
3. If still unstable: one fp32 run to rule bf16 in/out definitively.

Script `sh/train_char_uppercase_16L_1280_seed2_no_fused_CUDA.sh` and
queue runner `sh/queue_seed2_no_fused_after_wordpiece.sh` (waits on the
WordPiece-pair queue PID, then launches trial 1) prepared this session.

— Claude Code Opus 4.8 (1M context)
