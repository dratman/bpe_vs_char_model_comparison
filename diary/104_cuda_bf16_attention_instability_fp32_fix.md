# 104 — CUDA bf16 attention instability: diary-103 verdict overturned, fp32-attention fix, + ops changes

**Date:** 2026-06-21 (linux-cuda / A6000 worker session)

## Headline

The 16L/1280 char "big config" does **not** train on the A6000 (CUDA) under plain
bf16: val descends to ~1.2 nats early (~iter 12K) then **diverges back up to a ~2.4
plateau** and never approaches the forward floor (0.7152). The cause is **bf16 in the
attention path at the long 4096 block**, NOT fused AdamW. **Diary 103's follow-up
verdict ("fused AdamW is the cause; `--no_fused` fixes it") is overturned.** Forcing
just the attention math to fp32 fixes it.

## How the diary-103 verdict fell

Diary 103 (seed-2 CUDA replication failure) was followed up with a code-level
diagnosis that fingered fused AdamW and prescribed `--no_fused`. That call was a
**false positive** — it judged the `--no_fused` run "smooth" at iter ~74K (val 2.37)
when the run had in fact already plateaued high. Run to completion, the seed-2
`--no_fused` trial sat at val **~2.40–2.53 for its entire 500K** (min 2.4031,
epochs 1→7). And the coupler-queue 0003 reversed full run (also `--no_fused`) failed
identically (val 1.20 @12K → ~2.4 plateau). So **both fused and non-fused diverge** —
the optimizer flag was never the variable.

## Scope (what isolates the bug)

- The **0002 pilot** (6L/768, block 256, batch 64) trained fine on this same A6000.
- The **forward 16L/1280 baseline** trained fine to 0.7152 — but on the **Studio (MPS)**.
- ⇒ failure is specific to **big config (16L/1280, block 4096, batch 4) + CUDA bf16**.
- Failure *shape* ("descend to 1.2, blow back up to 2.4") is a numerical blowup, not a
  stuck optimizer or low-LR — pointing at bf16 in the attention logits / softmax at the
  long block.

## The fp32 diagnostic (precision confirmed as cause)

Ran the big config in **full fp32** (`--precision float32` → train.py uses
`nullcontext`, no autocast) on the forward corpus, `max_iters=500000` so the 0–20K LR
matched the known-good MPS forward curve (avoids the fast-LR-decay confound). Result:
val descended **monotonically to ~0.97 nats by iter ~15K with NO divergence** — it
sailed straight through the ~1.2 region where every bf16 run turned around. (The diag
process then died on a disk-full checkpoint save, but its job was already done.)

## The fix: fp32 attention only (production recipe)

Full fp32 is ~6× slower (~19 days for 500K), so the production recipe keeps **bf16
everywhere except the attention core, which runs in fp32**:

- `model.py`: new `GPTConfig.fp32_attention`. In `CausalSelfAttention.forward`, when
  set, the QK^T / softmax / SDPA run inside `torch.autocast(..., enabled=False)` on
  fp32 q/k/v, then cast back to the stream dtype. Rest of the net stays bf16.
- `train.py`: `--fp32_attention` flag threaded into `model_args`. Smoke-tested
  fwd+bwd OK on CUDA under bf16 autocast.

**Reversed 0003 re-run** (`sh/train_char_uppercase_16L_1280_reversed_fp32attn_CUDA.sh`,
bf16 + `--fp32_attention`, `--no_fused`, seed 1337): **stable** — val 2.39(2K) →
1.29(4K) → 1.00(12K) → 0.95(18K) → **0.91 @32K**, monotone, no divergence. The fix
carries the full run.

**Caveat — speed:** fp32 attention at block 4096 can't use the fast bf16 flash kernel,
so throughput is only ~0.3–0.6 it/s (vs bf16's ~1.8) → ETA ~9–17 days, not the hoped
~4–5. Barely faster than full fp32. Open question for a future pass: a cheaper stable
recipe (e.g. fp32 softmax only, logit soft-capping, or smaller block) — or just run on
MPS where bf16 is stable.

## Consequences

- **diary-094 char-vs-BPE error bar from seed-2 is VOID** — seed-2 failed this
  instability, so its best-val is not a clean replication number. Blocked behind a
  stable CUDA recipe or a second-seed replication on MPS.
- coupler-queue 0003's first attempt is in `failed/`; the fp32-attention re-run is the
  real measurement (reversed best-val bpc vs forward 1.03 bpc) when it finishes.

## Operational changes this session

- **Disk incident:** `/` hit 100% (pt/ = 380 GB). With Ralph's OK, reclaimed ~109 GB
  (void seed-2 `_iter` ckpts, failed-0003 `_iter` ckpts, 2 dead tokens caches); all
  finals/bests/metas kept.
- **Standing policy (Ralph):** graceful override-file stops are pre-authorized — helper
  `sh/stop_run_via_override.sh` (whitelisted). Deny-list unchanged; RNG drift on resume
  accepted.
- **Heartbeat (editor task 2026-06-21):** the periodic status-wake changed from **every
  12h → every 3h** (cron `be7a0506`, fires at :13). It now also **publishes status to
  the coupler-queue repo at `status/linux-cuda.md`** (commit+push) each fire, via
  `sh/heartbeat_status.sh`, in addition to the terminal print — so the browser Editor
  can read "what's running?" without a relay.
