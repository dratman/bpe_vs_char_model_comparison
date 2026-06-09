# Diary 096 — Bringing up the Linux/CUDA A6000 box: port, environment, and an ~8× speedup

Date: 2026-06-09

## What happened

The third machine — the Linux/CUDA workstation with an RTX A6000 (48 GB),
the one the Mac-era handoffs never accounted for — is now a working training
node. The no-GELU matched-LR char ablation that has been grinding for ~3 days
on the Studio at ~4.18 sec/iter is now also running here at **~0.52 sec/iter**,
roughly **8× faster**, with no model-code changes.

This entry records how the box came up, because the next instance that touches
it should not have to rediscover any of it.

## The box, as found

- GPU: RTX A6000, 48 GB, driver 560.35.03 (supports CUDA 12.6). No CUDA
  toolkit (`nvcc`) — irrelevant, since PyTorch wheels bundle their own runtime.
- **No Python tooling at all** beyond the system `/usr/bin/python3` (3.12.3,
  no pip, no venv, no conda). No torch. No corpora, no checkpoints — exactly
  as the `linux-a6000-workstation` memory warned.
- 393 GB free disk.

## Environment (matches the Mac convention)

Installed Miniforge into `~/miniforge3` (no sudo; conda-forge, consistent with
the Studio/M3 miniforge convention in CLAUDE.md). Created env **`bpe_char`** on
Python 3.12.13 and installed:

    pip install torch numpy --index-url https://download.pytorch.org/whl/cu124
    pip install tokenizers

- **torch 2.6.0+cu124**, numpy 2.4.4, tokenizers 0.23.1.
- Verified `torch.cuda.is_available()` → True, sees the A6000 (capability 8.6),
  **bf16 supported** (essential — training uses `--precision bfloat16`), and a
  real bf16 matmul ran on-device.
- Activate with: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate bpe_char`.

**Gotcha:** `py/tokenizer.py` does an *unconditional* top-level
`from tokenizers import Tokenizer`, so the HuggingFace `tokenizers` package is
required even for char-only training. The first launch died on this import;
installing `tokenizers` fixed it. Anything that imports `train.py`/`sample.py`
on a fresh env needs `tokenizers` present.

## The port

`py/train.py` already auto-selects the device (`cuda → mps → cpu`, train.py:661)
and handles bf16 on CUDA correctly (train.py:682), so **no Python changes were
needed**. Only the shell wrappers were macOS-specific (`#!/bin/zsh`, bare
`python`). Two new files:

- **`sh/train_cuda.sh`** — bash port of `sh/train.sh`. Runs the conda-env
  interpreter via an absolute path (`$HOME/miniforge3/envs/bpe_char/bin/python`,
  overridable with `PYTHON=...`), so it works without activating the env. Unlike
  `train.sh` it backgrounds the run with `nohup`+`disown` and **does not
  `tail -f`** — the tail-follow is what left orphaned wrapper processes holding
  a static log on the Macs (the recurring SIGINT-vs-SIGTERM cleanup pain). The
  wrapper prints the PID and log path and returns.
- **`sh/train_char_uppercase_16L_1280_no_gelu_matched_LR_trial_CUDA.sh`** —
  bash, calls `train_cuda.sh`, hyperparameters *identical* to the macOS trial
  (batch_size=4, lr=1.5e-4, warmup 2000, block 4096, bf16) so the comparison
  stays valid. Output is suffixed `_cuda`
  (`pt/char_uppercase_16L_1280_no_gelu_matched_lr_cuda.pt`) to keep this box's
  checkpoints distinct from the identically-named Studio run.
  - NB: batch_size=4 was an M3/Studio MPS memory limit, not a CUDA one. The
    A6000 has room for much larger batches, but raising it would change the LR
    regime and break the matched comparison, so it's deliberately left at 4.

The corpus (`corpus_high_quality_uppercase_2026_05_08.txt`, 1.27 GB) was rsynced
from the Studio over the LAN (SSH from this box was authorized this session —
see HANDOFF). Byte-exact size match after transfer.

## The payoff: speed

| machine | sec/iter | iters/sec | MFU | est. 500K-iter wall time |
|---|---:|---:|---:|---:|
| Mac Studio (M3, MPS) | ~4.18 | ~0.24 | ~3.5% | ~24 days |
| **Linux A6000 (CUDA)** | **~0.52** | **~1.9** | **~29%** | **~3 days** |

(iter-100 step time 522 ms; iter 0 took 36 s for CUDA kernel compile/warmup,
which drags the first averaged "1.12 it/s" reading.) MFU jumping from ~3.5% on
MPS to ~29% on CUDA is the real story — the A6000 is not just faster silicon,
it's far better *utilized* by the PyTorch CUDA path than the M3 is by MPS.
Loss fell 4.30 → 2.50 by iter 100, the normal early trajectory.

Thermals: the GPU sat at ~55°C during the CPU-bound tokenization phase (0–2%
util, ~27 W — essentially idle), then heat-soaked to **~86°C** within ~2 minutes
once training compute started (100% util, ~298 W, clocks holding at the 1560 MHz
boost). The eighties is the true under-load temperature for this card; the 55°C
reading was idle and should not be mistaken for a loaded one. 86°C at near-cap
power is warm but not throttling (throttle ~93°C). Worth watching case airflow
on a multi-day run.

## Significance

The practical iteration speed of this whole research program just changed by
~8×. A full 500K-iter char run that costs ~24 days on the Studio costs ~3 days
here. This makes previously-impractical experiments (full val-floor runs of
ablations, the LARQL/gated-FFN model, larger-corpus scaling) tractable on a
human timescale. The Studio and M3 remain useful, but the A6000 is now the
fast path.

Caveat worth stating plainly: the run launched here is a **duplicate** of the
live Studio no-GELU matched-LR run (same hyperparameters, faster hardware). If
the goal is just to get the no-GELU val floor, the A6000 will reach it first and
the Studio run can be retired; if the goal is cross-hardware reproducibility,
keeping both is the point. That decision is open.

## Throughput sweep (batch size)

Follow-up benchmark to find the A6000's real ceiling, since results don't matter
here — only speed does. `py/benchmark_throughput.py` times a faithful training
step (zero_grad → autocast forward → backward → clip_grad_norm_(1.0) → step) on
synthetic on-GPU batches, building the model once and reusing it across batch
sizes. The matched-LR run was stopped at iter 12,500 (still a rock-steady
1.86 it/s after ~2 h) to free the GPU for this. Fidelity gate: batch 4
reproduced **502 ms/iter** (vs the real run's 535 ms) and 30.3% A100-MFU (vs the
logged ~29%), so the synthetic step tracks the real one.

| batch | tokens/iter | ms/iter | tokens/sec | MFU(A100) | MFU(A6000) | peak mem |
|--:|--:|--:|--:|--:|--:|--:|
| 4  | 16,384 |   502 | 32,632 | 30.3% | 61.0% | 14.3 GB |
| 8  | 32,768 |   980 | 33,427 | 31.0% | 62.5% | 24.0 GB |
| 16 | 65,536 | 1,946 | 33,682 | 31.3% | 62.9% | 43.4 GB |
| 32 | —      | —     | —      | —     | —     | **OOM** |

Findings:
- **Throughput is nearly batch-independent here.** tokens/sec rises only +3.2%
  from batch 4 → 16. At block 4096 the sequence dimension already saturates the
  tensor cores, so the MPS-era batch=4 was *not* costing throughput on CUDA.
- **Memory, not compute, is the cap.** ~2.4 GB per batch-unit over a ~4.8 GB
  base; batch 16 fills 43 of 48 GB, batch 32 OOMs. Max practical batch ~16–18.
- **Real utilization is ~61%, not 29%.** `estimate_mfu()` normalizes to an
  A100's 312 TFLOPS; against the A6000's own ~155 TFLOPS dense bf16 peak this
  workload runs at ~61–63% — genuinely well utilized.
- The 7.8× cross-hardware speedup (diary table above) holds and is slightly
  conservative — 8.6× at the A6000's best batch (16).

Practical upshot: keep batch 4 for the matched comparison — raising batch buys
no throughput on this card, only changes optimization dynamics (and would need
LR retuning). Memory, not speed, is what would force batch choices here.

## For the next instance

- Env: `conda activate bpe_char` (torch 2.6.0+cu124). `tokenizers` is a hard
  import dep even for char runs.
- Launch char training: `sh/train_char_uppercase_16L_1280_no_gelu_matched_LR_trial_CUDA.sh`.
- Generic launcher: `sh/train_cuda.sh <train.py args>` — prints PID + log, no tail.
- Stop a run: `kill -TERM <PID>` (SIGTERM; train.py exits cleanly, rolling
  checkpoint loses at most `eval_interval` iters).
- Corpora/checkpoints are NOT in git (gitignored, large). Pull corpora from the
  Studio via rsync as done here.
- Throughput benchmark: `python py/benchmark_throughput.py --batches 4,8,16,32`
  (needs the conda env). Faithful training-step timing, no data/checkpoints.
