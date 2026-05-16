# Diary 092 — Two-machine operations, plot infrastructure, and macOS memory accounting

Date: 2026-05-09 / 2026-05-15

## Where the trainings stand

Six days into the comparison set up in diary 091. Snapshot at
2026-05-15 21:14:

| run | machine | iter | epoch | best val loss | per-character |
|---|---|---:|---:|---:|---:|
| char_uppercase_16L_1280 | Studio | 140,900 / 500,000 | 2.03 | 0.8076 (iter 138K) | 0.808 |
| bpe_uppercase_16L_1280_b2 | M3 laptop | (M3 currently asleep; last mirrored snapshot ~iter 7,800) | — | — | — |

The Studio char run has been entirely uneventful. Loss has fallen from
~4.5 at iter 0 to 0.81 at iter 140K. Train and validation losses are
tracking together to within 0.03 the whole way. Speed steady at
4.18 sec/iter, no thermal events, no out-of-memory incidents.

The M3 BPE run is harder to inspect from here. SSH from Studio to M3
times out when the M3's display is asleep and network sleep is
engaged — the network goes quiet to save power. Waking the M3 locally
brings it back. Worth knowing: this is not a crash or a hang, just a
sleep mode.

## A two-machine workflow that works

The setup that emerged for running two trainings on two machines
turned out to be small and clean:

1. Both machines have the same git repository at GitHub
   `dratman/bpe_vs_char_model_comparison`.
2. The Studio rsyncs the corpus (gitignored, 1.27 GB) to the M3 over
   SSH at setup time. MD5 verified on both sides.
3. SSH key authentication Studio → M3 means the Studio can
   non-interactively pull the M3's training log via rsync, run scripts,
   or check status. Took one `ssh-copy-id` invocation to set up.
4. The Studio is the orchestration point. Plot scripts on the Studio
   rsync the M3's log to Studio's `terminal_logs/` (a small file —
   tens of KB), then produce PNGs side-by-side in Studio's `plots/`.
5. Code changes flow through git in the normal way; the M3 pulls when
   needed.

A clean idiom for "run something on the M3" is:

```bash
ssh RalphDratman@192.168.1.177 "command ..."
```

For copying files:

```bash
rsync -avh --progress local/path RalphDratman@192.168.1.177:remote/path
```

Both work with no password prompt because of the key. No mounted
shares are required — the SMB share infrastructure was bypassed
entirely.

## A small `train.py` fix that should have been there all along

`main()` was holding the corpus string `text` (1.27 GB) in scope for
the entire run. It's used during data preparation, then never again,
but Python only frees a local variable when its function returns —
and `main()` doesn't return until training finishes, weeks later. On
the Studio with 192 GB this was invisible. On the M3 with 64 GB it
was driving macOS swap and the memory compressor unnecessarily. Added
`del text` after the data-prep conditional (commit `974cc43`). Trivial
fix, real consequences for future M3 runs.

## macOS memory accounting, one more time

A side trip into macOS memory accounting because the Studio's
`training-monitor` LaunchAgent kept firing false alarms today. The
script was checking `Pages free` from `vm_stat` and alerting at
< 2 GB. On a Mac Studio running heavy training, `Pages free` is
almost always near zero — and **that is normal**, not a problem.

macOS classifies RAM into several states. The relevant ones:

| state | meaning |
|---|---|
| **wired** | kernel-pinned, not pageable (kernel itself, drivers, MPS GPU memory) |
| **active** | recently used by some process |
| **inactive** | assigned but not recently used — mostly file cache |
| **free** | literally unassigned to any purpose |
| **compressor** | pages compressed in place |

The intuitive English meaning of "free memory" — *space I can use right
now* — corresponds to **free + inactive**, sometimes called
*available*. macOS pre-emptively fills "free" with file cache (which
becomes "inactive") because it's faster to re-read from RAM than from
SSD. When something needs memory, macOS evicts an inactive page at
zero cost — the original is still on disk.

So `Pages free` near zero on a hot-running Mac means **macOS is doing
its job**: every page is doing something useful. The thing to watch is
*available*. Fixed the threshold to `available_gb < 4`, which is the
same notion Activity Monitor uses for its Memory Pressure indicator.
Now the alarms only fire when there's real pressure.

A second knob — **`free + inactive` is what `vm_stat` calls
`available` only roughly**. Mac OS's `memory_pressure` command and the
Activity Monitor pressure indicator use a more sophisticated metric
that also considers swap pressure and compressor load. But for an
in-script alert, free + inactive < 4 GB is a good-enough first cut.

## Plot infrastructure now in place

Two shell-script launchers in `sh/`:

- `sh/plot_m3_bpe_snapshot.sh` — rsyncs the M3 log, then plots
- `sh/plot_studio_char_snapshot.sh` — plots the local Studio log

Both write PNGs whose names mirror their checkpoints:

| training | checkpoint | plot |
|---|---|---|
| Studio char | `pt/char_uppercase_16L_1280.pt` | `plots/char_uppercase_16L_1280_loss.png` |
| M3 BPE | `pt/bpe_uppercase_16L_1280_b2.pt` | `plots/bpe_uppercase_16L_1280_b2_loss.png` |

The underlying `py/plot_current_run.py` was generalized to take
`--log` and `--out` arguments and to infer "per BPE token" vs "per
character" y-axis from the log file name. Smoothing window scales
with data length so the smoothed train-loss curve is visible whether
training has produced 50 raw points or 5,000.

## Things to watch in the next two weeks

1. The Studio char run is approaching the **half-data point**
   (~iter 250K = epoch 3.2 = `char_high_quality`'s end-of-training
   point). Past that point we are in territory `char_high_quality`
   never explored on this architecture. The train/val gap is the key
   signal — it has been ~0.03 the whole way; the question is whether
   it widens, narrows, or flips sign as the cosine schedule decays
   the learning rate further.

2. The M3 BPE run is far behind in iter count (last seen ~7.8K vs
   140K on the Studio) but **the M3 was unreachable when this diary
   was written**. Pull the log via the snapshot script and verify it
   has been making progress.

3. Sometime past iter 200K on the Studio it may be worth doing a
   per-position analysis on the in-progress checkpoint — the
   "appalpittidax" copying mechanism (diary 088) might already be in
   place; same for the L9-H3 region locker. Cheap to check on
   `pt/char_uppercase_16L_1280.pt` once it exists; the model is large
   enough that mid-training behavior should be interpretable.

4. When both trainings finish, compute per-character loss for both
   and tabulate against `char_high_quality` as a baseline. This is
   the comparison set up in diary 091.
