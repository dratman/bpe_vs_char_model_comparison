# Diary 107 — Memorizing Green Eggs and Ham: greedy decoding reveals it, capacity limits it

Date: 2026-06-26

## Where this came from

The M2 MacBook had just been reconfigured (home folder renamed
`RalphDratman` → `RalphDratman_1`, PyTorch reinstalled, repo paths made
machine-agnostic). Before trusting it for real work we wanted to confirm the
whole pipeline runs on this machine — tokenize → train on the Apple GPU (MPS) →
checkpoint → reload → sample. The smallest possible corpus was ideal, so we
used `txt_local/Green_Eggs_And_Ham_definitive_1b.txt` (3,375 characters, the
Dr. Seuss text). What started as a smoke test turned into a small but clean
study of *verbatim memorization* in a tiny character model.

All four runs are character-level, continuous mode, on MPS, seed 1337. Models
are toy-sized (2–4 layers, 128-dim; 0.1–0.8M params). Checkpoints live in
`pt/green_eggs_*` on the MacBook (gitignored).

## The four runs

| Run | Config | Iters | Train loss | Result (greedy, from line 1) |
|---|---|---|---|---|
| `green_eggs_smoketest` | 2L/4H/128, block 64 | 300 | 0.30 | pipeline works; text still garbled |
| `green_eggs_memorize` | 4L/4H/128, block 128 | 2000 | 0.043 | verbatim locally; 18% of book (608 chars) then collision |
| `green_eggs_memorize_bs256` | 4L/4H/128, block 256 | 3000 | 0.021 | **90.1%** of book (3040 chars) then breaks |
| `green_eggs_memorize_bs256_long` | 4L/4H/128, block 256 | 5000 | 0.020 | still 90.1%, **identical** break point |

## Finding 1 — memorization is real; stochastic sampling *hides* it

The smoke test's in-training samples looked like noise ("I do not like gred hem,
youlin..."). That is **not** a memorization failure — it is a decoding artifact.
`train.py` samples at temperature 0.8 with top-k 40, so even a perfectly
memorized model emits garble because it is randomly sampling the distribution
instead of taking the argmax. Decoding **greedily** (temperature 0) on the same
checkpoints reveals long verbatim stretches. The strongest evidence: the model
reproduced the book's own typo — line 18's "Would you like *then* with a mouse"
(*then*, not *them*) — character-for-character. Lesson worth remembering:
to test what a model has memorized, decode greedily; never judge memorization
from temperature-0.8 samples.

## Finding 2 — block_size 128: a genuine context collision

Greedy from the first line reproduced 608 chars (18%), then jumped from the
"box" stanza into the "A train!" stanza. This was a real collision: the exact
128-character window ending "...I do not like them, Sam-I-Am.\n\n" occurs twice
in the book (char 480 → "Would you eat them in a box?"; char 1220 → "A train! A
train!..."). Identical 128-char context, different required continuations —
**information-theoretically impossible** for any 128-context model to satisfy
both from free-running greedy. Widening the window is the correct fix.

## Finding 3 — block_size 256: 90%, then an *under-memorization* gap (not a collision)

block_size 256 pushed the first divergence from char 608 to char 3040 (90.1%) —
it recites almost the entire book, breaking only at "...And in the dark. And ▮"
where it should say "**on** a train" but instead produces "**y**ou them...". We
probed char 3040 three ways:

- **Teacher-forced readout (full true context):** the model is *confidently
  wrong* — P('y') = 0.89, P(correct 'o') ≈ 0.0000. Not a near-tie.
- **Far-context ablation** (blank the far prefix, hold absolute positions
  fixed): P('o') never recovers. The failure is local and total.
- **Suffix-collision scan (objective text fact):** the last 16, 32, ... 256
  characters before char 3040 are **unique** in the book and always precede
  'o'. There is **no** collision at any context length.

So unlike char 608, the context here is unique and sufficient — the model simply
failed to memorize this one transition. This refuted our first guess (we
predicted a collision; the probes disproved it).

Note on method: this model uses **learned absolute position embeddings**
(`wpe`), no RoPE/ALiBi, so there is no architectural recency bias. Any reliance
on recent context is learned, not structural — which is why we measured it by
ablation rather than assuming a short "effective context."

## Finding 4 — training longer did NOT close the gap

Prediction: more training would close the under-memorized spot. Test: same
architecture, 3000 → 5000 iters. Result: train loss moved only 0.0214 → 0.0204
and the model broke at the **exact same character 3040** with the same wrong
'y'. So "just train longer" is refuted too — this is a stable wrong minimum, a
**capacity/representation limit**, not a matter of more gradient steps.

Mechanism: the book's finale packs in "And I would/will eat them {in the rain /
in the dark / on a train / in a car / in a tree}" — near-duplicates of the many
earlier *negative* lines ("I will not eat them...", "Not in the dark. Not on a
train."). At "And ", the model is grooved by the far more frequent patterns,
mis-fires to "And you" (from "And you may"), and derails into the well-worn
negative loop, never reaching the unique ending. The correct low-frequency
continuation loses to the high-frequency look-alikes; the model collapses to
the average. A handful of confidently-wrong characters barely move the 3,375-
char average loss (0.02), so the low training loss *hid* this spot.

## Two rules of thumb this produced

1. **Greedy to test memorization; temperature to test fluency.** They answer
   different questions.
2. **Diagnose the break before prescribing.** A verbatim-recitation failure can
   be (a) a true context collision — fix with wider context — or (b) an
   under-memorized transition against high-frequency distractors — wider context
   and longer training both fail; it needs more capacity. The probes
   (teacher-forced readout + suffix-collision scan) distinguish them.

## Side outcome — generation code unified

This work exercised both generation paths and exposed that the project had two
independent sampling loops (`GPT.generate()` used by train.py, and
`generate_local`/`generate_batched` in sample.py). They were unified onto a
single canonical `GPT.generate()` (commit `bd31cf6`), verified bit-identical on
a seed-1337 training run. See the HANDOFF "CODE NOTE (2026-06-26)".

## Status

Toy experiment, complete. Not a research result about the production models —
its value is methodological (the greedy-vs-temperature and
collision-vs-capacity distinctions) and as a confidence check that training
works end-to-end on the reconfigured M2 MacBook.
