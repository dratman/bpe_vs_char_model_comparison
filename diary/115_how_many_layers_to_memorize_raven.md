# Diary 115 — How many layers does it take to memorize The Raven? (char, width 16)

Date: 2026-07-13
Prompt (Ralph): "Can you determine how many layers are actually needed?
Probably not all 6." (Follow-up to diary 114's bonus finding: char can't
memorize The Raven at 1 layer but can at 6.)

## Method

A matched depth sweep. Identical recipe at every rung — char tokenizer,
n_embd 16, block 256, batch 32, lr 1e-3, val_split 0.08, 40k iters, seed 1337,
on `txt_local/Poe_The_Raven.txt` (6226 chars, vocab 57). Only `n_layer` varies,
1..6. Each trained to a flat plateau. (L1 had to be retrained: the old
`raven_char_L1_d16` was only 3000 iters — undertrained, not comparable.)

Measured two ways:
- **train / val loss** at plateau (from the run logs). Memorizing =
  *overfitting* = held-out (val) loss climbs ABOVE random (ln 57 = 4.04 nats).
- **recite top-1 %** (`py/recite_accuracy.py`): teacher-forced pass over the
  whole poem, fraction of positions whose argmax next-char is correct. One
  consistent, comparable instrument across all rungs. (Non-overlapping 256-char
  windows, so window-starts have little context — a constant handicap that
  lowers absolute numbers equally for every model; the *relative* curve is fair.)

## Result (figure: doc/figures/115_layers_to_memorize_raven.png)

| layers | recite top-1 % | train loss (nats/char) | val loss | verdict |
|---|---|---|---|---|
| 1 | 62.0 | 1.16 | 2.52 | can't — val below random, still generalizing |
| 2 | 66.8 | 0.98 | 3.15 | partial overfit, not reciting |
| **3** | **77.0** | **0.61** | **4.96** | **memorizing** — val crosses above random |
| 4 | 77.7 | 0.57 | 5.37 | memorizing |
| 5 | 80.5 | 0.48 | 5.96 | memorizing |
| 6 | 83.5 | 0.37 | 5.85 | memorizing |

## Reading it

- **The answer is ~3 layers, not 6.** Held-out loss first shoots above random at
  L3 — the memorize-and-overfit fingerprint. That's also where the single
  biggest step happens (train 0.98→0.61; recite 67%→77%). One and two layers
  can't memorize (val stays below random — they're still partly generalizing).
- **It's a smooth climb, not an on/off switch.** Each added layer buys a bit
  more (recite 62→67→77→78→80→84). Consistent with our earlier "no capacity
  cliff, soft slope" result (diary 112). "How many layers to memorize" has no
  single magic number; L3 is where it *crosses into* memorizing, and depth keeps
  helping past that.
- **Char memorizes SOFTLY.** Even 6 layers only recites ~84% of next-letters
  (train 0.37 nats/char ≈ 0.53 bits/char) — not the near-perfect lookup the BPE
  memorizer achieved (0.04 bits/char, diary 113). More layers would keep nudging
  it up; the curve had not saturated at L6.

## One honest caveat

Absolute recite % is depressed by the non-overlapping-window context handicap
(and by the 8% val split inside the measured text). It is a fair *relative*
ladder, not an absolute recitation score. The train/val-vs-random crossover (the
memorization criterion) is independent of that handicap and lands cleanly at L3.

## Verdict for the account

Sharpens the diary-114 "depth buys capacity" line: for The Raven at width 16,
memorizing *begins around 3 layers* and deepens smoothly with more — you don't
need all six, and there is no crisp single threshold. New tool:
`py/recite_accuracy.py`. Models: `pt/raven_char_L{1..6}_d16*` (L1 = `_L1_d16_40k`).
