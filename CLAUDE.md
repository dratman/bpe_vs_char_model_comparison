# Current work: a popular account of how a tiny LLM completes a small text

*(This file was simplified 2026-07-08 to describe only the current line of work.
The fuller, older project instructions and HANDOFF.md remain in git history if
the work ever broadens back out.)*

## What we are doing right now

Investigating, from first principles, **how a tiny language model performs one
small, fully-understandable task: literal completion of a small memorized text**
— and turning it into a **popular account for a general reader**. Ralph's frame:

> *Here is how a tiny LLM does a tiny task — continuing text literally from a
> small corpus it was trained on. It invents nothing; it just continues from what
> it memorized.*

Deliberately reductionist: shrink the model, shrink the text, understand it
completely, stay honest about scope. This is **not** novel research (the field
knows this); the value is a **clear, honest, worked example** a reader can trust.

**Example text:** `txt_local/Poe_The_Raven.txt` — Poe's "The Raven" (6,226
chars), canonical/public-domain. (We set Green Eggs and Ham aside — its refrains
cause confusing repetition — and we do **not** use the Storyland-abridged Alice:
publication constraint, canonical texts only.)

**The account:** living outline at `doc/popular_account_outline.md` (no prose
drafted yet). Stay honest about scale — a toy, a "true small foothold," NOT "how
ChatGPT works."

## The strongest result so far (the account's key "scene")

A 1-layer model with embedding width cut to **16** (from 128), trained on The
Raven to a true plateau:

- **char CANNOT memorize it** — hard capacity floor ~1.6 bits/char.
- **BPE memorizes it completely** — ~0.04 bits/char.
- Only the tokenizer differs. BPE compresses 6,226 chars → ~2,130 chunks that
  **fit** under capacity; 6,000 characters don't.
- So at one fixed tiny size, **tokenization decides whether the model
  memorizes-and-overfits (BPE — its loss on *unseen* text explodes above random)
  or is forced to generalize (char — stays below random).** Overfitting requires
  capacity.
- Figure `doc/figures/113_d16_char_vs_bpe_capacity.png`; details in
  `diary/113_...md`.

Also established: for the *literal* task, a 3-line Python `str.find` lookup
**beats** the net on every axis (exact vs 99.6%, unlimited capacity, **1 pass vs
thousands of epochs**, honest "I don't know" at the edge vs confident
confabulation). The net's only distinctive ability — guessing past the edge of
memory — is worthless for literal completion and only earns its keep once the
text is too big to memorize, where the model is forced to learn transferable
structure. **The benefit is born exactly where memorization dies.**

## How to work (this matters most)

- **Interpret, don't just run experiments.** After a result, STOP and say what it
  *means*; don't reflexively loop into the next run. (Ralph's central critique.)
- **Train to a definitive plateau before concluding.** Undertraining repeatedly
  masqueraded as a capacity limit. Log loss finely (e.g. every 500 steps); a real
  capacity floor is flat *while the learning rate is still healthy*, not just the
  rate winding down.
- **Separate a real property of the model from an artifact of the measurement** —
  that has been the crux of every correction we've made.
- Ralph sets direction and understands via plain (child-simple) explanations; you
  do the technical and interpretive heavy lifting. See the memory files for the
  working relationship (cognitive load, one question at a time, emotional
  attunement).

## Technical setup

- Python: `$HOME/miniforge3/bin/python3` (= `/Users/RalphDratman_1/miniforge3/bin/python3`).
  Runs on the Apple GPU (MPS). Never use the system python.
- Train (the pattern for these tiny models):
  ```
  $HOME/miniforge3/bin/python3 -u py/train.py \
    --input txt_local/Poe_The_Raven.txt --output pt/<name> --checkpoints_to pt \
    --tokenizer char --mode continuous --n_layer 1 --n_head 4 --n_embd <D> --block_size 256 \
    --batch_size 32 --max_iters <N> --warmup_iters 100 --learning_rate 1e-3 \
    --eval_interval 500 --eval_iters 20 --log_interval 1000000 \
    --sample_interval 100000000 --save_interval 100000000 \
    --val_split 0.08 --precision float32 --seed 1337
  ```
  BPE variant: `--tokenizer bpe --vocab_size 512 --block_size 85` (BPE ≈ 2.9
  chars/token, so block 85 ≈ 256 chars of context). Integer args only (not `1e9`).
- The MLP hidden width is hardwired to **4 × n_embd** (so n_embd 16 → 64 "switch"
  neurons; n_embd 128 → 512).
- Models → `pt/` (gitignored). Corpora → `txt_local/` (gitignored). `*.png` is
  gitignored — `git add -f` any figure meant for the account.
- git: `git pull` at session start, `git push` at session end.

## Not the current focus

The Mac Studio is running a separate, unrelated long job (a big 16L/1280 char
model, resumed after a disk-full crash, guarded by the `training-monitor`
watchdog). It is **not** part of this account work — leave it alone unless asked.
