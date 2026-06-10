---
name: feedback-git-pull-push-discipline
description: "Pull before editing model code; push only after testing it. For HANDOFF.md and diary/, pull at session start and push at session end. Reduces merge conflicts from parallel sessions on the M3, M2, Studio, and Linux A6000 box."
metadata:
  node_type: memory
  type: feedback
---

When working on this project across multiple machines (M3, M2, Mac
Studio, Linux A6000 workstation), conform to this discipline:

1. **For model code** — anything in `py/` (especially `model.py`,
   `train.py`, `sample.py`, `tokenizer.py`) and anything in `sh/` that
   launches training (`train.sh`, `train_*.sh`, `train_cuda.sh`):
   run `git pull` *before* editing, and `git commit && git push` only
   *after* a correctness check (training launches, sample runs, test
   passes). No broken commits should leave a machine.

2. **For session logbook** — `HANDOFF.md` and `diary/`:
   `git pull` at session start, `git commit && git push` at session
   end. These files get touched by almost every session and accumulate
   updates fast; pulling at start grounds the session in the latest
   shared state, pushing at end makes the session's contributions
   visible to the other machines.

3. **For HANDOFF.md specifically**: prefer appending a new dated
   entry at the top rather than overwriting the existing "Last
   updated" slot. Git can usually merge non-overlapping additions
   automatically. The current convention of demoting prior "Last
   updated" entries to "Prior update" causes both sides of a
   parallel edit to touch the same lines, which is what produces
   the merge conflicts (e.g., the 2026-06-07 + 2026-06-10 conflicts
   between the Studio session and the Linux/A6000 sessions).

**Why:** Ralph's mental model is one experimenter, several machines,
all "thinking together" through one repo. That model works smoothly
when git operations stay aligned — every session sees the previous
session's work. It breaks when unpushed commits sit on one machine
while another machine runs a session and modifies overlapping files.
The pull-before / test-then-push discipline keeps the timeline
linear without sacrificing the multi-machine concept. Adding the
append-style HANDOFF convention reduces conflicts on the file that
gets touched most.

**How to apply:**
- Before editing any file in `py/` or any `sh/train_*` script,
  run `git pull` first. After editing, run a correctness check
  before committing. Push immediately after commit; don't let
  model-code commits sit unpushed overnight.
- At the start of any session, `git pull` for the working tree.
  At the end of any session, commit and push.
- When updating HANDOFF.md's "Last updated" section, append a new
  dated entry above the previous one rather than rewriting the
  slot. Leave previous entries untouched (or only minimally
  edited).
- The terminal logs, plots, and large `.pt` checkpoints don't
  need this discipline — they don't trigger conflicts because
  each run produces uniquely-named files.

**Related memories:** [[feedback-one-git-per-machine]] —
this discipline operates within the one-git-per-machine rule.
Each machine still runs its own git operations; the new
discipline is about *when* during a session those operations
should happen.
