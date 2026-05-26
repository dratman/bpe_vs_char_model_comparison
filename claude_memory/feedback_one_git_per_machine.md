---
name: feedback-one-git-per-machine
description: "Across the M3, M2, and Studio, each machine runs its own git operations — never coordinate commits/pulls across machines from a single session via SSH"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 93b1f9ac-00fd-4bee-9367-ec763061ef8b
---

When working in a session on one machine (M3, M2, or Studio) and editing
this project's working tree there, run git commit/push from THAT machine
only. Do NOT chain an SSH command to also `git pull` (or commit, or
push) on a sibling machine in the same step.

**Why:** Ralph prefers to keep git operations local to each machine so
there's no confusion about which working tree is in what state. Even
though SSH between the machines works (Studio→M3 since 2026-05-09;
M2→Studio since 2026-05-23; M3→Studio confirmed 2026-05-26), using
SSH to push/pull on a remote machine mid-session is exactly the kind
of cross-machine action he wants to avoid.

**How to apply:**
- On a commit, push from the current machine only. Mention that the
  other machine(s) will need a `git pull` later, but don't run it.
- On a pull, only pull on the current machine. Don't SSH out to
  pull on the others.
- Read-only inspection over SSH (`tail` a log, `ls -lh` checkpoints,
  `pgrep` a process) is fine — that's not a git operation and not
  what this rule is about.
