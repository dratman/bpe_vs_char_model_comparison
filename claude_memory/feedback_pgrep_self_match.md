---
name: feedback-pgrep-self-match
description: "Never trust `pgrep -af <pattern>` alone to identify child processes — it matches the calling shell's argv too, producing phantom PIDs"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 93b1f9ac-00fd-4bee-9367-ec763061ef8b
---

`pgrep -af <pattern>` searches the full argv of every process AND matches
the calling shell itself, because the shell's argv contains the literal
pattern string. This produces a phantom "extra" PID that looks like a
child process but is just the shell that invoked pgrep. The phantom PID
changes every invocation (each new shell has a new PID), which mimics
the look of a transient subprocess.

**Why:** On 2026-05-26 I claimed `train.py` (PID 36141) had a transient
child subprocess because `pgrep -af "train.py.*char_uppercase"`
consistently returned two PIDs. The second one was always the
SSH-spawned shell running pgrep itself, matching its own argv. The
correct check `pgrep -P 36141` showed zero children, which I had
already run and ignored. Ralph caught this. The training process is
single-threaded, no DataLoader workers, no sampling subprocess —
just one Python process.

**How to apply:**
- To find actual children of a known parent PID: `pgrep -P <parent_pid>`.
- To match exactly on program name: `pgrep -x python` (no `-f`).
- To verify a specific PID exists: `ps -p <pid>` (empty output = gone).
- If `pgrep -af` and `pgrep -P` disagree, trust `-P` — `-af` is
  self-matching.
- Anchor regex patterns: `pgrep -af "^python "` excludes most shells.
- Cross-check anomalies before reporting them as facts. Two PIDs from
  `pgrep -af` is a *clue to investigate*, not a finding to report.

Related: [[feedback-one-git-per-machine]] — both about being careful
when running commands across SSH.
