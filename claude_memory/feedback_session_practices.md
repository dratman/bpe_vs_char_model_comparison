---
name: Session practices and preferences
description: Ralph's preferences for how Claude Code sessions should behave
type: feedback
---

Always check the system clock (run `date`) before writing timestamps. Do not guess or pattern-complete timestamps from internal sense of elapsed time — they will be wrong.

**Why:** Prior sessions consistently got timestamps wrong by an hour or more. Ralph identified this as a case of the model treating a lookup task as a formatting task and confabulating the answer.

**How to apply:** Every time you need to write a timestamp at the start of a reply, run `date` first. This applies to every reply, not just the first one.

---

Ralph dislikes trailing summaries of what was just done — he can read the output.

**Why:** He finds them redundant and verbose.

**How to apply:** After tool calls, report only what's necessary — errors, decisions needing input, or key results. Don't recap what the tool output already shows.
