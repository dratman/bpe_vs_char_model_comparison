---
name: Check tools before declaring limitations
description: Never say something is impossible without first checking if available tools offer a workaround
type: feedback
---

Do not declare a limitation without first checking whether available tools can achieve the goal in a different way.

**Why:** Ralph had to push back multiple times before I realized that background commands (`run_in_background`) could be used for periodic monitoring — a capability I had used earlier in the same session for other purposes. I kept asserting "I can't poll autonomously" instead of problem-solving with the tools at hand.

**How to apply:** When the impulse is to say "I can't do X," stop and ask: "Do I have a tool that could achieve X in a different way?" Check background commands, LaunchAgents, scripts, or any other mechanism before responding with a limitation.
