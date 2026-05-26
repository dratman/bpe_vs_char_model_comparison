---
name: Imitator experiment
description: Mid-layer vector prediction experiment — train small transformer to predict residual stream at layer 8, decode through frozen model's back half
type: project
originSessionId: f13c961b-b895-4196-9417-b60e28615c36
---
Ralph proposed training a model to predict the "language" of a frozen model's
mid-layer residual stream. Idea developed with regular Claude, implemented
in Claude Code on 2026-04-25.

**Architecture:** 21M-param vector-to-vector transformer (d_model=512, n_layer=6,
n_head=8). No embeddings — takes 2048-dim vectors in, produces 2048-dim vectors out.
Linear projections at input/output, reuses Block from model.py internally.

**First run result (2026-04-26):** Cosine similarity 0.948 on validation, but
token-level match only 14-20% when decoded through frozen model's back half.
Output collapses to high-frequency tokens. The gap between vector accuracy and
token accuracy is the core open problem.

**Why:** This is the key finding. High cosine similarity in high-dimensional
space does not translate to functional equivalence. The back half amplifies
small directional differences into large token-probability differences.

**How to apply:** Next experiments should either (a) increase imitator capacity
to close the remaining 5% cosine gap, or (b) switch to downstream KL loss
that directly optimizes token-distribution match. Ralph also wants to explore
retokenizing the predicted vectors into discrete mid-layer tokens.

**Files:** py/imitator_model.py, py/small_model_split.py, py/train_imitator.py,
py/sample_imitator.py, sh/train_imitator.sh, sh/train_imitator_L8_first_run.sh,
sh/sample_imitator_compare.sh. M3 repo has copies of everything.
