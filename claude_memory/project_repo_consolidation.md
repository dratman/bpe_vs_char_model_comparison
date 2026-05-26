---
name: Repo consolidation plan
description: Plan to merge bpe_vs_char_model_comparison into small_transformer_research as a subdirectory
type: project
originSessionId: f13c961b-b895-4196-9417-b60e28615c36
---
Ralph wants to consolidate bpe_vs_char_model_comparison into the
small_transformer_research GitHub repo as a subdirectory.

**Why:** The two repos share the same model.py and related code. Having
them separate creates confusion and duplication.

**Current state:**
- `small_transformer_research` — on GitHub at dratman/small_transformer_research,
  local clone at ../small_transformer_research/
- `bpe_vs_char_model_comparison` — on GitHub at dratman/bpe_vs_char_model_comparison,
  this is the active working directory

**Plan:**
1. Wait for a pause between experiments (both training runs finished)
2. Move bpe_vs_char_model_comparison contents into a subdirectory of
   small_transformer_research (e.g., small_transformer_research/bpe_comparison/)
3. Update all path references in scripts and CLAUDE.md
4. Keep the shared code (model.py, tokenizer.py, train.py) at the top
   level of small_transformer_research
5. Archive or redirect the standalone bpe_vs_char_model_comparison repo

**How to apply:** When both training runs are done and there is a natural
pause, remind Ralph about this consolidation.
