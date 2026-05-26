---
name: Future training ideas
description: Experimental corpus mutations and training variations to try after current standard runs
type: project
---

Ralph sees corpus mutation as a potentially powerful tool for understanding
what models learn. Ideas collected here for future experiments.

## 1. Stop-words-only training (syntax skeleton model)

Replace all non-stop-words in the corpus with underscores (or POS tags
like NOUN, VERB, ADJ) and train on the result. The model would learn
pure English grammar templates — the structural patterns made of function
words ("the", "and", "of", "was", "to", "that", "which", etc.) with
slots where content words go.

**Why:** Reveals how many distinct syntactic templates English uses and
their frequency distribution. Could show that the middle layers of a
normal model are essentially learning these same templates.

**Variant:** Use POS tags instead of underscores to give the model
part-of-speech information at each slot.

**Status:** stop_words.txt and show_stop_words_only.py already exist
in the repo (created 2026-04-17).

## 2. Bilingual training (French + English)

Add 10-20% French literary text to the corpus. The 1.2% incidental
French in an earlier corpus produced surprisingly coherent French output.

**Why:** French and English may share middle-layer abstract representations.
French prose is more structurally uniform and may help the model learn
syntax faster.

**Status:** Decided not to pursue for current run (English-only corpus).
Instructions written for M3 Claude. See project_french_corpus.md.

## 3. LARQL / Gated-FFN (SwiGLU) model

Same corpus, same hyperparameters, but with gated MLP layers instead of
standard MLP. Code exists in GitHub repo (use_gated_mlp flag in model.py).

**Why:** LARQL compatibility. See diary 082_B.

**Status:** Planned as the next training run after the current one.
