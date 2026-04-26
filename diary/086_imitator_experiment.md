# Diary 086 — Imitator Experiment: Predicting Mid-Layer Vector Dynamics

Date: 2026-04-25 / 2026-04-26

## The idea

Ralph proposed training a model to predict the "language" of a frozen
model's mid-layer residual stream. The concept: take the 875M BPE model,
freeze it, split it at layer 8. Train a separate small transformer to
predict the next residual-stream vector at that boundary, treating the
sequence of layer-8 vectors as a "language" to be modeled. Then decode
the predicted vectors through the frozen model's back half (layers 8-15)
to get English out.

The idea was developed in conversation with regular Claude (chat interface),
then implemented in Claude Code.

## Architecture

The imitator is a vector-to-vector transformer. No token embeddings, no
positional embeddings (the residual stream already carries position from
the frozen model's first 8 layers). It takes 2048-dimensional vectors in
and produces 2048-dimensional vectors out.

First run: d_model=512 (projects down from 2048, processes, projects back),
6 layers, 8 heads, 21M parameters.

Loss: (1 - cosine_similarity).mean() + 0.1 * MSE. Cosine handles direction
(the semantic content); MSE pins magnitude.

## First run results (d_model=512, 5000 iterations, M3 Pro)

Training took ~70 minutes. Cosine similarity climbed from -0.0004 (random)
to 0.9484 within 5000 iterations. Most of the learning happened in the
first 800 iterations. Val loss closely tracked train loss — no overfitting
(the imitator saw <5% of the training data, less than one epoch).

## The gap between vector accuracy and token accuracy

When we decoded the imitator's predicted vectors through the frozen model's
back half, the results were poor. Token-level match with the frozen model's
own predictions: 14-20%. The decoded text collapsed to high-frequency tokens
(commas, "the", "of", "to").

Cosine similarity of 0.948 in 2048-dimensional space corresponds to about
18 degrees of angular error. This is small in vector terms but large enough
that the back half — which is designed to make sharp token decisions from
subtle directional differences — produces different tokens.

### Evaluation bug found and fixed

The original compare script had a causal context error: the back half was
missing position 0's activation when decoding shifted sequences. Both the
frozen model's column and the imitator's column were degraded. Fixed by
passing the full activation sequence through the back half and prepending
real position 0 to the imitator's predictions.

## Per-position analysis of the frozen model

Ran the War and Peace passage ("prince andrew screwed up his eyes...")
through the full frozen model (all 16 layers) and examined predictions
position by position.

Key findings:
- Top-1 accuracy: 27% (20 of 74 positions exactly right)
- Actual token in top 5: 65%
- Actual token in top 10: 72%
- Median rank of actual token: 3
- Mean probability on actual token: 18.6%

The model knows English well. Most "wrong" predictions are near misses:
"looked" instead of "turned", "cordial" instead of "pleasant". The model
fails on specific content words that require knowing this particular text
("unexpectedly" — rank 9594, essentially unpredictable from context).

## Free generation vs forced prediction

The most striking finding: the same model that scores 27% on forced
next-token prediction produces coherent, stylistically appropriate prose
when generating freely. Given the first two sentences of the War and Peace
passage as a prompt, the model generated plausible 19th-century novel
dialogue with appropriate characters and emotional register.

Ralph's insight: in free generation, the model is always continuing from
its own output. It "knows its own idiom" — its statistical tendencies are
consistent from one token to the next. In forced prediction, it is trying
to guess someone else's specific word choices, which is inherently harder.

This distinction reframes the imitator experiment. The compare mode (forced
prediction) is the harsh test. Rollout mode (free generation in vector
space) may produce much more coherent output, because the imitator gets
to find its own trajectory rather than matching the frozen model's exactly.

## Comparison with frontier model

Called Claude Sonnet API with the same prompt. It produced natural,
character-appropriate dialogue but — like the 875M model — did not predict
Tolstoy's actual continuation. The frontier model's prose is more natural
and the character dynamics are more accurate, but the fundamental situation
is the same: no model can predict an author's specific word choices.

## Second run started (d_model=2048)

Started a 302M-parameter imitator at full residual dimension (no projection
bottleneck). Hypothesis: the 512-dim bottleneck discards subtle features
the back half needs. At initialization, cosine similarity was already 0.84
(due to residual connections acting as approximate identity). Stopped early
to explore the questions above. To be resumed.

## Open questions

1. Will the imitator's rollout mode produce coherent decoded text?
2. Does the full-dim (2048) imitator close the vector-to-token gap?
3. Would downstream KL loss (optimize token distribution match directly)
   be more effective than cosine + MSE loss?
4. What do imitators at different split layers reveal?
5. Can the predicted vectors be retokenized into discrete mid-layer tokens?
6. Would this experiment on a stronger base model (Llama 3B) be more
   informative?

## Scripts created

- `py/imitator_model.py` — vector-to-vector transformer
- `py/small_model_split.py` — split frozen GPT at any layer
- `py/train_imitator.py` — training loop
- `py/sample_imitator.py` — compare, rollout, and stats modes
- `sh/train_imitator.sh` — logging wrapper
- `sh/train_imitator_L8_first_run.sh` — first run (d_model=512)
- `sh/train_imitator_L8_full_dim.sh` — second run (d_model=2048)
- Various evaluation scripts on M3 share
