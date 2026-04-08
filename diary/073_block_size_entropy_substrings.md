# Diary Entry 073: Block Size, Entropy Landscape, and Repeated Substrings
## Characterizing what a transformer can learn from the Cat in the Hat

### The experiment

Trained the same 2L1H dim=32 architecture on lowercase Cat in the Hat (7099 chars,
33-char vocab) with three different block sizes:

| Model | block_size | val_split | iters | epochs | best val loss |
|-------|-----------|-----------|-------|--------|---------------|
| block64 (original) | 64 | 0.1 | 4000 | ~2563 | **1.55** |
| ctx700 | 700 | 0.1 | 4000 | 438 | 2.09 |
| fullctx | 7098 | 0.0 | 500 | 500 | 0.15 (train) |

Also ran: fullctx with 4x repeated corpus (loss 2.14), fullctx with dropout=0.2
(loss 0.37).

### Key finding: longer context hurts this tiny model

block64 (val loss 1.55) beats ctx700 (val loss 2.09) despite 10x less context.
dim=32 with 1 attention head cannot organize attention across 700 positions — it
dilutes its signal. The 64-position constraint actually helps by forcing the model
to focus on local patterns it can handle.

fullctx reaches loss 0.15 but this is pure memorization — batch_size=1,
block_size=entire text, no val split. Every iteration is the same sequence.
To support val_split=0.0 we added permanent handling for missing val data in
train.py (estimate_loss_continuous, estimate_loss, and the training loop).

### Entropy analysis (py/entropy_analysis.py)

Computed per-position entropy for all three models across the full text:

| Metric | block64 | block700 | full7098 |
|--------|---------|----------|----------|
| Mean entropy | 1.41 nats | 2.13 nats | 0.78 nats |
| Word boundary entropy | 2.53 | 2.79 | 0.94 |
| Mid-word entropy | 1.16 | 1.99 | 0.74 |
| Sentence-end entropy | 0.73 | 0.98 | 0.86 |
| Mean free path | 1.79 chars | 1.62 chars | 2.10 chars |

block64 has the best figure-ground separation: word boundaries (2.53) vs mid-word
(1.16) = 2.2x ratio. It clearly distinguishes "inside a word" from "between words."
block700 barely sees this distinction (1.4x ratio). fullctx is confident everywhere
(memorization).

"Mean free path" = average length of low-entropy runs (below median threshold).
Analogous to mean free path in statistical mechanics — how far the prediction
cursor travels before hitting an uncertainty "collision." All models have mean free
path under 2.1 characters — none can reliably sustain prediction through a full word.

### Repeated-substring analysis (py/repeated_substrings.py)

Analyzed the text itself (no model involved) for repeated substrings at every
length scale:

| Length | Repetition ratio | Random baseline | Interpretation |
|--------|-----------------|-----------------|----------------|
| 1-2 | ~1.0 | ~1.0 | Everything repeats at character level |
| 3-8 | 0.94 -> 0.26 | ~0.0 | **Words**: steepest drop zone |
| 10-20 | 0.26 -> 0.06 | 0.0 | Repeated phrases |
| 30-44 | 0.01 -> 0.0 | 0.0 | Longest repeats (Seuss refrains) |
| 50+ | 0.0 | 0.0 | Every span is unique |

Characteristic lengths (steepest drops): 4, 5, 6, 7, 8 — exactly word+space size.
Random baseline drops to zero by length 3 — all repetition beyond bigrams is
genuine linguistic structure.

Longest repeated substring: "to shake hands with thing one and thing two" (44 chars).

### The hierarchy of randomness

The text has a natural hierarchy of structural levels, each with a characteristic
length scale:

1. **Characters within words** (~1-3 chars): nearly deterministic
2. **Words** (~4-8 chars): the dominant structural unit (steepest repetition drop)
3. **Phrases** (~10-30 chars): Seuss repetitions ("said the cat in the hat")
4. **Passages** (~30-44 chars): longest repeated spans
5. **Beyond ~50 chars**: every span is unique

A model's "mean free path" tells you which level it operates at. block64's
mean free path of 1.79 is below the word level (4-8 chars) — it partially captures
words but can't reliably complete them. This explains why generation produces
recognizable word fragments in random order.

### Connection to combinatorics on words

The repetition ratio curve is the complement of the **subword complexity function**
p(n), studied extensively in combinatorics (Lothaire, Morse-Hedlund theorem). The
rate at which p(n) approaches N characterizes the text's complexity at each scale.

The random baseline (birthday-problem approximation) gives the null hypothesis.
Excess repetition above this baseline = genuine structure. The characteristic
lengths where the curve drops fastest are the boundaries between structural levels.

This connects to **Lempel-Ziv complexity** (LZ compression exploits exactly these
repeated substrings) and to the **Shannon entropy rate** (the limit of H(n)/n as
context length n grows).

The potentially novel bridge: connecting the text's subword complexity profile
to what a finite-capacity transformer can actually exploit. The model's entropy
landscape should reflect the text's repetition structure, filtered through the
model's architectural constraints (dim, layers, heads, block_size).

### Practical implications

- block_size should match the model's capacity to organize attention, not just
  be "as large as possible"
- For a 2L1H dim=32 model, block_size=64 is well-matched; 700 is too large
- The text's characteristic lengths (4-8 for words) set a minimum capability
  target: a model should have mean free path >= characteristic length to
  capture that structural level
- Reaching mean free path of ~5 (word level) would require more model capacity
  (bigger dim, more layers), not just more context

### Tools created

- `py/entropy_analysis.py` — per-position entropy, mean free path, boundary
  analysis, comparison plots
- `py/repeated_substrings.py` — subword complexity, characteristic lengths,
  longest repeated substrings, random baseline comparison
- `train.py` updated to handle `--val_split 0.0` gracefully (permanent change)

### Output files

- `doc/entropy_comparison.txt` — summary statistics
- `doc/block64_entropy.csv`, `doc/block700_entropy.csv`, `doc/full7098_entropy.csv`
- `doc/entropy_trace.png`, `doc/entropy_histogram.png`, `doc/mean_free_path.png`,
  `doc/entropy_boundaries.png`
- `doc/repeated_substrings_summary.txt`, `doc/repeated_substrings_stats.csv`
- `doc/repetition_curve.png`, `doc/repetition_derivative.png`
- `doc/the_cat_in_the_hat_2b_lower_word_counts.txt`
- `doc/the_cat_in_the_hat_2b_lower_char_counts.txt`

### Models saved

- `saved_models/cathat_2L1H_d32_ctx700` (500 iter)
- `saved_models/cathat_2L1H_d32_ctx700_long` (4000 iter)
- `saved_models/cathat_2L1H_d32_fullctx` (memorizer, 500 iter)
- `saved_models/cathat_2L1H_d32_fullctx_4x` (4x repeated corpus)
- `saved_models/cathat_2L1H_d32_fullctx_drop` (dropout=0.2)
