# Diary 085: Corpus Rebuild — Document-Level Shuffling and Cleanup

Date: 2026-04-18

## Problems discovered

### 1. Corpus was not shuffled (April 18)

The corpus built on April 15 (`corpus_cleaned_2026_04_15.txt`, 2.5 GB)
was not shuffled. Books appeared sequentially — the model saw one book
at a time rather than a mix. This was an error in the build process:
the `rebuild_corpus.py` script includes shuffling, but the corpus was
apparently built by a different method that omitted it.

The model trained on this unshuffled corpus for ~53,000 iterations
(reaching val loss 2.88) before we discovered the problem. The training
was stopped and checkpoints saved in `unshuffled_corpus_pt/`.

The unshuffled corpus also creates a flawed validation split: the last
10% of the token stream (used for validation) consists of entire books
rather than a random sample from all books, making the val loss less
meaningful as a measure of generalization.

### 2. Paragraph-level shuffling is non-standard (April 18)

Research into how major LLMs handle corpus shuffling revealed that
paragraph-level shuffling — which is what `rebuild_corpus.py` was
designed to do — is non-standard and likely harmful. Every major LLM
(GPT-2, GPT-3, LLaMA, Pythia, OPT, BLOOM) shuffles at the document
level, keeping each document's internal paragraph order intact.

Key findings from the literature:

- Press (2019): sentence-level shuffling destroys inter-sentence
  dependencies the model needs to learn.
- "Refining Packing and Shuffling Strategies" (2024): the optimal
  "atom size" for shuffling should match the context length (block_size).
  Smaller atoms create "language incoherence."
- Shi et al. (2023, "In-Context Pretraining"): random document
  concatenation wastes compute because unrelated prior documents
  provide no signal for predicting the next document.
- All major models use a special `<|endoftext|>` token between documents.

With our block_size of 1024 tokens (~4000 characters) and median
paragraph length of 227 characters (~57 tokens), paragraph-level
shuffling puts ~18 unrelated paragraph transitions per training block.
Each transition is unpredictable noise the model wastes capacity on.

### 3. Non-English and dialect content (April 18)

The corpus contained:
- ~716 non-English books (French, Spanish, Latin, etc.) — caught by
  an English function-word frequency test (threshold: 10% of words
  must be common English function words like "the", "and", "of", etc.)
- ~139 books with heavy dialect (Uncle Remus, slave narrative
  transcriptions, Yorkshire dialect, minstrel-style writing) —
  identified by frequency of dialect markers (brer, gwine, dat, dem,
  dey, wuz, dere, etc.)
- Scattered non-prose content: footnotes, subscriber lists, recipes,
  tables of contents, bibliographic entries

## Solutions applied

### Document-level shuffling

Modified `rebuild_corpus.py` to:
1. Keep each book's paragraphs in original order
2. Join lines within paragraphs (one paragraph = one line, no wrapping)
3. Shuffle the order of books (seed 42)
4. Separate books with `<|endoftext|>` token on its own line
5. Added `<|endoftext|>` as a special token in the BPE tokenizer

### Content filtering

Applied at the book level:
- English function-word test removes non-English books (716 removed)
- Quality report thresholds remove noisy/dialect-dense books (192 removed)
- Dialect marker frequency test removes heavy-dialect books (139 removed)

Not applied at the paragraph level, because removing individual
paragraphs from within a book would break narrative continuity — the
opposite of what document-level shuffling is meant to preserve.

## Final corpus

File: `txt_local/corpus_books_shuffled_2026_04_18.txt`
- 6,496 books from Project Gutenberg
- 2.14 GB
- Books shuffled, paragraphs in original order within each book
- Books separated by `<|endoftext|>` token
- Paragraphs separated by single newlines
- No internal line breaks within paragraphs
- English-only (occasional French/Latin phrases within English novels
  are retained as authentic literary content)

## Training plan

The next training run will use this corpus with increased block_size
(to be determined — larger blocks mean most blocks fall within a single
book, reducing wasted transitions). Hyperparameters to be discussed
before starting.

## Checkpoints preserved

- `unshuffled_corpus_pt/` — checkpoints from the unshuffled corpus run
  (iters 10K-50K plus latest at ~53K). Can resume if needed.
- `old_8_GB_corpus_pt/` — checkpoints from the original 8 GB corpus run
  (iters 10K-160K plus latest at ~163K).
- `../valuable_checkpoints/bpe_16L16H_old_corpus_iter160k_Excellent_from_8GB_corpus.pt`
  — the best old-corpus checkpoint, separately preserved.

## Files modified

- `py/rebuild_corpus.py` — document-level shuffling, English filter,
  `<|endoftext|>` separator
- `py/tokenizer.py` — added `<|endoftext|>` as special token
- `py/filter_paragraphs.py` — paragraph-level filter (written but
  not applied to the final corpus, for the reasons above)
- `py/sample.py` — cleaned up output formatting
