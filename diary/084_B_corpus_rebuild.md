# Diary 084_B: Corpus Rebuild — Filtering and Cleaning

Date: 2026-04-15

## Problem discovered (April 13, Mac Studio session)

Sampling the BPE model at iter ~49,000 revealed corpus quality issues:

1. **Non-fiction contamination**: The corpus isn't just novels — it contains
   architecture treatises (producing "ionic" and "dorian" outputs),
   religious texts (3 copies of the Koran, ~5MB), math data tables,
   encyclopedias, engineering manuals, CIA factbooks, and more.

2. **Encoding artifacts**: Em-dashes mangled into isolated " - " fragments,
   producing incoherent model outputs like "The - and - are..."

3. **Dialect-heavy files**: Yorkshire ditties, Mr. Dooley, Uncle Remus,
   and slave narratives with heavy apostrophe-based dialect notation.

## What was done

### Step 1: filter_corpus.py (Mac Studio, April 13)

Classifies 8,794 Gutenberg source texts using bookshelf categories and
filename pattern matching:

- **REMOVE**: architecture, religious scriptures, math data, encyclopedias,
  engineering, non-English, cookbooks, military manuals, periodicals,
  very short fragments (<3KB)
- **KEEP**: novels, short stories, literary essays, memoirs, travel writing,
  biography, literary criticism, philosophy, adventure, detective fiction,
  children's fiction

Result: 7,543 keep, 1,251 remove.

### Step 2: scan_corpus_quality.py (Mac Studio, April 13)

Scanned the 7,543 kept files (44 minutes) for:
- Dialect density (apostrophe fragments per word)
- Character noise (dashes, pipes, brackets per 1K chars)

Found: 65 dialect-heavy files, 23 high-apostrophe files, 759 noisy files.

### Step 3: rebuild_corpus.py (written April 15)

Builds the final corpus:
1. Reads corpus_keep.txt
2. Excludes files above noise threshold (>15/1k) or dialect threshold (>0.05)
3. Strips Gutenberg headers/footers
4. Lowercases, normalizes Unicode, cleans noise characters
5. Splits into paragraphs, filters fragments (<50 chars)
6. Shuffles paragraphs (seed 42)
7. Writes output

Dry run confirms 7,351 files accessible (192 excluded by quality scan).

## Status

The filter lists and rebuild script are committed to the repo. The actual
rebuild has NOT been run yet — it should be run on the Mac Studio where
the Gutenberg source texts are local (not over network file sharing).

Command to run on Mac Studio:
```
cd small_transformer_research
python py/rebuild_corpus.py \
    --texts_dir ~/Library/Mobile\ Documents/com~apple~CloudDocs/0-HomeFolder-Working-iCloud_A/Gutenberg_Project_Books/gutenberg_texts \
    --keep corpus_keep.txt \
    --quality_report corpus_quality_report.txt \
    --output txt_local/corpus_cleaned_2026_04_15.txt
```

Or if the M3's copy is accessible:
```
python py/rebuild_corpus.py \
    --texts_dir /path/to/Gutenberg_Project_Books/gutenberg_texts \
    --keep corpus_keep.txt \
    --quality_report corpus_quality_report.txt \
    --output txt_local/corpus_cleaned_2026_04_15.txt
```

## After the rebuild

Once the new corpus exists:
1. Retrain the BPE model from scratch (new corpus = new BPE vocabulary)
2. Train the gated-FFN model on the same new corpus
3. Both can then be compared fairly against Model B (char-level)

## Files added to repo

- `py/filter_corpus.py` — classify Gutenberg texts
- `py/scan_corpus_quality.py` — scan for dialect/noise
- `py/rebuild_corpus.py` — build cleaned corpus
- `corpus_keep.txt` — 7,543 files to keep
- `corpus_remove.txt` — 1,251 files to remove
- `corpus_quality_report.txt` — quality scan results
