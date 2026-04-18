#!/usr/bin/env python3
"""
rebuild_corpus.py — Build a cleaned Gutenberg corpus from individual text files.

Reads corpus_keep.txt (files to include) and corpus_quality_report.txt
(files to exclude due to noise/dialect), strips Gutenberg headers/footers,
cleans character-level noise, shuffles at paragraph level, and writes
the final corpus.

Usage:
    python py/rebuild_corpus.py \
        --texts_dir /path/to/gutenberg_texts \
        --keep corpus_keep.txt \
        --quality_report corpus_quality_report.txt \
        --output txt_local/corpus_cleaned_YYYY_MM_DD.txt \
        [--noise_threshold 15] \
        [--dialect_threshold 0.05] \
        [--no_shuffle] \
        [--seed 42]

The script:
  1. Reads corpus_keep.txt for the list of files to include
  2. Reads corpus_quality_report.txt and removes files above noise/dialect thresholds
  3. For each kept file: strips Gutenberg header/footer, lowercases, cleans noise
  4. Splits all text into paragraphs (on blank lines)
  5. Shuffles paragraphs (reproducible with --seed)
  6. Writes final corpus

Character cleaning:
  - Lowercase everything
  - Normalize Unicode: smart quotes -> straight, em/en dashes -> " -- "
  - Collapse runs of dashes: " - - - " -> " -- "
  - Remove non-prose characters: |{}[]<>~^@#\\_
  - Collapse multiple spaces to one
  - Collapse 3+ newlines to 2
  - Keep only: a-z, space, newline, and basic punctuation . , ; : ! ? ' " - ( )
"""

import argparse
import os
import random
import re
import sys
import time
import unicodedata


def parse_quality_report(path, noise_threshold, dialect_threshold):
    """Parse corpus_quality_report.txt, return set of filenames to exclude."""
    exclude = set()
    if not path or not os.path.exists(path):
        return exclude

    section = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if 'DIALECT-HEAVY FILES' in line:
                section = 'dialect'
                continue
            elif 'HIGH APOSTROPHE DENSITY' in line:
                section = 'apostrophe'
                continue
            elif 'HIGH CHARACTER NOISE' in line:
                section = 'noise'
                continue

            if not line.strip() or line.startswith('Found') or line.startswith('Checked') or line.startswith('Corpus') or line.startswith('Skipped'):
                continue

            # Parse lines like: "  0.1139 dialect/word (277 fragments)  filename.txt"
            # or: "  94.2/1k (  0 dashes, 1225 noise chars)  filename.txt"
            parts = line.strip().split()
            if not parts:
                continue

            try:
                value = float(parts[0].rstrip('/1k'))
            except ValueError:
                continue

            # Extract filename (last token ending in .txt)
            filename = None
            for p in reversed(parts):
                if p.endswith('.txt'):
                    filename = p
                    break
            if not filename:
                continue

            if section == 'dialect' and value > dialect_threshold:
                exclude.add(filename)
            elif section == 'noise' and value > noise_threshold:
                exclude.add(filename)
            elif section == 'apostrophe' and value > 0.15:
                exclude.add(filename)

    return exclude


GUTENBERG_HEADER_PATTERNS = [
    re.compile(r'\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG', re.IGNORECASE),
    re.compile(r'Produced by ', re.IGNORECASE),
]

GUTENBERG_FOOTER_PATTERNS = [
    re.compile(r'\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG', re.IGNORECASE),
    re.compile(r'End of (?:the )?Project Gutenberg', re.IGNORECASE),
]


def strip_gutenberg_header_footer(text):
    """Remove Project Gutenberg header and footer."""
    lines = text.split('\n')

    # Find header end
    header_end = 0
    for i, line in enumerate(lines[:200]):  # header is in first 200 lines
        for pat in GUTENBERG_HEADER_PATTERNS:
            if pat.search(line):
                header_end = i + 1
                break

    # Find footer start
    footer_start = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 200, 0), -1):
        for pat in GUTENBERG_FOOTER_PATTERNS:
            if pat.search(lines[i]):
                footer_start = i
                break

    return '\n'.join(lines[header_end:footer_start])


def clean_text(text):
    """Clean a single text: lowercase, normalize chars, remove noise."""
    # Lowercase
    text = text.lower()

    # Normalize Unicode
    # Smart quotes to straight
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # single
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # double
    text = text.replace('\u2014', ' -- ')   # em dash
    text = text.replace('\u2013', ' -- ')   # en dash
    text = text.replace('\u2026', '...')     # ellipsis
    text = text.replace('\u00a0', ' ')       # non-breaking space

    # Remove other non-ASCII (accented chars etc) by decomposing and stripping
    # Actually, keep common accented letters by normalizing to closest ASCII
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Collapse runs of dashes
    text = re.sub(r'(?:\s*-\s*){3,}', ' -- ', text)
    text = re.sub(r'\s*---+\s*', ' -- ', text)

    # Remove noise characters
    text = re.sub(r'[|{}\[\]<>~^@#\\_]', '', text)

    # Remove tab characters
    text = text.replace('\t', ' ')

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Final filter: keep only allowed characters
    # a-z, 0-9, space, newline, and basic punctuation
    allowed = set('abcdefghijklmnopqrstuvwxyz 0123456789\n.,;:!?\'"-()')
    text = ''.join(c for c in text if c in allowed)

    # Re-collapse spaces after character removal
    text = re.sub(r' {2,}', ' ', text)

    return text


def split_paragraphs(text):
    """Split text into paragraphs on blank lines. Filter short ones."""
    paragraphs = re.split(r'\n\n+', text)
    # Keep paragraphs with at least 50 characters (skip fragments)
    return [p.strip() for p in paragraphs if len(p.strip()) >= 50]


# Common English function words for language detection
ENGLISH_FUNCTION_WORDS = frozenset({
    'the', 'and', 'of', 'to', 'a', 'in', 'is', 'was', 'that', 'it',
    'he', 'she', 'his', 'her', 'for', 'with', 'not', 'but', 'had',
    'have', 'are', 'be', 'on', 'at', 'by', 'from', 'this', 'which',
    'an', 'or', 'as', 'were', 'been', 'has', 'their', 'would', 'they',
    'we', 'if', 'my', 'no', 'so', 'did', 'its',
})

ENGLISH_MIN_RATIO = 0.10


def is_english(text):
    """Return True if text appears to be English based on function word frequency.
    English prose typically has 35-42% function words. Non-English is under 2%.
    Threshold is set at 10% to be safe."""
    words = text.split()
    if len(words) < 100:
        return True  # too short to judge, keep it
    eng_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return (eng_count / len(words)) >= ENGLISH_MIN_RATIO


def main():
    parser = argparse.ArgumentParser(description='Rebuild cleaned Gutenberg corpus')
    parser.add_argument('--texts_dir', required=True, help='Directory containing individual Gutenberg text files')
    parser.add_argument('--keep', required=True, help='corpus_keep.txt file')
    parser.add_argument('--quality_report', default=None, help='corpus_quality_report.txt file')
    parser.add_argument('--output', required=True, help='Output corpus file path')
    parser.add_argument('--noise_threshold', type=float, default=15.0, help='Noise/1k threshold for exclusion (default: 15)')
    parser.add_argument('--dialect_threshold', type=float, default=0.05, help='Dialect/word threshold for exclusion (default: 0.05)')
    parser.add_argument('--no_shuffle', action='store_true', help='Do not shuffle paragraphs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffle (default: 42)')
    parser.add_argument('--dry_run', action='store_true', help='Just report stats, do not write output')
    args = parser.parse_args()

    # Read keep list
    with open(args.keep) as f:
        keep_files = [line.strip() for line in f if line.strip()]
    print(f"Keep list: {len(keep_files)} files")

    # Parse quality report for exclusions
    quality_exclude = parse_quality_report(args.quality_report, args.noise_threshold, args.dialect_threshold)
    print(f"Quality exclusions: {len(quality_exclude)} files (noise>{args.noise_threshold}/1k, dialect>{args.dialect_threshold})")

    # Final file list
    final_files = [f for f in keep_files if f not in quality_exclude]
    print(f"Final file count: {len(final_files)}")

    if args.dry_run:
        # Just check how many files are accessible
        accessible = 0
        missing = 0
        for fn in final_files:
            path = os.path.join(args.texts_dir, fn)
            if os.path.exists(path):
                accessible += 1
            else:
                missing += 1
        print(f"Accessible: {accessible}, Missing: {missing}")
        return

    # Process files — keep each book as a unit
    all_books = []  # list of (filename, list_of_paragraphs)
    total_chars = 0
    total_paragraphs = 0
    processed = 0
    skipped = 0
    non_english = 0
    start_time = time.time()

    for fn in final_files:
        path = os.path.join(args.texts_dir, fn)
        if not os.path.exists(path):
            skipped += 1
            continue

        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                raw = fh.read()

            text = strip_gutenberg_header_footer(raw)
            text = clean_text(text)

            if not is_english(text):
                non_english += 1
                continue

            paragraphs = split_paragraphs(text)
            if not paragraphs:
                skipped += 1
                continue

            # Join lines within each paragraph into one line
            paragraphs = [' '.join(p.split()) for p in paragraphs]

            all_books.append((fn, paragraphs))
            total_chars += sum(len(p) for p in paragraphs)
            total_paragraphs += len(paragraphs)
            processed += 1

            if processed % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  {processed}/{len(final_files)} processed, "
                      f"{len(all_books)} books, "
                      f"{total_paragraphs} paragraphs, "
                      f"{total_chars/1e9:.2f} GB, "
                      f"{elapsed/60:.1f} min elapsed",
                      flush=True)

        except Exception as e:
            print(f"  Error reading {fn}: {e}", file=sys.stderr)
            skipped += 1

    elapsed = time.time() - start_time
    print(f"\nProcessed: {processed} files, Skipped: {skipped}, Non-English: {non_english}")
    print(f"Total books: {len(all_books)}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total characters: {total_chars:,} ({total_chars/1e9:.2f} GB)")
    print(f"Time: {elapsed/60:.1f} min")

    # Shuffle books (not paragraphs)
    if not args.no_shuffle:
        print(f"Shuffling {len(all_books)} books with seed {args.seed}...")
        random.seed(args.seed)
        random.shuffle(all_books)

    # Write output: books separated by <|endoftext|>, paragraphs by single newlines
    SEPARATOR = '<|endoftext|>'
    print(f"Writing to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as out:
        for i, (fn, paragraphs) in enumerate(all_books):
            if i > 0:
                out.write(SEPARATOR + '\n')
            out.write('\n'.join(paragraphs))
            out.write('\n')
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_books)} books written", flush=True)

    final_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({final_size/1e9:.2f} GB)")
    print(f"Format: {len(all_books)} books, separated by {SEPARATOR}")


if __name__ == '__main__':
    main()
