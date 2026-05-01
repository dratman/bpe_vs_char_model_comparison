#!/usr/bin/env python3
"""
build_context_table.py — Add context word co-occurrence data to a
concordance database built by build_concordance.py.

For each word in the corpus, records every non-stop-word that appears
within WINDOW_SIZE positions, along with the position offset. Does not
cross sentence boundaries (sentences end at . ? !).

Also adds an is_stop column to the existing word table.

Usage:
    python py/build_context_table.py \
        txt_local/corpus_full_2026_04_29.txt \
        txt_local/whole_word_concordance_for_corpus_full_2026_04_29.db \
        --window 5

This will:
  1. Add is_stop column to the word table
  2. Create a context table
  3. Populate both from the corpus
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from nltk.corpus import stopwords

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Characters to strip from word edges when tokenizing
STRIP_CHARS = '.,;:!?"\'-()[]{}*_~`<>|/\\@#$%^&+='

# Sentence-ending punctuation
SENTENCE_ENDERS = {'.', '?', '!'}


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tokenize_into_sentences(text):
    """
    Split text into sentences, where each sentence is a list of
    (cleaned_word, raw_word) tuples. Sentences are split at . ? !

    Returns a generator of lists.
    """
    current_sentence = []

    for raw_word in text.split():
        cleaned = raw_word.strip(STRIP_CHARS).lower()
        if not cleaned:
            continue

        current_sentence.append(cleaned)

        # Check if raw_word ends with sentence-ending punctuation
        last_char = raw_word.rstrip(STRIP_CHARS.replace('.', '').replace('?', '').replace('!', ''))
        if last_char and last_char[-1] in SENTENCE_ENDERS:
            if len(current_sentence) > 0:
                yield current_sentence
                current_sentence = []

    # Yield any remaining words as a final sentence
    if current_sentence:
        yield current_sentence


def build_context(input_path, db_path, window_size):
    """
    Add context co-occurrence data to an existing concordance database.
    """
    print(f"[{get_timestamp()}] Input corpus: {input_path}")
    print(f"[{get_timestamp()}] Database: {db_path}")
    print(f"[{get_timestamp()}] Window size: {window_size}")

    if not os.path.exists(db_path):
        print(f"Error: database {db_path} not found. Run build_concordance.py first.")
        sys.exit(1)

    # Load stop words
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback minimal stop word list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'can', 'not', 'no', 'nor', 'so', 'if', 'than', 'that', 'this',
            'these', 'those', 'it', 'its', 'he', 'she', 'they', 'we', 'you',
            'i', 'me', 'him', 'her', 'us', 'them', 'my', 'his', 'our',
            'your', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
            'how', 'why', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'about', 'up', 'very', 'just',
            'also', 'now',
        }
    print(f"[{get_timestamp()}] Stop words: {len(stop_words)}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Add is_stop column to word table if not present
    print(f"[{get_timestamp()}] Adding is_stop column to word table...")
    try:
        cursor.execute("ALTER TABLE word ADD COLUMN is_stop INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Update is_stop for all stop words
    cursor.execute("SELECT word FROM word")
    all_words = [row[0] for row in cursor.fetchall()]
    stop_updates = [(1, w) for w in all_words if w in stop_words]
    cursor.executemany("UPDATE word SET is_stop = ? WHERE word = ?", stop_updates)
    conn.commit()
    print(f"[{get_timestamp()}] Marked {len(stop_updates):,} words as stop words")

    # Step 2: Create context table
    print(f"[{get_timestamp()}] Creating context table...")
    cursor.execute("DROP TABLE IF EXISTS context")
    cursor.execute("""
        CREATE TABLE context (
            target_word TEXT NOT NULL,
            context_word TEXT NOT NULL,
            position_offset INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (target_word, context_word, position_offset),
            FOREIGN KEY (target_word) REFERENCES word(word),
            FOREIGN KEY (context_word) REFERENCES word(word)
        )
    """)
    conn.commit()

    # Step 3: Read corpus and count co-occurrences
    print(f"[{get_timestamp()}] Reading corpus...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[{get_timestamp()}] Read {len(text):,} characters")

    # Get set of words that are in the database (for filtering)
    word_set = set(all_words)

    print(f"[{get_timestamp()}] Counting context co-occurrences (window={window_size})...")
    # Use a Counter keyed by (target, context, offset) tuples
    context_counts = Counter()
    sentences_processed = 0

    for sentence in tokenize_into_sentences(text):
        sentences_processed += 1

        if sentences_processed % 1000000 == 0:
            print(f"[{get_timestamp()}]   {sentences_processed:,} sentences, "
                  f"{len(context_counts):,} unique co-occurrences")

        n = len(sentence)
        for i in range(n):
            target = sentence[i]
            if target not in word_set:
                continue

            # Look at positions within the window
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                j = i + offset
                if j < 0 or j >= n:
                    continue
                context_word = sentence[j]
                if context_word not in word_set:
                    continue
                if context_word in stop_words:
                    continue

                context_counts[(target, context_word, offset)] += 1

    print(f"[{get_timestamp()}] Sentences processed: {sentences_processed:,}")
    print(f"[{get_timestamp()}] Unique co-occurrences: {len(context_counts):,}")

    # Step 4: Insert into database
    print(f"[{get_timestamp()}] Inserting co-occurrences into database...")
    batch = []
    inserted = 0
    for (target, context_word, offset), count in context_counts.items():
        batch.append((target, context_word, offset, count))
        if len(batch) >= 100000:
            cursor.executemany(
                "INSERT INTO context (target_word, context_word, position_offset, count) "
                "VALUES (?, ?, ?, ?)",
                batch
            )
            inserted += len(batch)
            batch = []
            if inserted % 1000000 == 0:
                print(f"[{get_timestamp()}]   Inserted {inserted:,}")

    if batch:
        cursor.executemany(
            "INSERT INTO context (target_word, context_word, position_offset, count) "
            "VALUES (?, ?, ?, ?)",
            batch
        )
        inserted += len(batch)

    # Create indexes
    print(f"[{get_timestamp()}] Creating indexes...")
    cursor.execute("CREATE INDEX idx_context_target ON context(target_word)")
    cursor.execute("CREATE INDEX idx_context_context ON context(context_word)")
    cursor.execute("CREATE INDEX idx_context_count ON context(count DESC)")

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM context")
    n_rows = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(count) FROM context")
    n_total = cursor.fetchone()[0]

    print(f"\n[{get_timestamp()}] Context table complete:")
    print(f"  Rows: {n_rows:,}")
    print(f"  Total co-occurrences: {n_total:,}")

    # Show examples
    print(f"\nExample context words for 'king' (top 10 by count):")
    cursor.execute(
        "SELECT context_word, position_offset, count FROM context "
        "WHERE target_word = 'king' ORDER BY count DESC LIMIT 10"
    )
    for row in cursor.fetchall():
        print(f"  {row[0]:>15s} at offset {row[1]:+d}: {row[2]:,}")

    print(f"\nExample context words for 'beautiful' (top 10 by count):")
    cursor.execute(
        "SELECT context_word, position_offset, count FROM context "
        "WHERE target_word = 'beautiful' ORDER BY count DESC LIMIT 10"
    )
    for row in cursor.fetchall():
        print(f"  {row[0]:>15s} at offset {row[1]:+d}: {row[2]:,}")

    conn.close()
    print(f"\n[{get_timestamp()}] Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Add context co-occurrence table to concordance database'
    )
    parser.add_argument('input', help='Input text file (same one used for build_concordance.py)')
    parser.add_argument('db', help='Concordance database file')
    parser.add_argument('--window', type=int, default=5,
                        help='Context window size (default: 5)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    build_context(args.input, args.db, args.window)


if __name__ == '__main__':
    main()
