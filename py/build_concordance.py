#!/usr/bin/env python3
"""
build_concordance.py — Build a whole-word concordance database for a text file.

Creates an SQLite database with:
  - 'stem' table: unique stems (primary key)
  - 'word' table: unique words with count and foreign key to stem

Usage:
    python py/build_concordance.py txt_local/corpus_full_2026_04_29.txt

Output:
    txt_local/whole_word_concordance_for_corpus_full_2026_04_29.db
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from nltk.stem import PorterStemmer


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_concordance(input_path):
    # Derive output path from input filename
    base = os.path.splitext(os.path.basename(input_path))[0]
    db_dir = os.path.dirname(input_path) or '.'
    db_path = os.path.join(db_dir, f"whole_word_concordance_for_{base}.db")

    print(f"[{get_timestamp()}] Input: {input_path}")
    print(f"[{get_timestamp()}] Output: {db_path}")

    # Read and count words
    print(f"[{get_timestamp()}] Reading corpus...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[{get_timestamp()}] Read {len(text):,} characters")

    print(f"[{get_timestamp()}] Counting words...")
    raw_words = text.lower().split()
    counts = Counter()
    strip_chars = '.,;:!?"\'-()[]{}*_~`<>|/\\@#$%^&+='
    for w in raw_words:
        cleaned = w.strip(strip_chars)
        if cleaned:
            counts[cleaned] += 1

    total_words = sum(counts.values())
    unique_words = len(counts)
    print(f"[{get_timestamp()}] Total words: {total_words:,}")
    print(f"[{get_timestamp()}] Unique words: {unique_words:,}")

    # Compute stems
    print(f"[{get_timestamp()}] Computing stems...")
    stemmer = PorterStemmer()
    word_stems = {}
    all_stems = set()
    for i, word in enumerate(counts):
        stem = stemmer.stem(word)
        word_stems[word] = stem
        all_stems.add(stem)
        if (i + 1) % 500000 == 0:
            print(f"  Stemmed {i+1:,} / {unique_words:,}")

    print(f"[{get_timestamp()}] Unique stems: {len(all_stems):,}")

    # Create database
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"[{get_timestamp()}] Creating database...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE stem (
            stem TEXT PRIMARY KEY
        )
    """)

    cursor.execute("""
        CREATE TABLE word (
            word TEXT PRIMARY KEY,
            count INTEGER NOT NULL,
            stem TEXT NOT NULL,
            FOREIGN KEY (stem) REFERENCES stem(stem)
        )
    """)

    # Insert stems
    print(f"[{get_timestamp()}] Inserting {len(all_stems):,} stems...")
    cursor.executemany(
        "INSERT INTO stem (stem) VALUES (?)",
        [(s,) for s in all_stems]
    )

    # Insert words
    print(f"[{get_timestamp()}] Inserting {unique_words:,} words...")
    batch = []
    for i, (word, count) in enumerate(counts.items()):
        stem = word_stems[word]
        batch.append((word, count, stem))
        if len(batch) >= 100000:
            cursor.executemany(
                "INSERT INTO word (word, count, stem) VALUES (?, ?, ?)",
                batch
            )
            batch = []
            print(f"  Inserted {i+1:,} / {unique_words:,}")

    if batch:
        cursor.executemany(
            "INSERT INTO word (word, count, stem) VALUES (?, ?, ?)",
            batch
        )

    # Create indexes for fast queries
    print(f"[{get_timestamp()}] Creating indexes...")
    cursor.execute("CREATE INDEX idx_word_count ON word(count DESC)")
    cursor.execute("CREATE INDEX idx_word_stem ON word(stem)")

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM stem")
    n_stems = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM word")
    n_words = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(count) FROM word")
    n_total = cursor.fetchone()[0]

    print(f"[{get_timestamp()}] Database complete:")
    print(f"  Stems: {n_stems:,}")
    print(f"  Words: {n_words:,}")
    print(f"  Total occurrences: {n_total:,}")
    print(f"  File: {db_path}")

    # Show a few example stem families
    print(f"\nExample stem families:")
    for example_stem in ['walk', 'think', 'beauti', 'happi']:
        cursor.execute(
            "SELECT word, count FROM word WHERE stem = ? ORDER BY count DESC LIMIT 10",
            (example_stem,)
        )
        rows = cursor.fetchall()
        if rows:
            words_str = ', '.join(f"{w} ({c:,})" for w, c in rows)
            print(f"  stem '{example_stem}': {words_str}")

    conn.close()
    print(f"\n[{get_timestamp()}] Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Build a whole-word concordance database for a text file'
    )
    parser.add_argument('input', help='Input text file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    build_concordance(args.input)


if __name__ == '__main__':
    main()
