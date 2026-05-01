#!/usr/bin/env python3
"""
download_wikipedia_biographies.py

Stream through the HuggingFace Wikipedia dataset, detect biography
articles by text pattern, and save them with titles.

Output format: each article is preceded by a line "=== TITLE: <title> ==="
and followed by "<|endoftext|>" on its own line. The text is saved as-is
from Wikipedia — no character cleaning, no lowercasing.

Usage:
    python py/download_wikipedia_biographies.py --output txt_local/wikipedia_biographies.txt
    python py/download_wikipedia_biographies.py --output txt_local/wikipedia_biographies.txt --target_gb 1.5
"""

import argparse
import re
import sys
from datetime import datetime
from datasets import load_dataset


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Pattern to detect biography articles from their opening text.
# Matches articles that contain "(born ...)" or "(YYYY-YYYY)" or
# "(YYYY-)" date patterns in the first 300 characters.
BIO_PATTERN = re.compile(
    r'^.{0,300}\(born\s'
    r'|^.{0,300}\(\d{4}\s*[-–—]\s*\d{4}\)'
    r'|^.{0,300}\(\d{4}\s*[-–—]\s*\)',
    re.IGNORECASE
)


def is_biography(text):
    """Check if article text looks like a biography."""
    return bool(BIO_PATTERN.search(text[:500]))


def is_list_heavy(text):
    """Check if article is mostly short lines (lists, tables)."""
    lines = text.split('\n')
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 80)
    total_lines = sum(1 for line in lines if line.strip())
    if total_lines == 0:
        return True
    return short_lines / total_lines > 0.5


def main():
    parser = argparse.ArgumentParser(
        description='Download Wikipedia biography articles'
    )
    parser.add_argument('--output', type=str, required=True,
                        help='Output text file')
    parser.add_argument('--target_gb', type=float, default=1.5,
                        help='Target size in GB (default: 1.5)')
    parser.add_argument('--min_chars', type=int, default=2000,
                        help='Minimum article length in characters')
    args = parser.parse_args()

    target_bytes = int(args.target_gb * 1e9)

    print(f"[{get_timestamp()}] Loading Wikipedia dataset (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    total_chars = 0
    articles_kept = 0
    articles_seen = 0
    bios_found = 0

    print(f"[{get_timestamp()}] Scanning for biographies...")
    print(f"  Target: {args.target_gb} GB")
    print(f"  Min article length: {args.min_chars} chars")

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in ds:
            articles_seen += 1

            text = item.get('text', '')
            title = item.get('title', '')

            if not text or len(text) < args.min_chars:
                continue

            if not is_biography(text):
                continue

            bios_found += 1

            if is_list_heavy(text):
                continue

            # Write with title header
            f.write(f"=== TITLE: {title} ===\n")
            f.write(text)
            f.write("\n<|endoftext|>\n")

            total_chars += len(text)
            articles_kept += 1

            if articles_kept % 5000 == 0:
                print(f"[{get_timestamp()}]   Kept {articles_kept:,}, "
                      f"{total_chars/1e6:.0f} MB, "
                      f"scanned {articles_seen:,}, "
                      f"bios found {bios_found:,}")

            if total_chars >= target_bytes:
                print(f"[{get_timestamp()}] Reached target size")
                break

    print(f"\n[{get_timestamp()}] Done.")
    print(f"  Articles scanned: {articles_seen:,}")
    print(f"  Biographies detected: {bios_found:,}")
    print(f"  Articles kept (bio + length + not list): {articles_kept:,}")
    print(f"  Total characters: {total_chars:,} ({total_chars/1e9:.2f} GB)")
    print(f"  Written to: {args.output}")


if __name__ == '__main__':
    main()
