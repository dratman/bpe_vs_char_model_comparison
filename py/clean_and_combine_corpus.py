#!/usr/bin/env python3
"""
clean_and_combine_corpus.py — Clean Wikipedia text to the Gutenberg
52-character set and combine with the Gutenberg corpus.

Steps:
  1. Read the Gutenberg high-quality corpus (already clean, pass through)
  2. Read Wikipedia biography text, clean it:
     - Remove "=== TITLE: ... ===" header lines
     - Transliterate accented characters to ASCII (e.g. é -> e)
     - Lowercase
     - Replace runs of disallowed characters with a single space
     - Collapse multiple spaces
  3. Write combined corpus

Usage:
    python py/clean_and_combine_corpus.py \
        --gutenberg txt_local/corpus_high_quality_2026_04_26.txt \
        --wikipedia txt_local/wikipedia_biographies_2026_04_30.txt \
        --output txt_local/corpus_gutenberg_plus_bios_2026_05_01.txt
"""

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime

# The allowed characters for prose text (excluding < > | which appear
# only in the <|endoftext|> separator token)
ALLOWED_PROSE = set("abcdefghijklmnopqrstuvwxyz0123456789 \n!\"'(),-./:;?")

# For final verification, also allow the separator characters
ALLOWED_WITH_SEP = ALLOWED_PROSE | {'<', '>', '|'}


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_wikipedia_text(text):
    """
    Clean Wikipedia text to match the Gutenberg character set.

    1. Remove title header lines
    2. Transliterate accented characters to ASCII
    3. Lowercase
    4. Replace runs of disallowed characters with a single space
    5. Collapse multiple spaces
    """
    # Remove title lines
    text = re.sub(r'^=== TITLE:.*===\n?', '', text, flags=re.MULTILINE)

    # Transliterate accented characters
    normalized = unicodedata.normalize('NFKD', text)
    ascii_approx = normalized.encode('ascii', 'ignore').decode('ascii')

    # Lowercase
    lowered = ascii_approx.lower()

    # Replace runs of disallowed characters with single space
    result = []
    in_disallowed = False
    for c in lowered:
        if c in ALLOWED_PROSE:
            result.append(c)
            in_disallowed = False
        else:
            if not in_disallowed:
                result.append(' ')
                in_disallowed = True
    cleaned = ''.join(result)

    # Collapse multiple spaces
    cleaned = re.sub(r'  +', ' ', cleaned)

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description='Clean Wikipedia text and combine with Gutenberg corpus'
    )
    parser.add_argument('--gutenberg', type=str, required=True,
                        help='Gutenberg high-quality corpus (already clean)')
    parser.add_argument('--wikipedia', type=str, required=True,
                        help='Wikipedia biography text (to be cleaned)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output combined corpus')
    args = parser.parse_args()

    for path in [args.gutenberg, args.wikipedia]:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            sys.exit(1)

    # Read Gutenberg corpus
    print(f"[{get_timestamp()}] Reading Gutenberg corpus: {args.gutenberg}")
    with open(args.gutenberg, 'r', encoding='utf-8') as f:
        gutenberg = f.read()
    print(f"  {len(gutenberg):,} characters")

    # Clean Gutenberg too (remove <, >, | which are not punctuation)
    gut_chars_before = set(gutenberg)
    extra = gut_chars_before - ALLOWED_WITH_SEP
    if extra:
        print(f"  Cleaning {len(extra)} disallowed characters from Gutenberg: {sorted(repr(c) for c in extra)}")
        # Temporarily replace <|endoftext|> markers to protect them
        placeholder = '\x00ENDOFTEXT\x00'
        gutenberg = gutenberg.replace('<|endoftext|>', placeholder)
        result = []
        in_disallowed = False
        for c in gutenberg:
            if c in ALLOWED_PROSE:
                result.append(c)
                in_disallowed = False
            else:
                if not in_disallowed:
                    result.append(' ')
                    in_disallowed = True
        gutenberg = ''.join(result)
        gutenberg = re.sub(r'  +', ' ', gutenberg)
        # Restore markers
        gutenberg = gutenberg.replace(' ENDOFTEXT ', '<|endoftext|>')
        print(f"  Gutenberg after cleaning: {len(gutenberg):,} characters")
    else:
        print(f"  Character set verified: {len(gut_chars_before)} unique, all allowed")

    # Read and clean Wikipedia text
    print(f"\n[{get_timestamp()}] Reading Wikipedia text: {args.wikipedia}")
    with open(args.wikipedia, 'r', encoding='utf-8') as f:
        wiki_raw = f.read()
    print(f"  {len(wiki_raw):,} characters (raw)")

    print(f"[{get_timestamp()}] Cleaning Wikipedia text...")
    wiki_clean = clean_wikipedia_text(wiki_raw)
    print(f"  {len(wiki_clean):,} characters (cleaned)")
    print(f"  Removed: {len(wiki_raw) - len(wiki_clean):,} characters")

    # Verify cleaned text
    wiki_chars = set(wiki_clean)
    extra = wiki_chars - ALLOWED_WITH_SEP
    if extra:
        print(f"  ERROR: cleaned text has {len(extra)} characters outside allowed set!")
        sys.exit(1)
    else:
        print(f"  Character set verified: all allowed")

    # Combine
    print(f"\n[{get_timestamp()}] Combining...")
    # Gutenberg ends with text, Wikipedia starts with text
    # Ensure there is a separator between them
    if not gutenberg.endswith('<|endoftext|>\n'):
        combined = gutenberg + '\n<|endoftext|>\n' + wiki_clean
    else:
        combined = gutenberg + wiki_clean

    print(f"  Combined: {len(combined):,} characters ({len(combined)/1e9:.2f} GB)")

    # Final character set check
    final_chars = set(combined)
    extra = final_chars - ALLOWED_WITH_SEP
    if extra:
        print(f"  ERROR: combined text has {len(extra)} characters outside allowed set!")
        for c in sorted(extra, key=ord):
            print(f"    U+{ord(c):04X} {repr(c)}")
        sys.exit(1)

    # Write
    print(f"\n[{get_timestamp()}] Writing to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(combined)
    print(f"  Done.")

    # Summary
    print(f"\n{'='*60}")
    print(f"Combined corpus: {args.output}")
    print(f"  Gutenberg: {len(gutenberg):,} chars")
    print(f"  Wikipedia:  {len(wiki_clean):,} chars")
    print(f"  Total:      {len(combined):,} chars ({len(combined)/1e9:.2f} GB)")
    print(f"  Characters: {len(final_chars)} unique, all in allowed set")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
