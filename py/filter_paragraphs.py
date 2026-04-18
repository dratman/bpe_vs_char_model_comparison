#!/usr/bin/env python3
"""
filter_paragraphs.py — Remove non-literary paragraphs from a shuffled corpus.

Reads a corpus file (one paragraph per blank-line-separated block),
tests each paragraph against several filters, and writes only the
paragraphs that pass all filters.

Filters:
  1. Non-English: paragraph has too few English function words
  2. Dialect-heavy: too many apostrophe fragments (e.g. runnin', brer)
  3. Non-prose junk: recipe instructions, footnotes, bibliographic entries,
     tables of contents, lists of numbers/data, Latin text, verse line numbers
  4. Too short: paragraphs under 50 characters after cleaning

Usage:
    python py/filter_paragraphs.py \
        --input txt_local/corpus_cleaned_shuffled_2026_04_18.txt \
        --output txt_local/corpus_final_2026_04_18.txt
"""

import argparse
import re
import sys
import time


# --- English detection ---

ENGLISH_FUNCTION_WORDS = frozenset({
    'the', 'and', 'of', 'to', 'a', 'in', 'is', 'was', 'that', 'it',
    'he', 'she', 'his', 'her', 'for', 'with', 'not', 'but', 'had',
    'have', 'are', 'be', 'on', 'at', 'by', 'from', 'this', 'which',
    'an', 'or', 'as', 'were', 'been', 'has', 'their', 'would', 'they',
    'we', 'if', 'my', 'no', 'so', 'did', 'its',
})


def is_english(words):
    """Check if a paragraph's words are predominantly English."""
    if len(words) < 15:
        return True  # too short to judge reliably
    eng_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return (eng_count / len(words)) >= 0.08


# --- Dialect detection ---

# Apostrophe fragments typical of heavy dialect writing
DIALECT_FRAGMENTS = re.compile(
    r"\b(?:an'|ol'|'em|'im|'er|'bout|'fore|'cept|'twas|'tis"
    r"|ne'er|o'er|e'er|e'en|'neath|'gainst|'mongst"
    r"|dat|dis|dem|dey|dere|wuz|wid|fer|ter|jes|gwine|brer"
    r"|nuthin|sumthin|somethin'|nothin'|anythin'"
    r"|runnin'|comin'|goin'|doin'|gettin'|lookin'|talkin')\b"
)


def is_dialect_heavy(text, words):
    """Check if paragraph has too many dialect markers."""
    if len(words) < 15:
        return False
    matches = len(DIALECT_FRAGMENTS.findall(text))
    return (matches / len(words)) > 0.03


# --- Non-prose detection ---

# Recipe/cooking indicators
RECIPE_RE = re.compile(
    r'\b(?:tablespoon|teaspoon|preheat|bake\b.*\boven'
    r'|cup of (?:flour|sugar|butter|milk|water)'
    r'|degrees? (?:f|fahrenheit|celsius)'
    r'|stir (?:in|until|well)|knead|simmer|saute)\b'
)

# Footnote/bibliography indicators
FOOTNOTE_RE = re.compile(
    r'\b(?:footnote \d|ibid\.|op\. cit\.|loc\. cit\.'
    r'|pp\. \d|vol\. [ivxlcdm\d]|cf\. )'
)

# Latin text (more than a few Latin words in a row)
LATIN_WORDS = frozenset({
    'est', 'et', 'in', 'non', 'ad', 'cum', 'sed', 'qui', 'quod',
    'ut', 'de', 'ab', 'ex', 'per', 'pro', 'sunt', 'quae', 'enim',
    'hoc', 'aut', 'nec', 'quid', 'iam', 'tamen', 'esse', 'sic',
    'ergo', 'atque', 'autem', 'vel', 'nam', 'etiam', 'nos',
})

# Spanish/Portuguese/Italian function words
ROMANCE_WORDS = frozenset({
    'el', 'la', 'los', 'las', 'del', 'una', 'por', 'con', 'para',
    'que', 'como', 'pero', 'esta', 'este', 'ese', 'mas', 'muy',
    'anche', 'della', 'delle', 'degli', 'nella', 'sono', 'questo',
    'quello', 'perche', 'quando', 'sempre', 'tutto',
    'pelo', 'pela', 'pela', 'como', 'onde', 'porque', 'quando',
})

# Table of contents / chapter listings
TOC_RE = re.compile(
    r'(?:chapter [ivxlcdm\d]+\s*[-:.]\s*){2,}|'
    r'(?:page \d+\s*){3,}|'
    r'(?:\.\s*\.\s*\.?\s*\d+\s*$)'
, re.MULTILINE)

# Newspaper/periodical mastheads
MASTHEAD_RE = re.compile(
    r'\b(?:publisher|editor|per annum|subscription|weekly|daily)\b.*'
    r'\b(?:publisher|editor|per annum|subscription|weekly|daily)\b'
)


def is_non_prose(text, words):
    """Check if paragraph is non-literary content."""
    if len(words) < 10:
        return False

    # Recipe content
    if len(RECIPE_RE.findall(text)) >= 2:
        return True

    # Footnotes/bibliography
    if len(FOOTNOTE_RE.findall(text)) >= 2:
        return True

    # Latin-heavy paragraphs
    if len(words) >= 15:
        latin_count = sum(1 for w in words if w in LATIN_WORDS)
        # 'in' and 'et' overlap with English, so require a higher bar
        # Only flag if clearly Latin (many non-overlapping Latin words)
        latin_only = LATIN_WORDS - {'in', 'a', 'non'}
        latin_specific = sum(1 for w in words if w in latin_only)
        if latin_specific / len(words) > 0.08:
            return True

    # Spanish/Portuguese/Italian
    if len(words) >= 15:
        romance_count = sum(1 for w in words if w in ROMANCE_WORDS)
        if romance_count / len(words) > 0.06:
            return True

    # Table of contents
    if TOC_RE.search(text):
        return True

    # Newspaper mastheads
    if MASTHEAD_RE.search(text):
        return True

    return False


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Filter non-literary paragraphs from corpus')
    parser.add_argument('--input', required=True, help='Input corpus file')
    parser.add_argument('--output', required=True, help='Output filtered corpus file')
    parser.add_argument('--report', action='store_true', help='Print examples of filtered paragraphs')
    args = parser.parse_args()

    start_time = time.time()

    with open(args.input) as f:
        text = f.read()

    paragraphs = text.split('\n\n')
    total = len(paragraphs)
    print(f"Input: {total} paragraphs")

    # Filter counts
    counts = {
        'non_english': 0,
        'dialect': 0,
        'non_prose': 0,
        'too_short': 0,
        'kept': 0,
    }

    # Examples of filtered paragraphs (for --report)
    examples = {
        'non_english': [],
        'dialect': [],
        'non_prose': [],
    }

    kept = []

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        if len(para) < 50:
            counts['too_short'] += 1
            continue

        words = para.split()

        if not is_english(words):
            counts['non_english'] += 1
            if args.report and len(examples['non_english']) < 5:
                examples['non_english'].append(para[:150])
            continue

        if is_dialect_heavy(para, words):
            counts['dialect'] += 1
            if args.report and len(examples['dialect']) < 5:
                examples['dialect'].append(para[:150])
            continue

        if is_non_prose(para, words):
            counts['non_prose'] += 1
            if args.report and len(examples['non_prose']) < 5:
                examples['non_prose'].append(para[:150])
            continue

        kept.append(para)
        counts['kept'] += 1

        if (i + 1) % 1000000 == 0:
            elapsed = time.time() - start_time
            print(f"  {i+1}/{total} processed, {counts['kept']} kept, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\nFiltering complete in {elapsed:.0f}s")
    print(f"  Total input:  {total}")
    print(f"  Non-English:  {counts['non_english']}")
    print(f"  Dialect:      {counts['dialect']}")
    print(f"  Non-prose:    {counts['non_prose']}")
    print(f"  Too short:    {counts['too_short']}")
    print(f"  Kept:         {counts['kept']} ({100*counts['kept']/total:.1f}%)")

    if args.report:
        for category, examples_list in examples.items():
            if examples_list:
                print(f"\n--- Examples of {category} ---")
                for ex in examples_list:
                    print(f"  {ex}...")

    # Write output
    print(f"\nWriting {len(kept)} paragraphs to {args.output}...")
    with open(args.output, 'w') as f:
        for i, para in enumerate(kept):
            f.write(para)
            f.write('\n\n')
            if (i + 1) % 1000000 == 0:
                print(f"  {i+1}/{len(kept)} written", flush=True)

    import os
    final_size = os.path.getsize(args.output)
    print(f"Done: {args.output} ({final_size/1e9:.2f} GB)")


if __name__ == '__main__':
    main()
