#!/usr/bin/env python3
"""
match_book_samples.py — Recover the index ↔ filename mapping for the
Haiku-judged corpus.

The Haiku quality-filter step that produced `doc/book_quality_decisions.json`
is not in the repository. Its decisions are keyed by integer indices
(`'0'` … `'6197'`) that refer to entries in `doc/all_book_samples.txt`,
which contains 3 short excerpts per book but no filenames. This script
reconstructs the missing mapping by fuzzy-matching each book's excerpts
against the 8,794 source `.txt` files, then writes:

  - `book_index_to_filename.json` — full mapping for reproducibility
  - `corpus_haiku_keep.txt`       — filenames where the Haiku verdict is KEEP

Usage:
    python py/match_book_samples.py \\
        --texts_dir "/path/to/Gutenberg_Project_Books/gutenberg_texts" \\
        --samples doc/all_book_samples.txt \\
        --decisions doc/book_quality_decisions.json \\
        --out_mapping book_index_to_filename.json \\
        --out_keep corpus_haiku_keep.txt
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import ahocorasick


SIG_LEN = 60          # length of the matching signature, in normalized chars
SIG_OFFSET = 40       # start position into normalized excerpt (skip first chars
                      # which may carry residual excerpt-marker context)


# Normalize text to lowercase ASCII alphanumerics + single-space separators.
# This avoids quote-style differences (' vs ’), em-dashes, accents, etc.
_NORMALIZE_KEEP = set(
    "abcdefghijklmnopqrstuvwxyz0123456789"
)


# Project Gutenberg header/footer marker patterns. The boilerplate around
# these markers (legal notices, donation requests, contact info) is identical
# across most PG files, so signatures extracted from this region can match
# many unrelated source files. Stripping the boilerplate before matching
# eliminates that confound.
_PG_HEADER_PATTERNS = [
    re.compile(r'\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG', re.IGNORECASE),
    re.compile(r'Produced by ', re.IGNORECASE),
]
_PG_FOOTER_PATTERNS = [
    re.compile(r'\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG', re.IGNORECASE),
    re.compile(r'End of (?:the )?Project Gutenberg', re.IGNORECASE),
]


def strip_pg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer if present.

    Mirrors the same routine in rebuild_corpus.py — checks the first/last
    200 lines for marker patterns. If no marker is found, returns text
    unchanged.
    """
    lines = text.split('\n')
    header_end = 0
    for i, line in enumerate(lines[:200]):
        for pat in _PG_HEADER_PATTERNS:
            if pat.search(line):
                header_end = i + 1
                break
    footer_start = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 200, 0), -1):
        for pat in _PG_FOOTER_PATTERNS:
            if pat.search(lines[i]):
                footer_start = i
                break
    return '\n'.join(lines[header_end:footer_start])


def normalize(text: str) -> str:
    """Lowercase, keep [a-z0-9], collapse runs of other chars to single space."""
    text = text.lower()
    out = []
    last_was_space = True
    for c in text:
        if c in _NORMALIZE_KEEP:
            out.append(c)
            last_was_space = False
        else:
            if not last_was_space:
                out.append(' ')
                last_was_space = True
    return ''.join(out).strip()


def parse_samples(path: str) -> dict:
    """Parse all_book_samples.txt → {idx (int): combined_text (str)}.

    The file format is
        === BOOK 0 ===
          [1] excerpt one ...
          [2] excerpt two ...
          [3] excerpt three ...
        === BOOK 1 ===
        ...
    """
    books = {}
    current_idx = None
    current_lines = []
    header_re = re.compile(r'^=== BOOK (\d+) ===\s*$')
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = header_re.match(line)
            if m:
                if current_idx is not None:
                    books[current_idx] = '\n'.join(current_lines)
                current_idx = int(m.group(1))
                current_lines = []
            else:
                current_lines.append(line.rstrip('\n'))
    if current_idx is not None:
        books[current_idx] = '\n'.join(current_lines)
    return books


def build_signatures(books: dict, sig_len: int = SIG_LEN, offset: int = SIG_OFFSET):
    """For each book, normalize the combined text and take a `sig_len`-char
    window starting at `offset`. Return (signatures, too_short_indices,
    duplicate_signatures).

    Each book's combined text contains all 3 excerpts joined; together they
    are a few hundred characters, so a 60-char window from offset 40 is
    well inside excerpt [1]'s body, past any leading bracketed-number
    marker.
    """
    sigs = {}                    # idx -> signature
    too_short = []
    sig_to_indices = defaultdict(list)

    for idx, text in books.items():
        normed = normalize(text)
        if len(normed) < offset + sig_len:
            too_short.append(idx)
            continue
        sig = normed[offset:offset + sig_len]
        sigs[idx] = sig
        sig_to_indices[sig].append(idx)

    duplicates = {sig: idxs for sig, idxs in sig_to_indices.items() if len(idxs) > 1}
    return sigs, too_short, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--texts_dir', required=True,
                        help='Directory containing the 8,794 individual Gutenberg .txt files')
    parser.add_argument('--samples', default='doc/all_book_samples.txt',
                        help='Path to all_book_samples.txt')
    parser.add_argument('--decisions', default='doc/book_quality_decisions.json',
                        help='Path to book_quality_decisions.json')
    parser.add_argument('--out_mapping', default='book_index_to_filename.json',
                        help='Where to write the recovered index→filename mapping')
    parser.add_argument('--out_keep', default='corpus_haiku_keep.txt',
                        help='Where to write the filtered keep list (Haiku KEEP verdicts)')
    parser.add_argument('--report_unmatched', default='unmatched_books.txt',
                        help='Where to log book indices we could not match (for inspection)')
    args = parser.parse_args()

    # 1. Parse samples
    print(f"[1/6] Parsing {args.samples} ...")
    books = parse_samples(args.samples)
    print(f"  Parsed {len(books)} books")
    if not books:
        sys.exit("No books parsed; check the samples file format.")

    # 2. Build signatures
    print(f"[2/6] Building {SIG_LEN}-char signatures ...")
    sigs, too_short, duplicates = build_signatures(books)
    print(f"  Built {len(sigs)} signatures")
    print(f"  Too short to sign: {len(too_short)}")
    print(f"  Duplicate signatures: {len(duplicates)}")
    if duplicates:
        # Two books with byte-identical 60-char signatures is suspicious.
        # Dump the first few for inspection.
        print("  First duplicate cases:")
        for i, (sig, idxs) in enumerate(list(duplicates.items())[:3]):
            print(f"    sig={sig!r}  idxs={idxs}")

    # 3. Build Aho-Corasick automaton over all signatures
    print(f"[3/6] Building Aho-Corasick automaton ...")
    A = ahocorasick.Automaton()
    for idx, sig in sigs.items():
        A.add_word(sig, (idx, sig))
    A.make_automaton()
    print(f"  Automaton ready ({len(sigs)} patterns)")

    # 4. Walk the source-file directory and match
    print(f"[4/6] Scanning {args.texts_dir} ...")
    files = sorted(f for f in os.listdir(args.texts_dir) if f.endswith('.txt'))
    print(f"  {len(files)} source files")

    # First-match-wins: once a book index gets a filename, ignore subsequent matches.
    # If a single file matches multiple unmatched book indices, accept all of them
    # (different books in the corpus could conceivably draw on the same source if,
    # e.g., the sampler hit different excerpts of the same long file — but in
    # practice each book index should map to exactly one file).
    mapping = {}                         # idx -> filename
    file_to_books = defaultdict(list)    # fn -> list of idxs matched in this file
    t0 = time.time()
    unmatched_so_far = set(sigs.keys())

    for i, fn in enumerate(files):
        if not unmatched_so_far:
            break
        path = os.path.join(args.texts_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                raw = fh.read()
        except OSError:
            continue
        # Strip Project Gutenberg header/footer boilerplate so we don't
        # match signatures drawn from license / donation / contact text.
        raw = strip_pg_boilerplate(raw)
        normed = normalize(raw)

        # Aho-Corasick scan returns (end_position, value) for each match
        seen_in_this_file = set()
        for end_pos, (idx, sig) in A.iter(normed):
            if idx in unmatched_so_far and idx not in seen_in_this_file:
                seen_in_this_file.add(idx)
                mapping[idx] = fn
                file_to_books[fn].append(idx)
        unmatched_so_far -= seen_in_this_file

        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - (i + 1)) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(files)} files  matched={len(mapping)}  "
                  f"unmatched={len(unmatched_so_far)}  "
                  f"elapsed={elapsed/60:.1f}min  rate={rate:.1f}/s  eta={eta/60:.1f}min",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Scan complete in {elapsed/60:.1f} min")
    print(f"  Raw matches: {len(mapping)} of {len(sigs)} signatures")

    # Files that absorbed multiple book indices — log and DEMOTE them.
    # If a single file matches multiple book signatures, the signatures
    # were almost certainly drawn from text shared across many files
    # (PG boilerplate, common quotations, anthology overlaps, etc.).
    # We can't trust any individual book→filename assignment in those
    # cases, so we drop them from the mapping and treat the affected
    # book indices as unmatched.
    multi_matches = {fn: idxs for fn, idxs in file_to_books.items() if len(idxs) > 1}
    if multi_matches:
        print(f"  Files with multiple book matches (demoting): {len(multi_matches)}")
        demoted_indices = []
        for fn, idxs in list(multi_matches.items())[:5]:
            print(f"    {fn}  idxs={idxs}")
        for fn, idxs in multi_matches.items():
            for idx in idxs:
                if idx in mapping and mapping[idx] == fn:
                    del mapping[idx]
                    demoted_indices.append(idx)
                    unmatched_so_far.add(idx)
        print(f"  Demoted {len(demoted_indices)} ambiguous assignments")
    print(f"  Final clean matches: {len(mapping)}")

    # 5. Write mapping artifacts
    print(f"[5/6] Writing artifacts ...")
    with open(args.out_mapping, 'w') as fh:
        json.dump({str(k): v for k, v in sorted(mapping.items())}, fh, indent=2)
    print(f"  {args.out_mapping}: {len(mapping)} entries")

    if unmatched_so_far or too_short:
        with open(args.report_unmatched, 'w') as fh:
            fh.write(f"# Unmatched book indices (could not find source file)\n")
            fh.write(f"# Total: {len(unmatched_so_far)} unmatched after scan, "
                     f"{len(too_short)} too short to sign\n\n")
            for idx in sorted(unmatched_so_far):
                excerpt = books.get(idx, '')[:200].replace('\n', ' ')
                fh.write(f"{idx}\t<UNMATCHED>\t{excerpt}\n")
            for idx in sorted(too_short):
                excerpt = books.get(idx, '')[:200].replace('\n', ' ')
                fh.write(f"{idx}\t<TOO_SHORT>\t{excerpt}\n")
        print(f"  {args.report_unmatched}: {len(unmatched_so_far) + len(too_short)} entries")

    # 6. Apply Haiku decisions to produce keep list
    print(f"[6/6] Applying Haiku decisions from {args.decisions} ...")
    with open(args.decisions) as fh:
        decisions = json.load(fh)

    keep_files = []
    decision_counts = {'KEEP_mapped': 0, 'KEEP_unmapped': 0, 'REMOVE': 0, 'OTHER': 0}
    for idx_str, verdict in decisions.items():
        idx = int(idx_str)
        if verdict == 'KEEP':
            if idx in mapping:
                keep_files.append(mapping[idx])
                decision_counts['KEEP_mapped'] += 1
            else:
                decision_counts['KEEP_unmapped'] += 1
        elif verdict == 'REMOVE':
            decision_counts['REMOVE'] += 1
        else:
            decision_counts['OTHER'] += 1

    keep_files = sorted(set(keep_files))  # dedupe (multi-match cases)
    with open(args.out_keep, 'w') as fh:
        for fn in keep_files:
            fh.write(fn + '\n')

    print(f"  {args.out_keep}: {len(keep_files)} unique filenames")
    print(f"  Decision counts: {decision_counts}")

    # Sanity: original Haiku run kept 4,430 of 6,198. We should be close.
    expected_keeps = decision_counts['KEEP_mapped'] + decision_counts['KEEP_unmapped']
    if expected_keeps > 0:
        coverage = decision_counts['KEEP_mapped'] / expected_keeps * 100
        print(f"  Mapped {coverage:.1f}% of KEEP verdicts to filenames")


if __name__ == '__main__':
    main()
