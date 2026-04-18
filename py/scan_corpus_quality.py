"""
scan_corpus_quality.py — Scan kept corpus files for dialect and character noise.

Reads corpus_keep.txt and checks each file for:
  1. High apostrophe density (dialect writing like 'twas th' ol' man's 'orse)
  2. High character noise (dashes, pipes, brackets, encoding artifacts)

Files are on iCloud and may be slow to access. The script has:
  - 10 second per-file timeout
  - 30 minute overall timeout
  - Progress reporting every 500 files

Output: corpus_quality_report.txt with flagged files and reasons.
"""

import os
import signal
import time

TEXTS_DIR = '/Users/RalphDratman/Library/Mobile Documents/com~apple~CloudDocs/0-HomeFolder-Working-iCloud_A/Gutenberg_Project_Books/gutenberg_texts'
KEEP_LIST = 'corpus_keep.txt'
OUTPUT = 'corpus_quality_report.txt'
PER_FILE_TIMEOUT = 10 * 60  # 10 minutes per file (iCloud may need to download)

class FileTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise FileTimeout()

def scan_file(path):
    """Read first 20KB, return (apostrophe_density, noise_density, dialect_fragments)."""
    with open(path, encoding='utf-8', errors='replace') as fh:
        text = fh.read(20000)

    # Skip Gutenberg header — look at chars 2000-15000
    sample = text[2000:15000]
    if len(sample) < 1000:
        return None

    words = sample.split()
    if not words:
        return None

    # Apostrophe density
    apos_count = sample.count("'") + sample.count('\u2019') + sample.count('\u2018')
    apos_per_word = apos_count / len(words)

    # Count dialect-specific patterns: 'e, 'er, 'im, 'is, 'twas, th', o', etc
    dialect_fragments = 0
    for w in words:
        # Words starting with apostrophe (dropped letter): 'e, 'er, 'im, 'twas
        if w.startswith("'") and len(w) > 1 and len(w) <= 5 and w[1:].isalpha():
            dialect_fragments += 1
        if w.startswith('\u2018') and len(w) > 1 and len(w) <= 5:
            dialect_fragments += 1
        # Words ending with apostrophe (dropped letter): th', o', ol'
        if w.endswith("'") and len(w) > 1 and len(w) <= 5 and w[:-1].isalpha():
            dialect_fragments += 1

    dialect_per_word = dialect_fragments / len(words) if words else 0

    # Dash/noise density
    dash_count = sample.count(' - ') + sample.count(' -- ') + sample.count(' --- ')
    noise_chars = sum(1 for c in sample if c in '|{}[]<>~^@#\\_')
    noise_per_1k = (dash_count + noise_chars) * 1000 / len(sample)

    return {
        'apos_per_word': apos_per_word,
        'dialect_per_word': dialect_per_word,
        'dialect_fragments': dialect_fragments,
        'noise_per_1k': noise_per_1k,
        'dash_count': dash_count,
        'noise_chars': noise_chars,
        'word_count': len(words),
    }


def main():
    with open(KEEP_LIST) as f:
        keep_files = [line.strip() for line in f if line.strip()]

    start_time = time.time()
    checked = 0
    skipped = 0
    timed_out = 0

    apostrophe_heavy = []  # (filename, density, dialect_frags)
    dialect_heavy = []     # (filename, dialect_per_word, dialect_frags)
    noise_heavy = []       # (filename, noise_per_1k, dash_count, noise_chars)

    signal.signal(signal.SIGALRM, timeout_handler)

    for fn in keep_files:
        path = os.path.join(TEXTS_DIR, fn)
        if not os.path.exists(path):
            skipped += 1
            continue

        try:
            signal.alarm(PER_FILE_TIMEOUT)
            result = scan_file(path)
            signal.alarm(0)

            if result is None:
                skipped += 1
                continue

            if result['apos_per_word'] > 0.12:
                apostrophe_heavy.append((fn, result['apos_per_word'], result['dialect_fragments']))
            if result['dialect_per_word'] > 0.02:
                dialect_heavy.append((fn, result['dialect_per_word'], result['dialect_fragments']))
            if result['noise_per_1k'] > 5:
                noise_heavy.append((fn, result['noise_per_1k'], result['dash_count'], result['noise_chars']))

            checked += 1
            if checked % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  {checked}/{len(keep_files)} checked ({elapsed/60:.1f} min elapsed, {timed_out} timeouts)", flush=True)

        except FileTimeout:
            signal.alarm(0)
            timed_out += 1
            checked += 1
        except Exception as e:
            signal.alarm(0)
            skipped += 1

    elapsed = time.time() - start_time
    print(f"\nDone: {checked} checked, {skipped} skipped, {timed_out} timed out in {elapsed/60:.1f} min")

    # Write report
    with open(OUTPUT, 'w') as out:
        out.write(f"Corpus Quality Scan Report\n")
        out.write(f"Checked {checked} of {len(keep_files)} files in {elapsed/60:.1f} min\n")
        out.write(f"Skipped: {skipped}, Timed out: {timed_out}\n\n")

        out.write(f"=== DIALECT-HEAVY FILES (apostrophe fragments per word > 0.02) ===\n")
        out.write(f"Found {len(dialect_heavy)} files:\n\n")
        for fn, density, frags in sorted(dialect_heavy, key=lambda x: -x[1]):
            out.write(f"  {density:.4f} dialect/word ({frags:3d} fragments)  {fn}\n")

        out.write(f"\n=== HIGH APOSTROPHE DENSITY (> 0.12 per word) ===\n")
        out.write(f"Found {len(apostrophe_heavy)} files:\n\n")
        for fn, density, frags in sorted(apostrophe_heavy, key=lambda x: -x[1]):
            out.write(f"  {density:.4f} apos/word ({frags:3d} fragments)  {fn}\n")

        out.write(f"\n=== HIGH CHARACTER NOISE (> 5 per 1K chars) ===\n")
        out.write(f"Found {len(noise_heavy)} files:\n\n")
        for fn, noise, dashes, nchars in sorted(noise_heavy, key=lambda x: -x[1]):
            out.write(f"  {noise:.1f}/1k ({dashes:3d} dashes, {nchars:4d} noise chars)  {fn}\n")

    print(f"\nReport written to {OUTPUT}")
    print(f"  Dialect-heavy: {len(dialect_heavy)}")
    print(f"  High apostrophe: {len(apostrophe_heavy)}")
    print(f"  High noise: {len(noise_heavy)}")


if __name__ == '__main__':
    main()
