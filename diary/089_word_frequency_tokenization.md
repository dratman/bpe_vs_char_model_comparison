# Diary 089 — Whole-Word Tokenization Based on Frequency

Date: 2026-04-29

## The idea

Instead of character-level tokenization (52 tokens, no compression) or
BPE (32,000 tokens, arbitrary subword splits), use whole words as tokens
for the most frequent words, with character-by-character fallback for
rare words.

## Corpus analysis

Corpus: corpus_full_2026_04_29.txt (3.2 GB, Gutenberg + Wikipedia)
- 574 million total words
- 4.4 million unique words
- 60.6% of unique words appear exactly once (hapax legomena)
- Zipf's law distribution

## Frequency thresholds

| Threshold | Vocab size | Text coverage | Compression |
|-----------|-----------|---------------|-------------|
| >= 50     | 153,207   | 97.7%         | 5.0x        |
| >= 100    | 99,680    | 97.0%         | 4.9x        |
| >= 200    | 66,298    | 96.2%         | ~4.8x       |
| >= 500    | 38,503    | 94.7%         | ~4.5x       |

## Preferred choice: threshold 100

- Vocabulary: ~100,000 whole-word tokens + alphabet (~52 characters)
- Total vocabulary: ~100,052 tokens
- Covers 97% of all word occurrences
- 3% of words fall back to character spelling
- 4.9x compression vs pure character (704M tokens vs 3.4B characters)
- Each vocabulary word seen at least 100 times in training

## Advantages over BPE

- Token boundaries always at word boundaries — no arbitrary subword
  splits like "un" + "expect" + "edly"
- The model never has to learn where words begin and end
- Whole words have clear semantic meaning as tokens
- Character fallback preserves ability to handle any word

## Advantages over pure character

- 5x compression means 5x more text context per position
- Common words recognized in one step instead of 5-8 steps
- Word-level patterns learned directly, not assembled from characters

## Implementation notes (not yet done)

- Need a delimiter mechanism for character-spelled words (e.g., a
  special token that signals "read characters until next delimiter")
- Tokenizer must handle punctuation: probably treat common punctuation
  marks as separate tokens (period, comma, quotes, etc.)
- Could use spaces as implicit word boundaries (like BPE byte-level)
- The vocabulary of 100K words is larger than typical BPE (32K) but
  each token carries more meaning

## Connection to earlier work

This connects to the imitator retokenization idea (diary 086): the
imitator might discover internal tokens that correspond to whole words
or to something else entirely. The frequency-based whole-word approach
is a simple, interpretable starting point for comparison.

## Status

Idea documented. Not yet implemented. The character model training
continues on both Mac Studio and M3. This tokenizer could be implemented
as an alternative to both character and BPE for future experiments.
