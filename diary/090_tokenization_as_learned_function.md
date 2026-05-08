# Diary 090 — Tokenization as a Learned Function

Date: 2026-05-08

## The idea

A transformer can be trained to perform tokenization. BPE, morphological
segmentation, syllabification, frequency-thresholded whole-word
tokenization, phonemic mapping — any of these is a learned function
from character streams to segmented output, and a small transformer
should learn any of them with ease.

This reframes tokenization from an opaque preprocessing algorithm into
a swappable architectural module that can be studied, ablated, and
compared.

## Why this matters

Conventionally, tokenization is a separate algorithmic stage: BPE merges
are computed once over the corpus, frozen, and applied as a Python
preprocessing step before training begins. The historical reason is
computational efficiency, not principle.

But there is no reason in principle that the tokenizer must be a
separate algorithmic stage. A small transformer trained on
(character_stream → token_sequence) pairs would perform the same
function, with these advantages:

- The tokenizer becomes a network whose internals can be inspected
- The tokenizer can be swapped, ablated, or replaced without changing
  the downstream language model architecture
- The tokenizer can perform context-dependent segmentation, which no
  static algorithm produces
- Multiple tokenization schemes become directly comparable as
  controlled experimental conditions

## Tokenization schemes that become tractable

Once tokenization is a trained module, the following all become
swappable alternatives:

- BPE (replicating the standard algorithm as a network)
- Frequency-thresholded whole-word (diary 089)
- Morphological boundaries (un- + expect- + -ed + -ly)
- Syllabic boundaries
- Phonemic transcription (orthography → phoneme sequence)
- Syntactic-parse-derived segmentation
- Random segmentation as control
- Context-dependent segmentation (read as one unit in "read the book",
  as re + ad in "re-adjusting")

Each becomes a frozen pretrained tokenizer-transformer that can be
attached as the input stage of a language model.

## Connection to the imitator retokenization idea (diary 086)

The imitator approach asks: what tokens does a trained network discover
when given freedom to retokenize?

The trained-tokenizer approach asks the inverse: what happens to a
language model when we impose a specific tokenization scheme?

Together these are a forward/inverse pair. The imitator reveals the
network's preferred segmentation; the trained tokenizer lets us specify
segmentations and measure downstream consequences. The two can be
compared directly — does the imitator's discovered tokenization
resemble any of the imposed schemes? Is it closer to BPE, to
morphological boundaries, to whole-word frequency carving, or to
something else?

## A clean experimental design

Train two language models on the same corpus, identical architectures,
where the only difference is the tokenizer module performing scheme A
vs scheme B. Same compute, same data, same downstream architecture.
Differences in performance and representation are attributable to
tokenization alone.

This is difficult with conventional tokenizers because input
vocabularies and sequence lengths differ across schemes. With trained
tokenizer-transformers, the interfaces can be arranged to match
exactly.

## Output format choice

Two ways the tokenizer-transformer can produce its output:

1. **Token ID classification** over a fixed vocabulary. Simpler, but
   the vocabulary is fixed at training time.

2. **Segmentation boundaries on a character stream.** The tokenizer
   outputs character-aligned boundary markers. The downstream model
   reads character sequences with boundary information. This is more
   flexible — the tokenizer can produce a unit it has never produced
   before, if its learned segmentation rule suggests it should.

The second framing is more powerful for interpretability work because
the "vocabulary" is not fixed; it emerges from the segmentation
function.

## Connection to the character/BPE dichotomy

Character models almost never misspell, suggesting they have memorized
spellings rigidly — a large lexical inventory of fixed strings retrieved
deterministically. BPE models can produce variant spellings because
their tokens combine flexibly.

If tokenization becomes a trained module, the character model's
spelling-retrieval system and the BPE model's merge-application system
can both be expressed as tokenizer-transformers and compared
mechanistically. The "rigid lexicon vs flexible recombination" contrast
becomes a property of the tokenizer module, separable from the
downstream language model.

## Status

Idea documented. Not yet implemented. A small tokenizer-transformer
(few layers, modest width) trained on (chars → BPE_tokens) would be a
natural first step, both as a proof of concept and as a baseline
against which other segmentation schemes can be compared.
