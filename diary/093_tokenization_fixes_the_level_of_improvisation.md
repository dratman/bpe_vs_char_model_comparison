# Diary 093 — Tokenization fixes the level at which the model improvises

Date: 2026-05-17

## The observation

Ralph's framing, after reading 5 samples each from the in-progress
Studio char run and the M3 BPE run at matched training epoch:

> BPE improvises sentences while char improvises words.

That is a simplification but it gives the idea. Stated more precisely:

**The tokenization scheme fixes a level of linguistic structure, and
leaves the model free to vary above it.** The model's creative degrees
of freedom — and its visible errors — live at scales *coarser* than
the token, not finer.

| tokenization | token = | level the model improvises at | typical errors |
|---|---|---|---|
| char (vocab 78) | one letter | **word** | invented names, garbled rare words, non-words |
| BPE (vocab 32K) | sub-word / word | **sentence** | non-sequiturs, semantically slippery propositions, repetition |
| whole-word (diary 089) | one word | **clause / discourse** | logical contradictions, but no word-level errors possible by construction |

## What the actual samples showed

Both runs at roughly matched corpus exposure on 2026-05-17, generating
on the same Studio hardware:

- char (320M, iter 154K, epoch 2.22): adventurous vocabulary
  ("irreparable", "discolored", "contiguity") but also invented words
  ("Sebolunes", "Warrange", "Hearsell"). Sample registers drifted
  between narrative dialogue, society-column catalogue, and
  historical commentary.
- BPE (361M, iter 95K, epoch 3.0): smoother and more conventional
  prose, more reliable narrative-with-dialogue register, fewer invented
  words (just "CHEVALS" in a letter-header context). Multi-paragraph
  plot coherence held over ~1,350-character generations.

Both produced credible 19th-century literary English. Both still had
local strangenesses (char: "the cattle had to skin themselves"; BPE:
"the country is full of deer — with deer —"). The level where the
strangeness lives differs.

Per-character validation loss was within 2% of each other (char 0.79,
BPE 0.77, both still well above their respective floors), so the
"how good is it numerically" comparison is not yet conclusive. The
qualitative difference *was* immediately conclusive.

## Why this is the right reading

This is just the topological framing from diary 074 stated in
ordinary terms. Tokenization eliminates dimensions the model would
otherwise have to construct in its own layers. The detailed
mechanistic version is in earlier diary entries:

- 014, 015: at character level, Layer 4 "reads back" the previous
  characters of the current word to recognize it. Attention head H2
  reads the first letter at 0.95+ probability. The model is *building
  word identity from letters* using layer-machinery.
- 035: word recognition typically happens at Layer 5 (the last layer)
  in a 6-layer char model. "said" is recognized at L5; "through"
  needs all 6 layers.
- 074: persistent homology of activations shows H1 loops (closed
  cycles in the activation manifold) corresponding to word-level
  structure — the topological signature of "word" emerging across
  early-to-middle layers.

In a BPE model, the analogue of these word-building layers can be
spent on something coarser, because the tokens already encode words.
The model's capacity moves *up the hierarchy*: from "what word am I
looking at" to "what proposition follows the previous proposition."

## A practical corollary

The choice of tokenization is, in part, the choice of *which level
of error you want to live with*.

- If you must never misspell, use a coarse tokenization. (The model
  cannot misspell a word it cannot decompose.)
- If you must allow novel words (technical jargon, proper nouns, code
  identifiers, transliterations of foreign words), use a finer
  tokenization. (The model can compose any letter sequence.)
- If you care most about long-range narrative coherence at fixed
  block_size, BPE's longer effective context (~4.5× more chars per
  sequence at the same `block_size`) is a real benefit, independent
  of any modeling-quality question.

There is no "best" tokenization, only the one matched to where you
want the imperfections to land.

## Connection to diary 090

Diary 090 framed tokenization as a learned function — something that
could in principle be a trainable transformer module rather than a
fixed preprocessing step. If that view is taken seriously, the
abstraction-level argument above has a useful corollary: the
tokenizer-network's job is to *choose at which level the downstream
language model should be creative*. Different downstream tasks would
benefit from different tokenizer outputs. A code model might want
syntactic-token tokenization (statements are atoms, expressions are
atoms). A poetry model might want syllabic tokenization (rhythm and
meter are atoms). A model of natural language for everyday discourse
might want BPE.

We have not built any of this. But the framing makes the design space
sharper: tokenization choice is not just an implementation
optimization; it is **a choice of where the model's freedom and
imperfection live**.
