# Comments on Diary 090 — Tokenization as a Learned Function

Date: 2026-05-08
Commenter: Claude Code (Opus 4.7), Mac Studio session

The framing is conceptually clean and the forward/inverse pairing with
the imitator (diary 086) is genuinely elegant — they ask different
questions about the same internal structure. Treating tokenization as
a function instead of a fixed preprocessing stage is the right move
for interpretability research, and the design has obvious appeal.

But I'd push back on a few things.

## "A small transformer should learn any of them with ease" is too easy

Replicating BPE specifically is not trivial: BPE is a deterministic
greedy algorithm operating on the full string, and an autoregressive
next-token predictor will fail at boundary cases and on rare strings.
Calling it easy before measuring sets up disappointment.

## The interpretability gain depends on which scheme

For BPE the static algorithm is *more* transparent than a learned
approximation — you can read the merge table directly. The
interpretability argument has real force only for schemes that have
no good static algorithm: context-dependent segmentation, morphological
boundaries with messy exceptions, etc. The proof-of-concept proposed
(chars → BPE_tokens) is the case where the gain is smallest. The
interesting experiments are buried in the middle of the entry.

## The framing of the char/BPE contrast is too tidy

> "Character models almost never misspell, suggesting they have
> memorized spellings rigidly — a large lexical inventory of fixed
> strings retrieved deterministically."

This conflicts with what diary 040 showed in this very project:
character models degrade gracefully on unknown sequences and apply
sub-word patterns even on nonsense. They're not pure lookup tables.
That paragraph would be stronger as an open question than a stated
suggestion.

## Prior art is missing

Charformer (Tay et al., 2021), GBST, ByT5, and MegaByte all explored
learned segmentation and byte-level models with learned subword
aggregation. The diary doesn't locate itself relative to these. For
an idea entry that's tolerable, but if this gets pursued, the
literature scan should come early — there may already be answers to
some of the proposed experiments.

## If this gets developed

The smallest informative experiment is probably a 2-layer
char→BPE-id classifier on a held-out chunk, then *measure where it
fails*. That alone tells you whether "with ease" holds and whether
the failures are concentrated at interesting linguistic boundaries
or just at rare strings. Cheap to run, surfaces real information.

## Net

A strong concept, oversold in tone, undersold on the experiments
that would actually matter.
