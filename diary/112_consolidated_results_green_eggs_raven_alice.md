# Diary 112 — Consolidated results: Green Eggs → The Raven → Alice (memorization studies)

Date: 2026-07-07

This entry preserves, at Ralph's request, his prompt and the full consolidated
answer — a reference synthesis of the memorization/interpretability arc in
diaries 107–111 plus the ongoing Alice capacity work. Solid findings and the
honest corrections are both marked.

## Ralph's prompt (verbatim)

> List out all of the results starting with Green Eggs.

## The answer

### Green Eggs and Ham (3,375 chars)

1. **It memorizes.** A tiny char model reproduces the text — but only under
   **greedy** decoding; temperature-0.8 sampling makes it look garbled even when
   fully memorized.
2. **Block 128:** recites 18% from the first line, then hits a **real context
   collision** (the "…Sam-I-Am." refrain recurs with different continuations).
3. **Block 256:** recites 90.1%, then stops.
4. **CORRECTION:** that 90% wall was **not** a capacity limit — it's the
   **train/val split boundary** (last 10% never trained on). Proven by moving the
   split (val 0.30 → wall at 70%).
5. **Training longer didn't help** (it can't recite text it never saw).
6. **Depth sweep: one layer is enough** — a 1-layer model memorizes as well as
   (slightly better than) a 4-layer one. Depth isn't needed.
   ("ablation-importance ≠ necessity.")
7. **BPE vs char: a tie** — matched context, both ~92% recitation. Tokenization
   doesn't affect memorization.
8. **BPE trains ~3× faster** — compression gives shorter sequences for the same
   amount of text.
9. **Mechanism (opened the model):** two stages — attention makes a fuzzy,
   *content-based* locate (~60–74%); the **MLP** is a **sparse switch-bank**
   (~5 of 512 "switches" on per spot; a few dozen clean single-letter detectors,
   the rest pattern features) that turns the fuzzy guess into a crisp answer
   (~94%).
10. **We rebuilt both brains by hand** (`char_model_by_hand.py`,
    `bpe_model_by_hand.py`) — plain-numpy from the weights, **exactly** matching
    the real model on any prompt.

### The Raven (6,226 chars)

11. **Still memorized by one layer** (99.6% per-letter); 4 layers no better.
12. **Richer text was *easier*** to memorize than Seuss — Poe's variety makes
    each spot distinctive; Seuss's repetition creates confusable look-alikes.
13. **BPE-vs-char story repeats:** both memorize with one layer, a tie, BPE far
    faster — so the char/BPE findings aren't a Green-Eggs quirk.
14. **Same switch mechanism**, just more switches in use (capacity filling, not a
    new machine).

### Alice in Wonderland

*(Abridged "Storyland" 52K — internal only, per Ralph's rule; then the canonical
full 146K novel.)*

15. **The two abilities split apart:** "knows-next-letter" stays ~99% far up in
    size, but **"recite it back on its own" collapses much earlier** — because one
    slip compounds and derails the whole recitation.
16. **Recitation is a noisy "weakest-link" measure** (20K broke at 4%, 24K at
    69%) — the first-slip location is near-random. **Training loss /
    knows-next-letter are the reliable signals.**
17. **First canonical result (5K steps):** train loss rising + val loss falling
    looked like a clean memorization→generalization crossover.
18. **CORRECTION:** those were **undertrained.** Bigger texts need more steps;
    trained longer they memorize *more* (loss keeps dropping) and generalize
    *less* (val loss rises). So both the "edge" and the "generalization" were
    **fixed-budget artifacts** — retracted.
19. **Converged study (25K steps):** **no hard capacity wall found up to 146K.**
    64K fully memorized (~0), 96K nearly, 128K/146K partway — but **all still
    improving**. Memorization here is limited by **training time, not brain
    size.**

### Pending (as of this entry)

20. **VERDICT (full 146K novel, 70,000 steps).** Train-loss trajectory:
    @10K 0.72 → @20K 0.60 → @30K 0.53 → @40K 0.44 → @50K 0.37 → @60K 0.31 —
    **still declining, no plateau.** So there is **no sharp capacity cliff**, and
    "greatest memorizable length" is a *range*, not a number:
    - Fully memorizes (loss → ~0) up to **~64K**; ~96K essentially too.
    - **128K–146K never finish** — the loss keeps inching down, ever more slowly,
      holding the text increasingly imperfectly, without hitting a wall.
    - Practical answer: **this 0.24M model fully memorizes ~64K–96K characters;
      beyond that, graceful (never-completing) degradation.**

    **Interpretation:** the *shape* (a gentle, ever-steepening hill, not a cliff)
    is evidence for the mechanism of diaries 110–111: memory here is a **tuned
    lookup, not a storage bin.** A bin has a hard edge and overflows; a tuned
    lookup degrades gracefully — each extra character makes the same shared
    machinery carry a little more, a little less perfectly. So "how it stores"
    (sparse switch lookup) and "how much it stores" (soft slope, no wall) are the
    same fact seen from two sides.

### Cross-cutting lessons

- Confident hypotheses kept getting overturned by direct tests (the split
  boundary, "depth is needed," the crisp fingerprint, the capacity edge).
  **Trust the intervention over the story.**
- **Do not use the Storyland-abridged Alice** in the popular account
  (see `doc/popular_account_outline.md`).

## Source pointers

Diaries 107 (memorization), 108 (split-boundary correction), 109 (clean
BPE-vs-char + speed), 110 (mechanism, interim), 111 (MLP opened);
`py/char_model_by_hand.py`, `py/bpe_model_by_hand.py`;
`doc/popular_account_outline.md`.
