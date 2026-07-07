# A Brain You Can Read — outline for a popular account

*Working titles: "A Brain You Can Read" · "How a Machine Memorized Green Eggs and
Ham — and How We Caught It in the Act"*

Living document. Built around our two strongest assets: (1) we rebuilt the brain
by hand and *proved* it matches, and (2) we were honestly wrong along the way and
corrected by experiment. Audience: a curious non-specialist / clever adult or
child. Guiding constraint: **stay honest about scale** — this is a toy, a true
small foothold on the ideas, not "how ChatGPT works."

Arc: hook → setup → three honest surprises → the mechanism → the by-hand proof →
the generalization → honesty → a step forward → why it matters.

1. **The question.** A machine memorized *Green Eggs and Ham*, word for word.
   Could we figure out exactly *how*, well enough to be sure? Honest stakes up
   front.

2. **What "memorize" even means here.** Tiny artificial brain; the book as its
   whole world; learning to say the next letter. The mystery: it's not a file —
   the book is somehow *in the connections*.

3. **First surprise — it sounded drunk, but wasn't.** Improvising vs picking its
   best guess (temperature vs greedy). Lesson: you can be fooled about what a
   model knows by how you ask it.

4. **How far can it recite? (My first wrong answer.)** Stopped at ~90%. I said
   "ran out of brain." A plain question + a direct test proved it had simply never
   been *shown* the last part (the train/val split).

5. **How simple can the brain be? One layer is enough.** The depth surprise; a
   second correction (looked like depth mattered until we trained a shallower
   one). Measure, don't assume.

6. **Opening it up — the two helpers.** A fuzzy "where am I" glance (attention),
   then a small committee of switches voting on the next letter (the MLP). Watch
   ~5 switches light up.

7. **The proof — rebuilding the brain by hand.** Wrote the whole brain as plain
   arithmetic from its own numbers; it recites *identically*, any prompt. If you
   can rebuild it and it behaves the same, you understood it. (Climax.)

8. **Letters vs word-chunks — two brains, one trick.** char vs BPE made concrete:
   single-letter switches vs whole-chunk switches; a bigger chord for a bigger
   alphabet. The mechanism is general.

9. **What this is — and isn't.** A toy memorizing 3 KB; real models are
   unimaginably bigger and do far more than memorize. But the *ideas*
   (locate-then-decide, sparse committees, the danger of confident stories) are
   real and scale. What specialists already knew; what we re-learned by doing.

10. **A first step toward the real thing.** *(Placeholder — the small non-toy
    step.)* One careful step up from the toy, and what we hope to still be able to
    see. Keeps the account honest and open-ended.

11. **Why look inside at all.** These systems are being handed real
    responsibilities; being able to actually *see* how even a small one works —
    and to be honest about the limits — is a small act of not being fooled.

## Notes / open decisions
- Section 10's non-toy step is being designed now; whatever we choose should also
  earn its place as the account's turn toward the real world.
- Sections 3–5 carry the "wrong, then corrected" spine; section 7 is the emotional
  center.
- Source material lives in diaries 107–111 and `py/*_by_hand.py`.
