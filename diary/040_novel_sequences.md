# Diary Entry 040: Handling Novel and Alien Sequences
## The model gracefully degrades on unknown words

### Real word: "church"
u→r: 0.95, r→c: 0.98, c→h: 1.00 — recognized with near-certainty.
Entropy drops from 2.5 bits at 'c' to 0.0 bits at the final 'h'.

### Fake but English-like: "blurch"
u→r: 0.01, r→c: 0.02, c→h: 0.32 — the model does NOT recognize
"blurch" as a word (it isn't). But it partially recovers at 'ch'
because 'ch' is a common English bigram regardless of context.

### Fake but English-like: "fremble"
b→l: 0.80! The model predicts 'l' after 'b' because 'bl' is a common
English cluster (presumably from -ble endings like "terrible",
"humble"). And r→e: 0.38 — 're' is common. The model uses sub-word
patterns even for unknown words.

### Alien: "xzqwp"
Every prediction is 0.00. Entropy stays high (1.1-2.9 bits). But
interestingly, after 'q' the model STILL predicts 'u' (entropy drops
to 1.1 bits). The qu association is so strong it fires even in a
completely alien context.

### Repeated consonant: "bbbbb"
Every b→b prediction is 0.00. But the model's predictions for what
SHOULD follow each 'b' are sensible: 'o', 'e', 'l', 'e' — vowels
and common followers. It keeps trying to make English out of nonsense.

### Rare real word: "rhythm"
y→t: 0.50, t→h: 0.98, h→m: 0.92 — the model recognizes "rhythm"!
It struggles at the start (r→h: 0.01) because 'rh' is rare, but
once it sees 'rhy', the word ramps up quickly. By 'th' it's at 0.98.

### Very rare: "quixote"
x→o: 0.89, o→t: 0.72 — the model knows "quixote"! The prefix 'qui'
is ambiguous (could be many words), and 'ix' is very unusual, but
'xo' and 'ot' are distinctive enough that the model recovers.

### What this reveals

**The model ALWAYS tries to make English.** Even for "xzqwp", it
predicts English-like continuations ('u' after 'q', 'i' after 'x').
It has no concept of "this isn't a word" — it just applies its
character statistics regardless.

**Entropy is the uncertainty signal.** Known words have entropy near
0 bits at recognition points. Unknown words stay at 2-3 bits. This
is the measurable difference between recognition and confusion.

**Sub-word patterns survive even in nonsense.** The model uses 'ch',
'bl', 're', 'th' patterns even in fake words. These bigram/trigram
associations are baked into the weights and fire automatically.

**Word recognition is robust to unusual beginnings.** Both "rhythm"
(r→h is rare) and "quixote" (qu→i→x is very unusual) are recognized
once enough disambiguating characters appear. The model can recover
from an uncertain start.

**There's no "novelty detector."** The model doesn't know when it's
processing something new. It just gets more uncertain (higher entropy)
without any explicit signal that this sequence is unusual. This is a
limitation — a more capable model might benefit from knowing "I haven't
seen this pattern before."
