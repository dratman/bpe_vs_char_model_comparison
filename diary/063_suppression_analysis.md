# Diary Entry 063: Is Suppression Targeted or Generic?
## Testing the "antagonist muscle" hypothesis

### The hypothesis

Like muscles that work in opposing pairs (bicep/tricep), maybe the
suppressor layers push toward a SPECIFIC alternative — the second-best
linguistically plausible prediction. If so, suppression would have a
PURPOSE beyond just reducing confidence.

### What the suppressors actually boost

| Word | True | Suppressor | Boosts |
|------|------|-----------|--------|
| said (i→d) | d | L0 | n, i, o, e, t |
| would (l→d) | d | L3 | ', j, ", \n, b |
| through (g→h) | h | L2 | SPACE, comma, ), ;, : |

### The verdict: suppression is NOT targeted

The boosted characters are NOT linguistically meaningful alternatives:
- For "said" (L0 suppresses 'd'): boosts n, i, o, e, t — these are
  just common letters, not alternatives to 'd' in "said"
- For "would" (L3 suppresses 'd'): boosts ', j, " — punctuation and
  rare characters. Nobody would spell "woulj" or "woul'"
- For "through" (L2 suppresses 'h'): boosts SPACE, comma, ) —
  punctuation. "throug," makes no sense

The suppressors boost ARBITRARY characters. They're not pushing toward
a plausible alternative — they're just reducing the dominant prediction
by dumping probability mass onto whatever characters happen to be
easiest to boost (often punctuation, which has large embedding norms).

### The antagonist muscle analogy doesn't hold

In the body, bicep and tricep have EQUAL AND OPPOSITE purposes — one
flexes, one extends. In this model, the promoter layers have a specific
purpose (predict 'd' because the word is "said") but the suppressor
layers have NO specific counter-purpose. They just say "be less sure"
without advocating for any particular alternative.

### A better analogy: thermostat

The suppressor is more like a thermostat than an antagonist muscle.
A thermostat doesn't care what temperature the room should be — it
just prevents the heater from making the room too hot. Similarly,
the suppressor doesn't care what character should come next — it just
prevents the model from being too confident about any single character.

The "heating" (prediction) is targeted and purposeful.
The "cooling" (suppression) is generic and regulatory.

### Why generic suppression works

The loss function (cross-entropy) penalizes overconfidence uniformly.
If P(true) = 0.95 but the actual frequency is 0.70, the model loses
the same amount regardless of WHERE the excess probability goes. So
the model doesn't need to redistribute probability to specific
alternatives — it just needs to reduce the peak.

Dumping probability onto punctuation is EFFICIENT because punctuation
embeddings have large norms (Entry 020), making them easy to boost
with small weight changes. The suppressor takes the path of least
resistance.

### But there IS one partial exception

For "said", L0's suppression boosts 'n' and 'i'. 'n' is actually the
bigram prediction (i→n is common). And 'i' is the current character.
These aren't random — they're the "default" predictions before word
recognition kicks in. L0 is suppressing 'd' not because it's wrong,
but because L0 HASN'T DONE WORD RECOGNITION YET. At L0, the most
natural predictions are the bigram ('n') and the identity ('i'). The
suppression of 'd' at L0 isn't really suppression — it's just the
absence of word recognition signal. L0 hasn't learned that 'd' should
follow 'sai' because that's a word-level computation.

This means: what looks like "suppression" at early layers might
actually be "ignorance" — the layer hasn't computed the word-level
prediction yet, so it defaults to character-level predictions that
happen to oppose the correct word-level answer.

### Correcting my understanding

TRUE suppression (deliberately reducing confidence) happens at:
- L5 in the 7-layer model (Entry 057): MLP reads the dominant
  prediction and actively opposes it

FALSE suppression (actually just ignorance/default) happens at:
- L0 in the 6-layer model: hasn't done word recognition, defaults
  to bigrams which happen to oppose the word-level answer

The distinction matters: true suppressors emerge only when there's
a later layer to make the final call. Early layers aren't suppressing
— they're just not yet predicting.
