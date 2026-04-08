# Diary Entry 027: What Layer 3 Actually Writes
## Decomposing L3's contribution to the residual stream

### L3 writes AGAINST the residual direction

cos(L3_addition, residual_before) = -0.357

Layer 3's contribution is anti-aligned with what's already in the
residual stream. It's not reinforcing — it's REDIRECTING the residual.
The residual going into L3 still looks like 'i' (the input character).
L3 pushes it AWAY from 'i' and toward something else.

### L3 suppresses the input character and boosts predictions

L3's total contribution to character logits:

**Boosted**: ':' (+0.57), 'x' (+0.50), 'n' (+0.49), '-' (+0.45)
**Suppressed**: 'a' (-0.48), 'i' (-0.35), 'y' (-0.29), 'h' (-0.19)

L3 is doing two things simultaneously:
1. Suppressing the input character 'i' (-0.35) and its predecessor 'a'
   (-0.48). It's saying "forget what character you are."
2. Boosting 'n' (+0.49) — the bigram prediction. But also boosting
   punctuation and rare characters (':',  'x', '-').

### The attention and MLP disagree within Layer 3

L3's attention boosts: ':', 'j', "'", 'i', '!'
L3's MLP boosts: 's', 'x', 'n', '-', 'l'
L3's MLP suppresses: 'j' (-0.44), 'y' (-0.47), 'i' (-0.68)

cos(attn, mlp) = -0.225 — they're slightly anti-aligned!

The attention actually boosts 'i' (+0.33) while the MLP strongly
suppresses it (-0.68). They're working at cross purposes on the current
character. The MLP wins because it suppresses 'i' harder (-0.68) than
attention boosts it (+0.33), resulting in net suppression (-0.35).

This internal tug-of-war within Layer 3 explains why its total
contribution is complex rather than simple. The attention is preserving
some character identity while the MLP is actively erasing it.

### The critical effect: Layer 3 erases 'i' and installs 'n'

**Without L3**, the residual at position 5 aligns with:
  i=+0.50, n=+0.31, t=+0.28, d=+0.13

**With L3**, it aligns with:
  n=+0.80, t=+0.47, d=+0.41, s=+0.31

Layer 3 transforms the residual from "I am the character i" to "the
next character is probably n (but d and t are possible)." The character
identity 'i' drops from dominant (+0.50) to negligible, replaced by
a PREDICTION about what comes next.

This is the ESSENTIAL transformation that Layer 3 provides. Without it,
Layer 4 receives a residual that still thinks it's 'i'. Layer 4's
attention heads (which read previous word characters) don't work well
when the residual is still anchored to character identity — they need
a representation that has already been partially transformed into a
prediction, which they can then CORRECT.

### Why Layer 3 is essential even though its prediction is wrong

Layer 3 does two critical things:
1. **Erases character identity**: Pushes the residual away from 'i'
   so later layers can write their own predictions without fighting
   the input signal.
2. **Installs a hypothesis**: Writes 'n' as the dominant prediction.
   This is wrong, but it gives Layers 4-5 something specific to
   OVERRIDE. Overriding a wrong prediction (n→d) is easier than
   building a prediction from scratch on top of raw character identity.

Think of it as: Layer 3 clears the canvas and sketches a rough draft.
Layer 4 corrects the draft. Layer 5 finalizes it. Without the rough
draft, the later layers can't efficiently do their corrections.

### The residual stream is being RECYCLED

This reveals something fundamental about how residual streams work.
The 128-dimensional residual space serves multiple purposes at
different layers:
- After L0-2: encodes "what character is here" (input identity)
- After L3: encodes "what character comes NEXT" (prediction)
- After L4-5: encodes a refined prediction

The same 128 dimensions are reused for different information at
different stages. Layer 3's job is the TRANSITION: it converts the
residual from encoding the present character to encoding the predicted
next character. This transition is essential — it's why skipping
Layer 3 breaks the model.

### Comparison with the noise experiment (Entry 025)

In Entry 025, noise of std=0.1 (11% of signal) broke word recognition
by flipping 46 neurons in Layer 4. Now I can see WHY those neurons are
sensitive: they're calibrated to receive a residual that has been
TRANSFORMED by Layer 3 from character-identity to prediction-hypothesis.
If Layer 3's transformation is perturbed, Layer 4's neurons see inputs
they weren't trained on, and they flip unpredictably.
