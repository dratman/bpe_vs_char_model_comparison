# Diary Entry 059: Predictions for 3-Layer and 4-Layer Models
## What I expect before seeing the results

### 4-layer model predictions

I predict the 4-layer model will show a COMPRESSED version of the
6-layer pipeline:

- L0: Character detection (same as 6L L0)
- L1: Combined context broadcast + gathering (merging 6L L1+L2)
- L2: Identity erasure + tentative prediction (same as 6L L3)
- L3: Word recognition + final prediction (same as 6L L5)

The pre-recognition role (6L L4) will be absorbed into L3.
Val loss: ~1.25 (worse than 6L's 1.18 but functional).

I predict Layer 3 will carry the full prediction burden — contributing
70%+ of every correct prediction, since there's no room for a
suppressor layer.

### 3-layer model predictions

I predict the 3-layer model will be CRAMPED — trying to fit 4 essential
roles into 3 layers:

- L0: Character detection (essential, can't be compressed)
- L1: Everything in the middle (context + erasure + gathering)
- L2: Word recognition + prediction

Val loss: ~1.35-1.40 (notably worse).

The key prediction: L1 will be OVERLOADED. It will show mixed attention
patterns (some heads FST, some SLF/PRV) because it's trying to do
multiple jobs. The attention/MLP ratio at L1 will be near 1.0 (neither
dominated) because both components are needed.

Word recognition will be weaker — the model won't have enough depth
to recognize words from 3+ character prefixes. "said" might work
(short, common), but "through" and "rhythm" probably won't.

### What would CHANGE my understanding

If the 3-layer model shows the SAME pipeline structure as the 6-layer
model (just compressed), that would mean the pipeline is a very robust
organizational principle. If it shows a DIFFERENT organization, that
would mean the pipeline depends on having sufficient depth.

If the 4-layer model achieves val loss close to the 6-layer model
(say < 1.20), that would suggest the 6-layer model wastes capacity
on calibration/suppression that isn't essential for prediction quality.
