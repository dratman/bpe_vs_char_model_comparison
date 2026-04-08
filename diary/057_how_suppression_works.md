# Diary Entry 057: How the Suppressor Actually Works
## It suppresses the TOP predictions and boosts low-probability alternatives

### What Layer 5 does in the 7L model at position 5 ('i' in "said")

The residual stream entering Layer 5 already predicts:
- 'd' at +0.844 (correct answer, strongly predicted)
- 'c' at +0.502
- 'n' at +0.463

Layer 5's MLP then SUPPRESSES the top predictions:
- 'd': pushed by -0.079
- 'c': pushed by -0.166
- 'n': pushed by -0.178

And BOOSTS low-probability alternatives:
- 'a': pushed by +0.623 (was not in the top)
- "'": pushed by +0.620
- 'o': pushed by +0.422

### The suppressor reads the residual and pushes AGAINST it

This is the key mechanism. Layer 5's MLP takes the residual stream
as input (through the layer norm and linear projection). The residual
stream encodes "d is the most likely prediction." The MLP has learned
to OUTPUT a vector that opposes this — pushing d down and boosting
alternatives like 'a' and 'o'.

It's like a contrarian: whatever the current consensus is, it argues
against it.

### HOW does it know what to suppress?

The MLP receives the 128-dimensional residual through layer norm,
then projects to 512 dimensions (the neurons), applies GELU, and
projects back to 128 dimensions. The WEIGHTS of these projections
determine the suppression behavior.

Specifically:
1. The input projection (c_fc) transforms the residual into neuron
   activations. Neurons that respond to "d is strongly predicted"
   will fire.
2. The GELU determines which neurons are active.
3. The output projection (c_proj) maps the active neurons to a
   128-dim vector that happens to point AWAY from 'd' and toward
   alternatives like 'a'.

The suppressor "knows" what to suppress because the c_fc weights
have been trained to detect "strong d prediction in the residual"
and the c_proj weights have been trained to output "push away from d"
when those neurons fire.

### This is trained, not designed

Nothing in the architecture says "Layer 5 should suppress." During
training, the model discovered that having a suppression layer before
the final prediction layer produces better calibrated probabilities.
If the 6-layer model tends to be overconfident about 'd' (predicting
70% when the true probability is, say, 60%), then adding a suppression
layer that brings it down improves the loss.

### Why boost 'a' and 'o' specifically?

Layer 5 boosts 'a' (+0.623) and 'o' (+0.422). These are plausible
alternatives to 'd' after 'sai':
- 'a' doesn't form a real word ("saia"?) but might be seen in
  unusual contexts
- 'o' doesn't form a real word either

Actually, these alternatives might not make linguistic sense. The
suppressor might be boosting ARBITRARY alternatives just to spread
probability mass away from 'd', without caring about WHICH alternatives
get the mass. The specific characters it boosts ('a', "'", 'o') might
just be the easiest directions for the c_proj weights to push toward,
not linguistically meaningful choices.

### The suppressor is a STATISTICAL calibrator, not a linguistic one

It reads "the residual is very confident about d" and outputs "be
less confident." The specific alternatives it boosts may be accidental
properties of the weight matrix rather than meaningful linguistic
alternatives. The IMPORTANT thing is that it reduces the top
prediction's probability; which alternatives gain is secondary.

### This answers the original question

"How does a suppressor know what to suppress and how?"

1. **What to suppress**: Whatever the residual stream is most
   confident about. The c_fc weights detect the dominant prediction.

2. **How**: By outputting a vector that points AWAY from the dominant
   prediction direction. The c_proj weights are trained to oppose
   strong signals in the residual.

3. **Why**: To produce better-calibrated probabilities. Overconfidence
   hurts the model's loss function.
