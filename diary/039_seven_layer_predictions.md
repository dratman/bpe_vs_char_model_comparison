# Diary Entry 039: Predictions for the 7-Layer Model
## What my understanding predicts BEFORE seeing the results

Training a 7-layer model (same dim=128, 4 heads, Gutenberg corpus).
Before looking at any results, here are my predictions based on
the 6-layer understanding:

### Prediction 1: Layer 0 will still be a character detector
The character detection role (sparse MLP neurons, diffuse attention)
should be the same. This is a fundamental input-processing step that
doesn't depend on model depth.

### Prediction 2: Layer 1 will still broadcast position 0
The global context distribution role should persist. The model needs
to distribute position-0 information regardless of depth.

### Prediction 3: The final layer will still be the primary word recognizer
In the 6-layer model, Layer 5 handles 8/10 words. In the 7-layer model,
I predict Layer 6 (the new final layer) will take this role. Layer 5
will become more like the current Layer 4 (pre-recognition, calibration).

### Prediction 4: The dark subspace U-shape will extend
The dark fraction should peak somewhere in the middle (L2-4) and
decrease at the final layer. The U-shape should stretch to accommodate
the extra layer, with the same pattern but spread over 7 layers.

### Prediction 5: Word recognition might improve
With an extra layer, the model has one more level of processing.
Words that the 6-layer model struggles with (like "did", which only
reaches P(d)=0.07) might get better. The extra depth allows more
sequential evidence integration.

### Prediction 6: The val loss should be slightly better
More depth usually helps (diminishing returns). I'd expect val loss
around 1.15-1.17 (vs 1.18 for the 6-layer model).

### Prediction 7: The identity erasure will happen at Layer 3 or 4
Layer 3 erases character identity in the 6-layer model. With 7 layers,
this role might shift to Layer 4, keeping the same pipeline structure
but with each stage having slightly more room.

### What would SURPRISE me
- If the extra layer is completely unused (dead)
- If the pipeline ORDER changes (e.g., word recognition before
  identity erasure)
- If val loss gets WORSE (would indicate the model is too deep for
  128 dims)
- If the dark subspace grows monotonically without the U-shape

I'll check each prediction when the training finishes.
