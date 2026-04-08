# Diary Entry 008: Ablation Reveals Internal Conflict
## The model argues with itself about what follows 'th'

### The surprising finding: L3H3 fights against 'e'

Removing L3H3 INCREASES P(e|th) from 0.624 to 0.733. This head is
actively suppressing the correct prediction. But it's not pushing toward
a specific alternative — 'e' is still the top prediction with or without
L3H3.

### L3H3 is a general-purpose suppressor

Testing L3H3's ablation across multiple bigrams:

| Context | Base P(true) | Without L3H3 | Change |
|---------|-------------|-------------|---------|
| th→e    | 0.624       | 0.733       | +0.108  |
| sh→e    | 0.344       | 0.411       | +0.067  |
| he→SPACE| 0.643       | 0.814       | +0.171  |
| th→a    | 0.108       | 0.105       | -0.003  |
| th→i    | 0.077       | 0.066       | -0.011  |

L3H3 suppresses the DOMINANT prediction in every case. When 'e' is the
likely answer (th→e, sh→e), it pushes P(e) down. When SPACE is the likely
answer (he→SPACE), it pushes P(SPACE) down even more strongly (+0.171).

This is not a bug — it's a CALIBRATION mechanism. Without L3H3, the model
would be overconfident. L3H3's job is to reduce the probability of the
most likely next character, spreading probability mass to alternatives.
It's implementing a form of uncertainty.

This makes sense: 'th' is followed by 'e' 62% of the time, but also by
'a' (11%), 'i' (8%), 'o', etc. The model's raw circuits might predict
'e' with 73% confidence, and L3H3 brings it down to 62%, which is closer
to the true corpus distribution.

### L2H0 is the 'e-after-consonant-cluster' expert

L2H0 has the opposite effect — removing it drops P(e|th) by 0.10:

| Context | Base P(true) | Without L2H0 | Change |
|---------|-------------|-------------|---------|
| th→e    | 0.624       | 0.525       | -0.099  |
| sh→e    | 0.344       | 0.278       | -0.066  |
| he→SPACE| 0.643       | 0.651       | +0.008  |

L2H0 boosts 'e' specifically after consonant pairs (th, sh) but has
NO effect on the SPACE prediction after 'he'. It has learned something
specific: "after two consonants, 'e' is likely." This is a genuine
phonological pattern — in English, consonant clusters are very often
followed by vowels, and 'e' is the most common vowel.

### Bigram knowledge is real and linguistically accurate

The model's bigram predictions match English:

| Bigram | Top prediction | Accuracy |
|--------|---------------|----------|
| th→    | e (0.62)      | Correct: the, them, then, there |
| sh→    | e (0.34)      | Correct: she, shed, shelf |
| wh→    | i (0.44)      | Correct: which, while, white |
| ch→    | SPACE (0.49)  | Correct: many words end in -ch |
| qu→    | e (0.36)      | All correct: que-, qui-, qua- |
| pr→    | o (0.42)      | Correct: pro-, prov-, prob- |
| he→    | SPACE (0.64)  | Correct: "he" is a complete word |
| ha→    | t (0.35)      | Correct: hat, had, has |
| hi→    | s (0.41)      | Correct: his, hist- |
| ea→    | r (0.25)      | Correct: ear, early, earth |

The model has internalized the statistical structure of English letter
combinations. It knows 'wh' is usually followed by 'i' (which, while),
not 'e' (where, when are less common). It knows 'hi' is followed by 's'
(his). These are not trivial predictions — they reflect real frequency
distributions.

### The internal economy of prediction

The model doesn't compute predictions with a single circuit. It has:

1. **Boosters** (L2H0, L2H1): Push probability toward the most likely
   next character based on local bigram patterns.
2. **Suppressors** (L3H3, L1 heads): Pull probability away from the
   dominant prediction, redistributing mass to alternatives.
3. **Context readers** (L5H1): Adjust predictions based on longer-range
   context (quotation marks, sentence structure).

The final prediction is the sum of these competing forces. The model
literally argues with itself: "L2H0 says 'e' is very likely, L3H3 says
don't be so sure, L5H1 says check the wider context." The residual
stream accumulates these votes, and the final logits reflect the balance.

### Through: evidence of sub-word chunking

The 'through' predictions show something interesting:
- t→h: 0.33 (just a bigram, same as 'the')
- h→r: 0.02 (model expects 'e', gets 'r' — surprised)
- r→o: 0.65 (now it's tracking 'thr...' → 'o' is likely)
- o→u: 0.86 ('thro' → 'u' is very likely, it's seeing 'through')
- u→g: 0.99 ('throu' → almost certainly 'through')
- g→h: 1.00 ('throug' → must be 'h')

The model recovers from its initial surprise at 'r' by recognizing
'thro' as a strong predictor of 'through'. The confidence ramps up
character by character: 0.02 → 0.65 → 0.86 → 0.99 → 1.00. By the
fourth character, the word is essentially certain.

This is NOT look-up — the model has never seen a word token. It's
doing sequential Bayesian updating: each new character narrows the
space of possible words, and the model's confidence rises accordingly.
