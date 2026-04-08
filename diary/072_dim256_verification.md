# Diary Entry 072: Verifying the dim=256 Functional Summary
## Do the patterns I see in the summary hold up under scrutiny?

### What the summary showed across 7 inputs

I ran the functional summary on 7 different texts for the dim=256
8L8H model. Here's what I observed:

**Pattern 1: L0 is always diffuse or mildly pos0**
"he said": 4 heads pos0, 4 diffuse
"the church": 1 head pos0, 7 diffuse
"she would": 2 heads pos0, 6 diffuse
"from the old": all diffuse
"\"yes,\" said": all diffuse
"but i do not": all diffuse
"the through": all diffuse

INCONSISTENT. With "he said" (short text), 4 heads read pos0.
With longer texts, L0 is fully diffuse. This makes sense: with more
positions, the uniform attention gives less weight to position 0,
so it falls below the 25% threshold for labeling. The pattern depends
on sequence length, not on a fundamental difference in behavior.
NOT A PROBLEM — just a labeling artifact.

**Pattern 2: L4 always does focused word reading**
Every input shows L4 reading specific word characters with very
high attention (80-100%). The specific characters read match the
current word:
- "said": H0→s(100%), H6→a(99%), H7→s(99%)
- "church": H0→c(98%), H3→r(75%), H5→r(99%), H6→u(95%)
- "would": H0→w(100%), H5→u(79%), H6→o(94%), H7→w(80%)
- "through": H0→h(96%), H2→g(84%), H4→r(79%), H5→u(69%), H6→o(86%)

CONSISTENT AND CORRECT. L4 is the word-reading layer in every case.
The specific heads point at different characters depending on the word,
but the FUNCTION (read earlier characters within the current word) is
the same.

**Pattern 3: L5 is often a suppressor**
- "he said": push=-0.06 (suppressor)
- "the church": push=-0.25 (suppressor)
- "she would": push=+1.03 (NOT suppressor)
- "from the old": push=-0.34 (suppressor)
- "\"yes,\" said": push=-0.05 (suppressor)
- "but i do not": push=+0.00 (neutral)
- "the through": push=+0.76 (NOT suppressor)

MIXED. L5 suppresses in 4 of 7 cases. It promotes in 2. It's neutral
in 1. This is MORE NUANCED than my earlier claim that "L5 is a
suppressor." L5 suppresses when the prediction is already strong
(after word reading at L4) but promotes when the word is harder to
recognize ("would" and "through" need more support).

This is actually a correction to Entry 057: suppression is not
automatic. It depends on whether the prediction NEEDS calibration.
Easy predictions get suppressed; hard predictions get boosted.

**Pattern 4: L7 (final layer) has the biggest MLP push**
- "he said": +6.42
- "the church": +10.01
- "she would": +12.31
- "from the old": +4.75
- "\"yes,\" said": +8.91
- "but i do not": +0.00 (no true next char defined)
- "the through": +10.98

CONSISTENT. The final layer always has the dominant push, ranging
from +4.75 to +12.31. This confirms Entry 053's finding that the
final layer contributes 60-77% of every prediction.

**Pattern 5: Visible fraction U-shape**
Every input shows vis% starting low (~15%), staying low through the
middle layers, and jumping at the final layer (49-77%). The U-shape
is universal.

**Pattern 6: L7 attention often reads 'h' or first characters**
For "he said": 6/8 heads read 'h'
For "the church": mixed (pos0, specific chars)
For "she would": pos0 + 's' reading
For "through": 5/8 heads read pos0

PARTIALLY CONSISTENT. The final layer reads SOMETHING from early
positions, but it's not always position 0 and not always the same
character. With 8 heads, some read pos0 while others read specific
characters from the beginning of the text.

### Confidence assessment for dim=256 understanding

HIGH CONFIDENCE:
- L4 is the word-reading layer (100% consistent across 7 inputs)
- Final layer has the dominant push (100% consistent)
- Visible fraction U-shape (100% consistent)

MODERATE CONFIDENCE:
- L5 is a context-dependent layer (suppresses sometimes, promotes others)
- L0-L2 are broadcasting/gathering layers (consistent but threshold-dependent)

LOW CONFIDENCE:
- The specific function of L6 (it varies substantially across inputs)
- Whether L3 in the dim=256 model does identity erasure like L3 in dim=128

### Conclusion

The functional summary gives me ACCURATE but INCOMPLETE understanding
of the dim=256 model. The core findings (word reading at L4, final
layer dominance, U-shape) are solid. The layer-by-layer details (L5
suppression, L6 role) are more nuanced than the dim=128 analysis
revealed. The tool extends my range but doesn't give me the same
depth of understanding I had at dim=128.

This is honest and expected. The tool compresses; compression loses
detail. But the details it preserves are the RIGHT ones for identifying
the pipeline structure.
