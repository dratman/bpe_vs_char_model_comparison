# Diary Entry 045: 7-Layer Model — Early Look at Step 7000
## Checking predictions against Entry 039 while training continues

### Val loss at step 7000: 1.276

Already well below the 6-layer model's final val loss (1.185 at 30k).
The 7-layer model is learning FASTER. My Prediction 6 (slightly better
val loss) seems confirmed — and it may end up substantially better,
not just slightly.

### Attention pattern at step 7000

| Layer | H0 | H1 | H2 | H3 |
|-------|-----|-----|-----|-----|
| L0 | DIF 3.7b | DIF 3.7b | DIF 3.7b | DIF 3.7b |
| L1 | DIF 3.3b | FST 3.6b | DIF 3.6b | DIF 3.7b |
| L2 | DIF 3.7b | SLF 2.8b | SLF 3.0b | FST 3.6b |
| L3 | PRV 1.8b | SLF 1.7b | DIF 2.9b | DIF 3.5b |
| L4 | SLF 0.4b | PRV 1.4b | SLF 1.2b | SLF 1.0b |
| L5 | DIF 2.3b | SLF 2.4b | DIF 2.3b | SLF 1.3b |
| L6 | FST 2.7b | DIF 3.0b | FST 3.0b | FST 3.0b |

### Checking my predictions

**Prediction 1: L0 will still be diffuse** ✓
L0 is completely DIF at 3.7b entropy. Same as the 6-layer model.

**Prediction 2: L1 will still broadcast position 0** PARTIAL
L1 H1 is FST, but the other three heads are DIF. In the 6-layer model,
ALL four L1 heads were FST by step 7000. The 7-layer model is developing
the FST pattern more slowly.

**Prediction 3: Final layer will be the word recognizer** LIKELY ✓
L6 has three FST heads — it looks like it's developing the context-
reading role that L5 had in the 6-layer model. The FST pattern at the
final layer matches my prediction.

**Prediction 7: Identity erasure at L3 or L4** CHECKING
L3 has PRV (H0) and SLF (H1) — this looks like the beginning of
word-internal processing. L4 is strongly focused (SLF/PRV, entropies
0.4-1.4b) — this is the most specialized layer, just like in the
6-layer model.

### The pipeline looks similar but SHIFTED

Tentative mapping of 7-layer to 6-layer roles:

| Role | 6-layer | 7-layer |
|------|---------|---------|
| Character detection | L0 | L0 |
| Global broadcast | L1 | L1 (developing) |
| Transition | L2 | L2-L3 |
| Word processing | L3-L4 | L3-L4 |
| Context reading | L5 | L5-L6 |

The extra layer seems to have been "inserted" between the transition
and context-reading stages. L5 in the 7-layer model shows a MIX of
SLF and DIF — it's not purely FST like the 6-layer L5. Instead, the
FST context-reading role has moved to L6 (the new final layer).

### The most interesting change: L5 has a new role

In the 6-layer model, L5 was the combined word-recognizer and context-
reader. In the 7-layer model, L5 is developing SLF/DIF patterns (not
FST). This suggests the 7-layer model SPLITS L5's dual role:
- L5: Word recognition (SLF — reading within the word)
- L6: Context reading (FST — reading global context)

If this holds, the 7-layer model should have BETTER separation of
concerns. The 6-layer model's L5 had to do both word recognition AND
context reading. The 7-layer model can dedicate separate layers to each.

### Still training — will check again when complete

Current step: ~7000 of 30000. The model is still developing. The final
pattern may differ from this early snapshot. I'll do a full comparison
when training completes.
