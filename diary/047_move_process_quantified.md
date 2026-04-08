# Diary Entry 047: The Move-Process Pattern — Quantified
## Attention/MLP norm ratios across all layers

### The numbers

Averaged across 5 different sentences:

| Layer | Attn/MLP ratio | Dominant | Role |
|-------|---------------|----------|------|
| L0 | 0.75 | MLP | Character detection |
| L1 | 5.02 | Attention | Global broadcast |
| L2 | 2.10 | Attention | Previous word gathering |
| L3 | 1.18 | Mixed | Identity erasure + prediction |
| L4 | 0.62 | MLP | Pre-recognition |
| L5 | 0.31 | MLP | Word recognition + prediction |

### The pattern is confirmed but more nuanced

The alternation I hypothesized (Move-Process-Move-Process) is NOT
perfectly clean. Here's what actually happens:

**L0 (MLP=0.75)**: MLP-dominated. Character detection. ✓
**L1 (Attn=5.02)**: STRONGLY attention-dominated. The attention output
is 5× larger than the MLP. This is the most lopsided layer — it's
almost purely an information-moving layer. ✓
**L2 (Attn=2.10)**: Moderately attention-dominated. Gathers cross-word
context but also has some MLP contribution. ✓
**L3 (Mixed=1.18)**: Nearly equal! Both attention and MLP contribute
substantially. This makes sense: L3 needs to BOTH gather information
(attention reads word characters) AND process it (MLP erases identity
and writes predictions). It's doing both simultaneously.
**L4 (MLP=0.62)**: MLP-dominated. I had called this "Gather current
word" (Move), but the ratio says it's actually Process-dominated.
**L5 (MLP=0.31)**: Strongly MLP-dominated. The MLP is 3× larger than
attention. This is where the heavy prediction computation happens. ✓

### Correcting Entry 046

My earlier claim was:
"MLP-dominated layers (L0, L3, L5) PROCESS; attention-dominated layers
(L1, L2, L4) MOVE"

The corrected picture:
- L0: MLP Process ✓
- L1: Attention Move ✓ (very strongly)
- L2: Attention Move ✓
- L3: Both — simultaneous Move+Process
- L4: MLP Process (NOT attention-dominated as I claimed)
- L5: MLP Process ✓ (very strongly)

Layer 4 is MLP-dominated, not attention-dominated. I made the wrong
assignment because Layer 4's attention is sharply FOCUSED (single
positions at 0.80-0.98), which made it seem important. But sharp
attention with a small output norm means the attention reads a specific
position but doesn't write much to the residual stream. The MLP does
the heavy lifting at Layer 4 too.

### The real pattern: Attention front-loads, MLP back-loads

Layers 1-2: attention > MLP (gathering phase)
Layers 3-5: MLP ≥ attention (processing phase)

The model gathers information in early layers (via attention) and
processes it in later layers (via MLP). This is a FRONT-LOADED
gathering, BACK-LOADED processing architecture.

Layer 3 is the transition point where both are roughly equal.

### Why Layer 1's attention is 5× larger than its MLP

Layer 1 has the most extreme ratio (5.02). Its MLP is nearly dead
(only 2-3 neurons fire, Entry 004). Layer 1 exists almost entirely
to MOVE information — specifically, to broadcast position 0's
representation to all positions. Its MLP has been trained to near-zero
because there's nothing useful for it to compute at this stage.
