# Diary Entry 062: 2 Heads vs 4 Heads — Fewer Heads, Same Performance
## The model compensates perfectly with wider heads

### Val loss: 2-head WINS

2-head: 1.1834
4-head: 1.1845

The 2-head model is marginally BETTER. With the same total parameters
(both ~1.20M), 2 wider heads (64-dim each) perform as well as 4
narrower heads (32-dim each).

### The layer pipeline is IDENTICAL

| Layer | 2-head pattern | 4-head pattern |
|-------|---------------|----------------|
| L0 | SLF SLF | DIF DIF DIF DIF |
| L1 | FST FST | FST FST FST FST |
| L2 | FST SLF | SLF PRV SLF DIF |
| L3 | PRV SLF | PRV FST SLF PRV |
| L4 | PRV PRV | SLF PRV PRV SLF |
| L5 | FST SLF | SLF FST FST SLF |

The 2-head model has the SAME functional structure:
- L0: Local (SLF instead of DIF, but both are unfocused)
- L1: Global broadcast (FST)
- L2-4: Word processing (SLF/PRV)
- L5: Context reading (FST) + local (SLF)

The pipeline is preserved EXACTLY with half the heads.

### Key difference: L0 is SLF in 2-head, DIF in 4-head

The 4-head model's L0 never specialized (stayed DIF at 3.7b entropy).
The 2-head model's L0 developed mild SLF (self=0.20-0.21, entropy
2.8-3.0b). With fewer heads, each head is forced to develop SOME
pattern rather than remaining diffuse.

### Head specialization within Layer 4

4-head model at Layer 4, pos 5 ('i' in said):
- H0: reads 's' at 0.80 (word start)
- H1: reads 'a' at 0.73 (prev char)
- H2: reads 's' at 0.98 (word start, very focused)
- H3: reads SPACE at 0.62 (word boundary)

2-head model at Layer 4, pos 5:
- H0: reads 's' at 0.61, 'h' at 0.18
- H1: reads 's' at 0.85

BOTH heads read 's'! With only 2 heads, both focus on the most
important information (the word-start character). The 4-head model
could afford to split responsibilities (one for 'a', one for space).
The 2-head model concentrates on the MOST CRITICAL signal.

### But it still works — how?

The 2-head model doesn't read 'a' or the space explicitly at Layer 4.
So how does it know about them?

Two possibilities:
1. The 64-dim head can encode MULTIPLE reading patterns within its
   64 dimensions. Different dimensions might attend to different
   positions, even though the AVERAGE attention points to 's'.
2. Earlier layers (L2-L3) gathered 'a' and space information into the
   residual stream, so Layer 4 doesn't need to re-read them.

I suspect (2) is more important. With 2 heads per layer, each head
must be MORE EFFICIENT. The model compensates by having earlier layers
do more pre-processing.

### Logit lens: same pattern, suppression at L4

L0: 0.000  L1: 0.000  L2: 0.007  L3: 0.092  L4: 0.043  L5: 0.666

P(d) drops from 0.092 at L3 to 0.043 at L4 — Layer 4 suppresses!
Then L5 jumps to 0.666. Same suppressor pattern as the 7-layer model's
L5 (Entry 056). The 2-head model ALSO has an internal suppressor.

### The profound implication

2 heads performs identically to 4 heads at the same dim. The pipeline
structure is the same. The head specialization is less fine-grained
but sufficient. The model finds the same organizational solution
regardless of how many heads it has.

This suggests the pipeline (detect → broadcast → gather → process →
predict) is a property of the TASK and ARCHITECTURE, not of the
specific number of heads. The model discovers the same solution with
2 heads or 4 heads because it's the most efficient way to solve
character-level prediction with this architecture.

### Number of heads matters less than I thought

I expected 2 heads to force a different organization — less
specialization, different tradeoffs. Instead, the model finds the
same solution. The head count affects the GRAIN of specialization
(2 heads: coarse, 4 heads: fine) but not the STRATEGY.
