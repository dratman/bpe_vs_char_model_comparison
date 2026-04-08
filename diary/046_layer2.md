# Diary Entry 046: Layer 2 — The Transition Layer
## What does Layer 2 actually do?

### Layer 2's attention is 2-3× larger than its MLP

The attention output norm is 2-3× the MLP output norm at every position.
This is DIFFERENT from other layers:
- Layer 0: MLP is larger (character detectors fire strongly)
- Layer 5: MLP is larger (word prediction writes strongly)
- Layer 2: Attention dominates

Layer 2 is primarily an ATTENTION layer. Its MLP contributes relatively
little.

### Layer 2 attention looks BACKWARD, not at position 0

Unlike Layer 1 (which reads position 0 with 60-66% attention), Layer 2
reads SPECIFIC previous characters:

At position 2 (' ' after "he"):
- H0: 'h'=0.64 — reads the first letter of the previous word
- H1: 'h'=0.53 — also reads 'h'
- H2: 'e'=0.83 — reads the second letter of "he"
- H3: 'h'=0.65 — reads 'h' again

Three of four heads read back to 'h' (the start of "he"). One head
reads 'e'. Layer 2 is gathering information about the PREVIOUS WORD
at the space position.

At position 5 ('i' in "said"):
- H0: 's'=0.22, 'h'=0.21, 'i'=0.21 — spread across recent chars
- H1: ' '=0.26, 'i'=0.22, 'a'=0.20 — reads space and nearby
- H2: 'e'=0.52 — reads 'e' from "he"!
- H3: 'h'=0.42 — reads 'h' from "he"

H2 at position 5 reads 'e' from "he" with 52% attention. It's reaching
back across a word boundary to read the PREVIOUS word. This is different
from Layer 4's within-word reading.

### Layer 2's attention pushes toward 'd' at position 5

The attention output at position 5 ('i') has its top character alignment
as 'd' (+0.48). This is the CORRECT next character for "said"!

But wait — Layer 2 is the third layer, and the logit lens showed
P(d)=0.03 after Layer 2 (Entry 012). The attention is pushing toward
'd', but the MLP and residual stream are still dominated by character
identity. The 'd'-alignment is small compared to the residual's overall
direction.

### Layer 2's MLP is weak and generic

The MLP outputs push toward ')' and ' ' at most positions — these are
generic predictions, not character-specific. The MLP at Layer 2 is not
doing significant computation. It's writing small, generic adjustments.

### Layer 2's role: cross-word information gathering

Layer 2 bridges BETWEEN words. While Layer 4 reads WITHIN the current
word (s, a), Layer 2 reads the PREVIOUS word (h, e). This provides
cross-word context: "what word came before this one?"

This cross-word information is written to the residual stream (via the
attention output, which is 2-3× larger than the MLP output). Later
layers can use this information for predictions that depend on word
pairs (like "he said" where knowing "he" helps predict "said").

### The pipeline refinement

With Layer 2 now understood:

- **L0**: Character detection (MLP-dominated)
- **L1**: Position-0 global broadcast (attention-dominated, FST)
- **L2**: Cross-word context gathering (attention-dominated, SLF/PRV)
  — reads the previous word's characters
- **L3**: Identity erasure + tentative prediction (MLP-dominated)
- **L4**: Within-word reading (attention-dominated) + pre-recognition
- **L5**: Word recognition + context reading (MLP-dominated)

There's an alternation: MLP-dominated layers (L0, L3, L5) do
PROCESSING, while attention-dominated layers (L1, L2, L4) do
INFORMATION MOVEMENT.

### The alternation pattern

| Layer | Dominant component | Role |
|-------|-------------------|------|
| L0 | MLP | Process: detect characters |
| L1 | Attention | Move: broadcast position 0 |
| L2 | Attention | Move: gather previous word |
| L3 | MLP | Process: erase identity, sketch prediction |
| L4 | Attention | Move: gather current word chars |
| L5 | MLP | Process: recognize word, write prediction |

This Move → Process alternation is not designed — it emerged from
training. But it makes functional sense: you need to GATHER information
before you can PROCESS it.
