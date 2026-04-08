# Diary Entry 038: Layer 5 Also Has Word-Specific Ensembles
## Confirmed: both recognition layers use distributed, independent coding

### Layer 5 activation patterns are word-specific

Top 5 neurons at Layer 5 for each word:

| Word | Top neurons |
|------|------------|
| said | n266, n115, n183, n492, n397 |
| through | n187, n440, n87, n316, n346 |
| strange | n239, n104, n29, n282, n443 |
| quick | n40, n7, n446, n453, n123 |
| nothing | n146, n350, n94, n23, n16 |
| would | n420, n77, n37, n50, n385 |
| himself | n374, n326, n437, n220, n337 |

Zero overlap in top-5 neurons. Average pairwise cosine: +0.017
(essentially zero). Layer 5 uses the same word-specific ensemble
strategy as Layer 4.

### Layer 5 is even sparser than Layer 4 for some words

- "said" at L5: only 61 active neurons (vs 101 at L4)
- "quick" at L5: only 56 active (vs 77 at L4)

But for other words it's denser:
- "nothing" at L5: 134 active (vs 88 at L4)
- "through" at L5: 128 active (vs 76 at L4)

The harder the word is to recognize (more characters needed, later
recognition), the MORE neurons Layer 5 activates. "through" and
"nothing" are long words recognized only at Layer 5, so they need
more neurons. "said" and "quick" have had some processing by Layer 4,
so Layer 5 needs fewer neurons to finish the job.

### Shared neurons across words: almost none

At Layer 5, only 8 neurons are active (>0.5) in 3+ of the 7 words.
Zero neurons are active in 4+ words. Each word's ensemble is almost
completely private.

At Layer 4, only 2 neurons were shared across 3+ words.

### Both layers use the same representational strategy

| Property | Layer 4 | Layer 5 |
|----------|---------|---------|
| Avg pairwise cosine | -0.001 | +0.017 |
| Top-5 overlap | 0 pairs | 0 pairs |
| Shared neurons (3+) | 2 | 8 |
| Active neurons range | 76-108 | 56-134 |

Near-orthogonal, non-overlapping ensembles at both layers. The model
has learned to use independent neuron populations for different words
at every recognition layer.

### Can 512 neurons support thousands of words?

With ensembles of 60-130 neurons and near-zero overlap, naive counting
suggests the model can handle 4-8 words simultaneously. But:

1. Only ONE word is processed at each position at a time
2. The 62% transition-zone neurons (Entry 024) allow graded sharing
3. The ensembles are not binary — activation LEVELS encode information

So the effective vocabulary is much larger than 4-8. The model processes
thousands of different words by using different activation patterns
across the same 512 neurons, with different words occupying different
regions of the activation space. The near-orthogonality ensures that
different words don't interfere with each other.

### This is a distributed code, not a localist code

In neuroscience terms:
- **Localist coding**: one neuron = one word (like "grandmother cells")
- **Distributed coding**: each word = a pattern across many neurons

Layer 0 uses localist coding for characters (one neuron per character).
Layers 4-5 use distributed coding for words (pattern per word).

The transition from localist to distributed as we go from characters
to words is a natural consequence of the combinatorial explosion:
39 characters fit in 512 neurons with room to spare, but thousands
of words require distributed representations.
