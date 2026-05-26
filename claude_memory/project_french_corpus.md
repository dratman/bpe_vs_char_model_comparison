---
name: French literature corpus plan
description: Plan to add French literary texts from Gutenberg to future training corpus for bilingual training
type: project
---

Consider adding French literature from Project Gutenberg to a future training corpus.

**Why:** The current cleaned corpus (2026-04-15) contains ~1.2% incidental French. Ralph believes the two languages should not interfere and may help each other in the middle layers — French provides additional examples of abstract structure (agent/patient roles, narrative, temporal sequencing) with different surface forms, acting as a kind of data augmentation.

**How to apply:** For the next corpus rebuild, gather French literary texts from Gutenberg (Balzac, Hugo, Flaubert, Stendhal, Zola, Dumas, Maupassant, Sand, Voltaire, Rousseau, etc.). Open questions: what English/French ratio to use, and whether to shuffle paragraphs together or keep them in blocks (shuffled likely better for shared middle-layer representations).
