---
name: Corpus cleanup project for future training run
description: Plan to filter Gutenberg corpus by removing non-literary-prose content, for a future BPE training run
type: project
---

The current BPE training (started 2026-04-05) uses an unfiltered Gutenberg corpus that contains ~23% non-literary material: CIA World Factbooks, architecture manuals, religious scriptures, math constants, cookbooks, engineering texts, non-English works, encyclopedias, etc. This degrades sample quality.

**Why:** Samples from the current model produce references to "ionic columns," Koran passages, and garbled dashes from non-prose content in the corpus. Ralph wants a cleaner corpus for a future training run.

**How to apply:**
- The current training run continues as-is. Do NOT retrain or modify the current model.
- Individual source files are at: `/Users/RalphDratman/Library/Mobile Documents/com~apple~CloudDocs/0-HomeFolder-Working-iCloud_A/Gutenberg_Project_Books/gutenberg_texts/` (8,794 files)
- The title list is at: `/Volumes/Expansion/0_backups_Mac_Studio_Expansion/study_corpus_and_training_2c_bpe/txt_local/list_of_gutenberg_novels.txt`
- A filtering script should classify each file as keep/remove based on filename patterns
- Keep: novels, short stories, literary essays, memoirs, travel writing, biography, literary criticism, philosophy in literary English, slave narratives, children's fiction
- Remove: architecture manuals, religious scriptures, math constants, CIA factbooks, cookbooks, engineering texts, non-English works, encyclopedias/dictionaries, zoology taxonomy papers, poetry, grammars/primers, medical treatises, military manuals, periodicals (Mirror of Literature, Scientific American Supplement)
- When in doubt, keep the work
- Estimated removal: ~750 MB out of ~8.3 GB corpus
