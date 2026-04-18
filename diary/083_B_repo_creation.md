# Diary 083_B: Repository Creation and File Consolidation

Date: 2026-04-14

## The problem

Project files were scattered across multiple directories on this machine
and two others, with no version control. Diary entries existed in two
separate directories with partially overlapping numbering. No mechanism
to keep three computers synchronized.

Directories on this machine (M2 MacBook):
- `0-HomeFolder-WorkingCopies-M2-Macbook_a1/temp/` — CLAUDE.md, diary/, doc/
- `0-HomeFolder-WorkingCopies-M2-Macbook_a1/bpe_vs_char_model_comparison/` — py/, sh/, sft_data/, pt/, txt_local/, terminal_logs/
- Also an iCloud copy at `Library/Mobile Documents/.../bpe_vs_char_model_comparison/`

No overlapping files between temp/ and bpe_vs_char_model_comparison/,
so merging was clean.

## What we did

1. Created a fresh directory `small_transformer_research/`
2. Copied contents from both `temp/` and `bpe_vs_char_model_comparison/`
3. Created `.gitignore` excluding large files:
   - `pt/` (checkpoints, 8-10GB each)
   - `txt_local/` (corpus, ~8GB)
   - `terminal_logs/`
   - `plots/`
4. `git init`, initial commit (94 files, 246K lines)
5. Created public GitHub repo: https://github.com/dratman/small_transformer_research
6. Pushed to `main` branch

## What's in the repo

| Directory | Contents | In git? |
|-----------|----------|---------|
| diary/ | 82 research entries (001-076, 080-082) | yes |
| py/ | model.py, train.py, tokenizer.py, sample.py | yes |
| sh/ | train.sh, train_bpe_16L16H.sh | yes |
| doc/ | Chat transcripts, vq_transformer_spec | yes |
| sft_data/ | round1_vocab.jsonl | yes |
| claude_code_sessions/ | Raw terminal session logs | yes |
| CLAUDE.md | Project instructions for Claude Code | yes |
| pt/ | Model checkpoints | no (gitignored) |
| txt_local/ | Training corpus | no (gitignored) |
| terminal_logs/ | Training output | no (gitignored) |

## Next steps

- Clone repo on the other two computers
- Compare files on each machine against repo, resolve any differences
- Once all three machines are synchronized, archive the old scattered
  directories
- Use git for all future changes — no more divergent copies

## Authentication note

`gh auth setup-git` was needed to configure git to use the GitHub CLI's
credentials for push/pull. Without this, HTTPS pushes failed with
"Invalid username or token."
