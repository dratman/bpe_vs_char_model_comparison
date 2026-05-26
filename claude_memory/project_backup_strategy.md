---
name: Backup strategy
description: Big binary .pt checkpoints go to the Expansion drive; ordinary-size files are covered by git pushes to the remote repo
type: project
originSessionId: b92ab29a-f226-4f76-9253-faef2b2fdb76
---
Backup is split by file size and type:

- **Ordinary-size files (source code, shell scripts, terminal logs, diary, docs)** — backed up via git push to `dratman/bpe_vs_char_model_comparison`. The repo's `.gitignore` excludes large binaries (`pt/`, `*_pt/`, `txt_local/`, `plots/`).
- **Large binary .pt checkpoints (multi-GB) and corpora** — backed up to the external Expansion drive under `/Volumes/Expansion/0_backups_Mac_Studio_Expansion/<project-name>/` (organizational convention). `sh/backup_checkpoints.sh` handles the best-val checkpoints for the current trainings.

**Why:** *Ralph confirmed this 2026-05-20 when asked whether Studio→Expansion backup was needed beyond the script's scope.* The script is intentionally narrow — it only covers the big files that can't go to git.

**How to apply:** Don't suggest backing up source/logs/etc. to Expansion; git already covers them. When proposing what to back up to Expansion, focus on:
  - Best-val checkpoints (already in `backup_checkpoints.sh`)
  - Intermediate checkpoints if needed for layer-stability analysis (currently on M3 / in `*_pt/` dirs, ~100+ GB each)
  - Large corpora (`txt_local/`)
  - Anything else that's gitignored due to size
