#!/bin/zsh
#
# backup_checkpoints.sh — Back up the in-progress training checkpoints
# to the external Expansion drive.
#
# *** Run from your own terminal on the Studio, NOT via Claude Code. ***
#
# Claude Code's sandbox lacks Full Disk Access and cannot reach
# /Volumes/Expansion. Your shell does. The .pt files do not care which
# process moves them.
#
# What this script does:
#   1. Verify /Volumes/Expansion is mounted.
#   2. Refresh the Studio's local copy of the M3's best-val BPE
#      checkpoint via rsync over SSH. Mtime-preserved skip means this
#      is a no-op when the M3 has not saved a new best since last
#      refresh. If the M3 is asleep, the rsync fails and the script
#      continues with whatever cached copy already exists on the Studio.
#   3. Rsync the best-val checkpoints (Studio char + M3 BPE) and their
#      tokenizer metadata to /Volumes/Expansion/bpe_vs_char_backups/.
#
# Idempotent: re-runs do nothing for files that have not changed.
# Safe to run repeatedly — e.g., once a day or after a long compute
# stretch.

set -u   # error on unset variables, but allow individual rsyncs to fail

BACKUP_DIR=/Volumes/Expansion/bpe_vs_char_backups
M3_HOST=RalphDratman@192.168.1.177
M3_REPO=0-Home-Working-on-M3-Pro/bpe_vs_char_model_comparison
CHAR_RUN=char_uppercase_16L_1280
BPE_RUN=bpe_uppercase_16L_1280_b2

# Step 1: verify backup destination is reachable
echo "==> Step 1/3: verifying /Volumes/Expansion is mounted"
if [[ ! -d /Volumes/Expansion ]]; then
    echo "ERROR: /Volumes/Expansion is not mounted."
    echo "       Plug in the external drive (or mount it) and re-run."
    exit 1
fi
mkdir -p "$BACKUP_DIR"
echo "    OK — backup target: ${BACKUP_DIR}"
echo

# Step 2: refresh the M3 BPE checkpoint on the Studio.
# Best-effort: if the M3 is asleep, fall through with cached copy.
echo "==> Step 2/3: refreshing the M3's best-val BPE checkpoint onto the Studio"
for FILE in "${BPE_RUN}.pt" "${BPE_RUN}_meta.pkl" "${BPE_RUN}_meta.json"; do
    if rsync -avh --partial -e "ssh -o ConnectTimeout=5 -o BatchMode=yes" \
        "${M3_HOST}:${M3_REPO}/pt/${FILE}" \
        "pt/" ; then
        : # ok
    else
        echo "    WARNING: rsync of ${FILE} from M3 failed."
        echo "             M3 may be asleep; will back up the existing local copy."
        break
    fi
done
echo

# Step 3: backup best-val checkpoints + meta to Expansion
echo "==> Step 3/3: backing up to ${BACKUP_DIR}"
FILES_TO_BACKUP=(
    "pt/${CHAR_RUN}.pt"
    "pt/${CHAR_RUN}_meta.pkl"
    "pt/${BPE_RUN}.pt"
    "pt/${BPE_RUN}_meta.pkl"
    "pt/${BPE_RUN}_meta.json"
)

# Verify each source file exists before attempting rsync; warn if any missing
for F in "${FILES_TO_BACKUP[@]}"; do
    if [[ ! -f "$F" ]]; then
        echo "    WARNING: source file ${F} does not exist; skipping"
    fi
done

# rsync all existing sources in one call (mtime-preserved skip avoids
# re-transferring unchanged files)
rsync -avh --progress \
    "${FILES_TO_BACKUP[@]}" \
    "$BACKUP_DIR/" 2>&1 | tail -20
echo

# Show the final inventory
echo "==> Backup complete. Contents of ${BACKUP_DIR}:"
ls -lh "$BACKUP_DIR/"
