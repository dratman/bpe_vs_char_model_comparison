#!/bin/bash
#
# stop_run_via_override.sh — GRACEFUL, RESUMABLE stop of a training run.
#
# Writes {"max_iters": N} into a run's --accept_overrides JSON. train.py polls that
# file (every 1000 iters) and ends cleanly at max_iters, saving its _final checkpoint
# so the run is fully resumable later. This NEVER kills a process — it is the
# deny-list-safe way to stop/extend a run (the deny-list blocks kill/pkill/killall).
#
# Standing policy (Ralph, 2026-06-19): graceful override-file stops like this may run
# without asking; the destructive deny-list (kill/pkill/rm -rf/force-push) stays.
#
# Usage: sh/stop_run_via_override.sh <path/to/overrides.json> <max_iters>
set -u
OJSON="${1:?usage: $0 <overrides.json> <max_iters>}"
MAXIT="${2:?usage: $0 <overrides.json> <max_iters>}"
printf '{"max_iters": %d}\n' "$MAXIT" > "$OJSON"
echo "wrote $OJSON"
cat "$OJSON"
echo "Run will stop gracefully at iter $MAXIT (next override check), save _final, free the GPU."
