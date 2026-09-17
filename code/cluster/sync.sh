#!/bin/bash
# Copy experiment results between the local machine and the cluster. Run ON the local machine.
#
#   cluster/sync.sh pull [experiment]   cluster -> local
#   cluster/sync.sh push [experiment]   local -> cluster (to resume started runs)
#
set -euo pipefail

HOST=${HPC_HOST:-hpc}
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/experiment_results"
REMOTE="$(ssh "$HOST" 'echo $WORK')/low-dimensional-control-from-lqr-to-rl/code/experiment_results"

case "$1" in
    pull) rsync -av --exclude='*.tmp' "$HOST:$REMOTE/${2:-}" "$LOCAL/" ;;
    push) ssh "$HOST" "mkdir -p '$REMOTE'"
          rsync -av --exclude='*.tmp' "$LOCAL/${2:-}" "$HOST:$REMOTE/" ;;
    *)    echo "usage: $0 pull|push [experiment]" >&2; exit 1 ;;
esac
