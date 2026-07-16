#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/scripts/two_wheel_balance/riser_remote_sync_manifest_20260716.txt"
REMOTE="${RISER_REMOTE:-yanbo@192.168.100.98}"
REMOTE_PORT="${RISER_REMOTE_PORT:-2222}"
REMOTE_ROOT="${RISER_REMOTE_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"

while IFS= read -r relative; do
  [[ -z "$relative" || "$relative" == \#* ]] && continue
  [[ -f "$ROOT/$relative" ]] || {
    printf 'missing manifest file: %s\n' "$relative" >&2
    exit 2
  }
done < "$MANIFEST"

ssh -o ConnectTimeout=8 -p "$REMOTE_PORT" "$REMOTE" \
  "test -d '$REMOTE_ROOT' && test -e '$REMOTE_ROOT/.git'"
rsync -av --files-from="$MANIFEST" -e "ssh -p $REMOTE_PORT" \
  "$ROOT/" "$REMOTE:$REMOTE_ROOT/"

printf 'synced %s to %s:%s\n' "$MANIFEST" "$REMOTE" "$REMOTE_ROOT"
