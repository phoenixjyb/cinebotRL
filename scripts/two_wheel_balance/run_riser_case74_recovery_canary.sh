#!/usr/bin/env bash
set -euo pipefail

AUTHORIZATION="${RISER_CASE74_GPU_AUTHORIZATION:-}"
EXPECTED_AUTHORIZATION="AUTHORIZED_CASE74_RECOVERY_V4"
if [[ "$AUTHORIZATION" != "$EXPECTED_AUTHORIZATION" ]]; then
  printf 'case-74 GPU authorization is absent; expected RISER_CASE74_GPU_AUTHORIZATION=%s\n' \
    "$EXPECTED_AUTHORIZATION" >&2
  exit 7
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RISER_GATE_C_CASES="74"
export RISER_GATE_C_STAMP="${RISER_GATE_C_STAMP:-20260717_gate_c_case74_recovery_direction_v4_exclusive}"

exec bash "$SCRIPT_DIR/run_riser_gate_c_canary.sh"
