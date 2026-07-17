#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZATION="AUTHORIZED_CASE74_RECOVERY_V4_RUNTIME_V2"
readonly OBSOLETE_AUTHORIZATION="AUTHORIZED_CASE74_RECOVERY_V4"
readonly PINNED_NAMESPACE="20260717_gate_c_case74_recovery_v4_runtime_v2_exclusive"
readonly PINNED_CONTRACT_RELATIVE="scripts/two_wheel_balance/case74_recovery_v4_runtime_contract_v2.json"
readonly PINNED_CASE_TIMEOUT_SECONDS="1200"

protected_variables=(
  RISER_ROOT
  RISER_WIN_ROOT
  ISAAC_PYTHON
  RISER_GATE_C_PORTFOLIO_STAMP
  RISER_GATE_C_MANIFEST_SHA256
  RISER_GATE_C_SOURCE_SHA256
  RISER_GATE_C_CASES
  RISER_GATE_C_STAMP
  RISER_GATE_C_CASE_TIMEOUT_SECONDS
  RISER_GAINS_WIN
  RISER_CASE74_CONTRACT
  RISER_CASE74_CONTRACT_ADMISSION
)
for variable in "${protected_variables[@]}"; do
  if [[ -n "${!variable+x}" ]]; then
    printf 'case-74 runtime contract rejects environment override: %s\n' "$variable" >&2
    exit 7
  fi
done

provided_authorization="${RISER_CASE74_GPU_AUTHORIZATION:-}"
if [[ "$provided_authorization" == "$OBSOLETE_AUTHORIZATION" ]]; then
  printf 'obsolete case-74 authorization is permanently rejected\n' >&2
  exit 7
fi
if [[ "$provided_authorization" != "$AUTHORIZATION" ]]; then
  printf 'case-74 runtime authorization is absent or unknown\n' >&2
  exit 7
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTRACT="$ROOT/$PINNED_CONTRACT_RELATIVE"
VALIDATOR="$SCRIPT_DIR/validate_riser_case74_recovery_contract.py"
ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION"' EXIT

python3 "$VALIDATOR" \
  --contract "$CONTRACT" \
  --repo-root "$ROOT" \
  --namespace "$PINNED_NAMESPACE" \
  --authorization "$provided_authorization" \
  --output "$ADMISSION" >/dev/null

export RISER_GATE_C_CASES="74"
export RISER_GATE_C_STAMP="$PINNED_NAMESPACE"
export RISER_GATE_C_CASE_TIMEOUT_SECONDS="$PINNED_CASE_TIMEOUT_SECONDS"
export RISER_CASE74_CONTRACT="$CONTRACT"
export RISER_CASE74_CONTRACT_ADMISSION="$ADMISSION"

bash "$SCRIPT_DIR/run_riser_gate_c_canary.sh"
