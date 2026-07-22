#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly NAMESPACE="20260722_model_based_corrective_teacher_case30_capture_v1_exclusive"
readonly CONTRACT="$ROOT/scripts/two_wheel_balance/model_based_corrective_teacher_case30_capture_contract_v1.json"
readonly VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_model_based_corrective_teacher_case30_capture.py"

reject() {
  printf '{"reason":"%s","runtime_started":false,"label_capture_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_CORRECTIVE_CAPTURE_NAMESPACE \
  RISER_CORRECTIVE_CAPTURE_CONTRACT RISER_CORRECTIVE_CAPTURE_OUTPUT \
  RISER_CORRECTIVE_CAPTURE_AUTHORIZATION_FILE; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || reject "unsupported_mode" 2
[[ "$MODE" == --preflight ]] || reject "runtime_authorization_not_issued" 4

ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --contract "$CONTRACT" \
  --repo-root "$ROOT" \
  --namespace "$NAMESPACE" \
  --output "$ADMISSION" >/dev/null
cat "$ADMISSION"
