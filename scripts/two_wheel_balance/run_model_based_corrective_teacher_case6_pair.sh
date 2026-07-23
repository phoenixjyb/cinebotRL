#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly CONTRACT="$SCRIPT_DIR/model_based_corrective_teacher_case6_pair_contract_v1.json"
readonly VALIDATOR="$SCRIPT_DIR/validate_model_based_corrective_teacher_case6_pair.py"
readonly NAMESPACE="20260724_model_based_corrective_teacher_case6_pair_v1_exclusive"

reject() {
  printf '{"reason":"%s","python_started":false,"isaac_started":false,"runtime_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON \
  RISER_CORRECTIVE_CASE6_NAMESPACE RISER_CORRECTIVE_CASE6_CONTRACT \
  RISER_CORRECTIVE_CASE6_PROFILE RISER_CORRECTIVE_CASE6_PLAN \
  RISER_CORRECTIVE_CASE6_PERTURBATION RISER_CORRECTIVE_CASE6_OUTPUT \
  RISER_CORRECTIVE_CASE6_AUTHORIZATION_FILE; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || reject "unsupported_mode" 2

# No runtime authorization exists in this committed CPU-only contract.
if [[ "$MODE" == --execute ]]; then
  reject "runtime_authorization_not_issued" 4
fi

OUTPUT="$(mktemp)"
trap 'rm -f "$OUTPUT"' EXIT
python3 "$VALIDATOR" \
  --contract "$CONTRACT" \
  --repo-root "$ROOT" \
  --namespace "$NAMESPACE" \
  --output "$OUTPUT" >/dev/null
cat "$OUTPUT"
