#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly NAMESPACE="20260723_model_based_corrective_case23_conversion_v1_cpu"
readonly CONTRACT="$ROOT/scripts/two_wheel_balance/model_based_corrective_case23_conversion_execution_contract_v1.json"
readonly CONTRACT_WIN="$WIN_ROOT\scripts\two_wheel_balance\model_based_corrective_case23_conversion_execution_contract_v1.json"
readonly VALIDATOR_WIN="$WIN_ROOT\scripts\two_wheel_balance\validate_model_based_corrective_case23_conversion_execution.py"
readonly CONVERTER_WIN="$WIN_ROOT\scripts\two_wheel_balance\convert_model_based_corrective_capture.py"
readonly FINALIZER_WIN="$WIN_ROOT\scripts\two_wheel_balance\finalize_model_based_corrective_case23_conversion.py"
readonly SOURCE_CAPTURE_WIN="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260723_case23_corrective_capture_v4\capture\case_0023_corrective_teacher_capture_v2.npz"
readonly SOURCE_FINAL_WIN="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260723_case23_corrective_capture_v4\final_status.json"
readonly OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
readonly OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${NAMESPACE}"
readonly DATASET_WIN="${OUTPUT_WIN}\\case_0023_model_based_corrective_case_dataset_v1.npz"
readonly TEMP_ROOT="$ROOT/artifacts/two_wheel_riser"

reject() {
  printf '{"reason":"%s","conversion_started":false,"output_created":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON \
  RISER_CASE23_V4_CONVERSION_NAMESPACE \
  RISER_CASE23_V4_CONVERSION_CONTRACT \
  RISER_CASE23_V4_CONVERSION_OUTPUT; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || reject "unsupported_mode" 2
AUTHORIZATION_FILE="${RISER_CASE23_V4_CONVERSION_AUTHORIZATION_FILE:-}"
AUTHORIZATION_SHA256="${RISER_CASE23_V4_CONVERSION_AUTHORIZATION_SHA256:-}"
if [[ "$MODE" == --execute ]]; then
  [[ -n "$AUTHORIZATION_SHA256" ]] || reject "conversion_authorization_not_issued" 4
  [[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
    reject "authorization_file_missing" 4
  }
fi

mkdir -p "$TEMP_ROOT"
ADMISSION="$TEMP_ROOT/.case23_conversion_admission.$$.json"
ADMISSION_WIN="$WIN_ROOT\artifacts\two_wheel_riser\.case23_conversion_admission.$$.json"
trap 'rm -f "$ADMISSION"' EXIT
VALIDATOR_ARGS=(
  --contract "$CONTRACT_WIN"
  --repo-root "$WIN_ROOT"
  --namespace "$NAMESPACE"
  --output "$ADMISSION_WIN"
)
if [[ "$MODE" == --execute ]]; then
  AUTHORIZATION_WIN="$(wslpath -w "$AUTHORIZATION_FILE")"
  VALIDATOR_ARGS+=(
    --authorization-file "$AUTHORIZATION_WIN"
    --authorization-sha256 "$AUTHORIZATION_SHA256"
  )
fi
"$PY" -X utf8 "$VALIDATOR_WIN" "${VALIDATOR_ARGS[@]}" >/dev/null
if [[ "$MODE" == --preflight ]]; then
  cat "$ADMISSION"
  exit 0
fi

[[ ! -L "$AUTHORIZATION_FILE" ]] || reject "authorization_file_is_symlink" 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || {
  reject "authorization_file_mode_not_0600" 4
}
[[ "$(sha256sum "$AUTHORIZATION_FILE" | awk '{print $1}')" == "$AUTHORIZATION_SHA256" ]] || {
  reject "authorization_hash_mismatch" 4
}
[[ ! -e "$OUTPUT" ]] || reject "conversion_namespace_not_fresh" 5
mkdir -p "$OUTPUT/logs"
cp "$CONTRACT" "$OUTPUT/contract.json"
cp "$ADMISSION" "$OUTPUT/admission.json"
rm -f "$AUTHORIZATION_FILE"

CONVERTER_STATUS=0
"$PY" -X utf8 "$CONVERTER_WIN" \
  --capture "$SOURCE_CAPTURE_WIN" \
  --final-status "$SOURCE_FINAL_WIN" \
  --output "$DATASET_WIN" \
  --expected-case 23 \
  --expected-split train \
  --execute \
  >"$OUTPUT/conversion_result.json" \
  2>"$OUTPUT/logs/converter.log" || CONVERTER_STATUS=$?
printf '%s\n' "$CONVERTER_STATUS" >"$OUTPUT/logs/converter.exit_code"

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
FINALIZER_STATUS=0
"$PY" -X utf8 "$FINALIZER_WIN" \
  --root "$OUTPUT_WIN" \
  --admission "$OUTPUT_WIN\admission.json" \
  --source-capture "$SOURCE_CAPTURE_WIN" \
  --conversion-result "$OUTPUT_WIN\conversion_result.json" \
  --runtime-commit "$HEAD" \
  --converter-exit-code "$CONVERTER_STATUS" \
  --output "$OUTPUT_WIN\final_status.json" \
  >"$OUTPUT/logs/finalizer.log" 2>&1 || FINALIZER_STATUS=$?
printf '%s\n' "$FINALIZER_STATUS" >"$OUTPUT/logs/finalizer.exit_code"
exit "$FINALIZER_STATUS"
