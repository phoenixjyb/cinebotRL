#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly NAMESPACE="20260723_model_based_corrective_teacher_case23_capture_v1_exclusive"
readonly CONTRACT="$ROOT/scripts/two_wheel_balance/model_based_corrective_teacher_case23_capture_contract_v1.json"
readonly VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_model_based_corrective_teacher_case23_capture.py"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly WIN_ROOT="G:\wSpace\cinebotRL-two-wheel-riser"
readonly OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
readonly OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${NAMESPACE}"
readonly PLAN_DIR="$WIN_ROOT\artifacts\two_wheel_riser\20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
readonly WRENCH_PROFILE="$WIN_ROOT\scripts\two_wheel_balance\model_based_corrective_teacher_case23_wrench_profile_v1.json"
readonly CORRECTIVE_PROFILE="$WIN_ROOT\scripts\two_wheel_balance\model_based_corrective_teacher_case23_profile_v1.json"
readonly GAINS="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json"
readonly PLAYBACK="$WIN_ROOT\scripts\two_wheel_balance\smoke_riser_reference_playback.py"
readonly FINALIZER="$WIN_ROOT\scripts\two_wheel_balance\summarize_model_based_corrective_teacher_case23_capture.py"
# The one-use authorization was consumed by the rejected v1 attempt.
readonly AUTHORIZATION_SHA256=""

reject() {
  printf '{"reason":"%s","runtime_started":false,"label_capture_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON \
  RISER_CORRECTIVE_CASE23_CAPTURE_NAMESPACE \
  RISER_CORRECTIVE_CASE23_CAPTURE_CONTRACT \
  RISER_CORRECTIVE_CASE23_CAPTURE_OUTPUT; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || reject "unsupported_mode" 2
AUTHORIZATION_FILE="${RISER_CORRECTIVE_CASE23_CAPTURE_AUTHORIZATION_FILE:-}"
if [[ "$MODE" == --execute ]]; then
  [[ -n "$AUTHORIZATION_SHA256" ]] || reject "runtime_authorization_not_issued" 4
  [[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
    reject "authorization_file_missing" 4
  }
fi

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true
  )"
  compute_owners="$(
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true
  )"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $queryProcessId = $PID
      Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $queryProcessId -and (
          $_.Name -eq "kit.exe" -or
          $_.CommandLine -match "smoke_.*playback|train_riser_residual_bc"
        )
      } | ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

wait_gpu_free() {
  local attempts=0
  while ! assert_gpu_free; do
    attempts=$((attempts + 1))
    (( attempts < 90 )) || return 1
    sleep 2
  done
}

ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION"' EXIT
VALIDATOR_ARGS=(
  --contract "$CONTRACT"
  --repo-root "$ROOT"
  --namespace "$NAMESPACE"
  --output "$ADMISSION"
)
if [[ "$MODE" == --execute ]]; then
  VALIDATOR_ARGS+=(--authorization-file "$AUTHORIZATION_FILE")
fi
python3 "$VALIDATOR" "${VALIDATOR_ARGS[@]}" >/dev/null
if [[ "$MODE" == --preflight ]]; then
  cat "$ADMISSION"
  exit 0
fi

assert_gpu_free || reject "exclusive_gpu_ownership_failed" 5
[[ ! -L "$AUTHORIZATION_FILE" ]] || reject "authorization_file_is_symlink" 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || {
  reject "authorization_file_mode_not_0600" 4
}
[[ "$(sha256sum "$AUTHORIZATION_FILE" | awk '{print $1}')" == "$AUTHORIZATION_SHA256" ]] || {
  reject "authorization_hash_mismatch" 4
}
[[ ! -e "$OUTPUT" ]] || reject "namespace_not_fresh" 5
mkdir -p "$OUTPUT/capture" "$OUTPUT/logs"
cp "$CONTRACT" "$OUTPUT/contract.json"
cp "$ADMISSION" "$OUTPUT/admission.json"
rm -f "$AUTHORIZATION_FILE"

PLAYBACK_STATUS=0
timeout --signal=TERM --kill-after=30s 600 \
  "$PY" -u -X utf8 "$PLAYBACK" \
  --gains "$GAINS" \
  --plan-dir "$PLAN_DIR" \
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
  --cases 23 \
  --controller-wz-kp 1.05 \
  --maximum-duration-scale 3.0 \
  --enable-camera-lever-arm-compensation \
  --camera-lever-arm-compensation-gain 1.0 \
  --maximum-camera-lever-arm-correction-m 0.05 \
  --residual-action-scales 0.05,0.05,0.02 \
  --policy-command-base model_based_planner \
  --zero-policy-action \
  --corrective-teacher-profile "$CORRECTIVE_PROFILE" \
  --corrective-teacher-capture-dir "$OUTPUT_WIN\capture" \
  --corrective-teacher-capture-admission "$OUTPUT_WIN\admission.json" \
  --deterministic-wrench-profile "$WRENCH_PROFILE" \
  --runtime-heartbeat "$OUTPUT_WIN\runtime_heartbeat.json" \
  --output "$OUTPUT_WIN\case_0023.json" \
  --headless \
  >"$OUTPUT/logs/playback.log" 2>&1 || PLAYBACK_STATUS=$?
printf '%s\n' "$PLAYBACK_STATUS" >"$OUTPUT/logs/playback.exit_code"

GPU_RELEASE_PASSED=1
wait_gpu_free || GPU_RELEASE_PASSED=0
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
FINAL_STATUS=0
"$PY" -X utf8 "$FINALIZER" \
  --root "$OUTPUT_WIN" \
  --admission "$OUTPUT_WIN\admission.json" \
  --runtime-commit "$HEAD" \
  --playback-exit-code "$PLAYBACK_STATUS" \
  --gpu-release-passed "$GPU_RELEASE_PASSED" \
  --output "$OUTPUT_WIN\final_status.json" \
  >"$OUTPUT/logs/finalizer.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalizer.exit_code"
exit "$FINAL_STATUS"
