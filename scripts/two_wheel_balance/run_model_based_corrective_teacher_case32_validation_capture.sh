#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly NAMESPACE="20260804_model_based_corrective_teacher_case32_validation_capture_v1_coexistence"
readonly CONTRACT="$SCRIPT_DIR/model_based_corrective_teacher_case32_validation_capture_contract_v1.json"
readonly VALIDATOR="$SCRIPT_DIR/validate_model_based_corrective_teacher_case32_validation_capture.py"
readonly RESOURCE_GUARD="$SCRIPT_DIR/check_windows_shared_resource_admission.py"
readonly RESOURCE_MONITOR="$SCRIPT_DIR/monitor_windows_shared_resource_pressure.py"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly WIN_ROOT="G:\\wSpace\\cinebotRL-two-wheel-riser"
readonly OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
readonly OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$NAMESPACE"
readonly PLAN_DIR="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260728_case16_validation_disposition_cpu_v1\\source"
readonly CORRECTIVE_PROFILE="$WIN_ROOT\\scripts\\two_wheel_balance\\model_based_corrective_teacher_case32_validation_natural_error_profile_v1.json"
readonly GAINS="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
readonly PLAYBACK="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
readonly FINALIZER="$WIN_ROOT\\scripts\\two_wheel_balance\\summarize_model_based_corrective_teacher_case32_validation_capture.py"

reject() {
  printf '{"reason":"%s","runtime_started":false,"label_capture_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON \
  RISER_CORRECTIVE_CASE32_VALIDATION_CAPTURE_NAMESPACE \
  RISER_CORRECTIVE_CASE32_VALIDATION_CAPTURE_CONTRACT \
  RISER_CORRECTIVE_CASE32_VALIDATION_CAPTURE_OUTPUT; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || {
  reject "unsupported_mode" 2
}
AUTHORIZATION_FILE="${RISER_CORRECTIVE_CASE32_VALIDATION_CAPTURE_AUTHORIZATION_FILE:-}"
AUTHORIZATION_SHA256="${RISER_CORRECTIVE_CASE32_VALIDATION_CAPTURE_AUTHORIZATION_SHA256:-}"
if [[ "$MODE" == --execute ]]; then
  [[ -n "$AUTHORIZATION_SHA256" ]] || {
    reject "runtime_authorization_not_issued" 4
  }
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
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name \
      --format=csv,noheader 2>/dev/null || true
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
RESOURCE_ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION" "$RESOURCE_ADMISSION"' EXIT
VALIDATOR_ARGS=(
  --contract "$CONTRACT"
  --repo-root "$ROOT"
  --namespace "$NAMESPACE"
  --output "$ADMISSION"
)
if [[ "$MODE" == --execute ]]; then
  VALIDATOR_ARGS+=(
    --authorization-file "$AUTHORIZATION_FILE"
    --authorization-sha256 "$AUTHORIZATION_SHA256"
  )
fi
python3 "$VALIDATOR" "${VALIDATOR_ARGS[@]}" >/dev/null
if [[ "$MODE" == --preflight ]]; then
  cat "$ADMISSION"
  exit 0
fi

assert_gpu_free || reject "exclusive_gpu_ownership_failed" 5
if ! python3 "$RESOURCE_GUARD" --phase launch \
  --output "$RESOURCE_ADMISSION" >/dev/null; then
  cat "$RESOURCE_ADMISSION" >&2
  reject "shared_windows_resource_admission_failed" 5
fi
[[ ! -L "$AUTHORIZATION_FILE" ]] || {
  reject "authorization_file_is_symlink" 4
}
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
cp "$RESOURCE_ADMISSION" "$OUTPUT/resource_admission.json"
rm -f "$AUTHORIZATION_FILE"

PLAYBACK_STATUS=0
setsid timeout --signal=TERM --kill-after=30s 600 \
  "$PY" -u -X utf8 "$PLAYBACK" \
  --gains "$GAINS" \
  --plan-dir "$PLAN_DIR" \
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
  --cases 32 \
  --controller-wz-kp 1.05 \
  --maximum-duration-scale 3.0 \
  --maximum-pitch-deg 12.0 \
  --maximum-position-p95-m 0.15 \
  --maximum-position-error-m 0.25 \
  --maximum-attitude-p95-deg 5.0 \
  --maximum-attitude-error-deg 10.0 \
  --maximum-riser-servo-error-m 0.03 \
  --maximum-proxy-servo-error-deg 5.0 \
  --maximum-internal-proxy-rate-deg-s 360.0 \
  --maximum-saturation-ratio 0.20 \
  --enable-camera-lever-arm-compensation \
  --camera-lever-arm-compensation-gain 1.0 \
  --maximum-camera-lever-arm-correction-m 0.05 \
  --residual-action-scales 0.05,0.05,0.02 \
  --policy-command-base model_based_planner \
  --zero-policy-action \
  --corrective-teacher-profile "$CORRECTIVE_PROFILE" \
  --corrective-teacher-capture-dir "$OUTPUT_WIN\\capture" \
  --corrective-teacher-capture-admission "$OUTPUT_WIN\\admission.json" \
  --corrective-teacher-capture-split validation \
  --runtime-heartbeat "$OUTPUT_WIN\\runtime_heartbeat.json" \
  --output "$OUTPUT_WIN\\case_0032.json" \
  --headless \
  >"$OUTPUT/logs/playback.log" 2>&1 &
PLAYBACK_PID=$!

MONITOR_STATUS=0
python3 "$RESOURCE_MONITOR" \
  --pid "$PLAYBACK_PID" \
  --output "$OUTPUT/resource_monitor.json" \
  --interval-s 5 \
  >"$OUTPUT/logs/resource_monitor.log" 2>&1 || MONITOR_STATUS=$?
wait "$PLAYBACK_PID" || PLAYBACK_STATUS=$?
if (( MONITOR_STATUS != 0 && PLAYBACK_STATUS == 0 )); then
  PLAYBACK_STATUS=75
fi
printf '%s\n' "$PLAYBACK_STATUS" >"$OUTPUT/logs/playback.exit_code"
printf '%s\n' "$MONITOR_STATUS" >"$OUTPUT/logs/resource_monitor.exit_code"

GPU_RELEASE_PASSED=1
wait_gpu_free || GPU_RELEASE_PASSED=0
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
FINAL_STATUS=0
"$PY" -X utf8 "$FINALIZER" \
  --root "$OUTPUT_WIN" \
  --admission "$OUTPUT_WIN\\admission.json" \
  --runtime-commit "$HEAD" \
  --playback-exit-code "$PLAYBACK_STATUS" \
  --gpu-release-passed "$GPU_RELEASE_PASSED" \
  --output "$OUTPUT_WIN\\final_status.json" \
  >"$OUTPUT/logs/finalizer.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalizer.exit_code"
exit "$FINAL_STATUS"
