#!/usr/bin/env bash
set -euo pipefail

readonly NAMESPACE="20260722_case78_camera_recovery_v1_exclusive"
readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly CONTRACT="$ROOT/scripts/two_wheel_balance/case78_camera_recovery_runtime_authorization_v1.json"
readonly VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_case78_camera_recovery_runtime_authorization.py"
readonly PLAYBACK="${WIN_ROOT}\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
readonly SUMMARIZER="${WIN_ROOT}\\scripts\\two_wheel_balance\\summarize_case78_camera_recovery_canary.py"
readonly PLAN_DIR="${WIN_ROOT}\\artifacts\\two_wheel_riser\\20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
readonly GAINS="${WIN_ROOT}\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
readonly OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
readonly OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${NAMESPACE}"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

reject() {
  printf '{"reason":"%s","runtime_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON RISER_CASE78_NAMESPACE \
  RISER_CASE78_CONTRACT RISER_CASE78_PLAN RISER_CASE78_OUTPUT \
  RISER_CASE78_TIMEOUT RISER_CASE78_HEARTBEAT; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] \
  || reject "unsupported_mode" 2

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef \
      | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' \
      || true
  )"
  compute_owners="$("$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader)"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $q=$PID
      Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $q -and ($_.Name -eq "kit.exe" -or
        $_.CommandLine -match "smoke_.*playback|train_riser_residual_bc")
      } | ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    assert_gpu_free 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION"' EXIT
validator_args=(
  --contract "$CONTRACT"
  --repo-root "$ROOT"
  --namespace "$NAMESPACE"
  --output "$ADMISSION"
)
AUTHORIZATION_FILE="${RISER_CASE78_DYNAMIC_AUTHORIZATION_FILE:-}"
if [[ "$MODE" == --execute ]]; then
  [[ -n "$AUTHORIZATION_FILE" ]] || reject "authorization_file_missing" 4
  validator_args+=(--authorization-file "$AUTHORIZATION_FILE")
fi
python3 "$VALIDATOR" "${validator_args[@]}" >/dev/null
assert_gpu_free || reject "exclusive_gpu_ownership_failed" 5

if [[ "$MODE" == --preflight ]]; then
  cat "$ADMISSION"
  exit 0
fi

[[ ! -e "$OUTPUT" ]] || reject "namespace_not_fresh" 5
mkdir -p "$OUTPUT/gates" "$OUTPUT/logs"
cp "$ADMISSION" "$OUTPUT/admission.json"
rm -f "$AUTHORIZATION_FILE"

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
STARTED_EPOCH="$(date +%s.%N)"
PLAYBACK_STATUS=0
timeout --signal=TERM --kill-after=30s 5400 \
  "$PY" -u -X utf8 "$PLAYBACK" \
  --gains "$GAINS" --plan-dir "$PLAN_DIR" \
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
  --cases 78 --controller-wz-kp 1.05 --maximum-duration-scale 3.0 \
  --enable-camera-lever-arm-compensation \
  --camera-lever-arm-compensation-gain 1.0 \
  --maximum-camera-lever-arm-correction-m 0.05 \
  --enable-camera-error-recovery-governor \
  --camera-recovery-error-start-m 0.13 \
  --camera-recovery-error-full-m 0.155 \
  --minimum-camera-recovery-scale 0.20 \
  --runtime-heartbeat "$OUTPUT_WIN\\runtime_heartbeat.json" \
  --runtime-heartbeat-interval-steps 2000 \
  --output "$OUTPUT_WIN\\gates\\case_0078.json" --headless \
  >"$OUTPUT/logs/playback.log" 2>&1 || PLAYBACK_STATUS=$?
FINISHED_EPOCH="$(date +%s.%N)"
printf '%s\n' "$PLAYBACK_STATUS" >"$OUTPUT/logs/playback.exit_code"

GPU_RELEASE_PASSED=1
wait_for_gpu_release || GPU_RELEASE_PASSED=0
SUMMARY_STATUS=0
"$PY" -X utf8 "$SUMMARIZER" \
  --root "$OUTPUT_WIN" --runtime-commit "$HEAD" \
  --playback-exit-code "$PLAYBACK_STATUS" \
  --wall-started-epoch-s "$STARTED_EPOCH" \
  --wall-finished-epoch-s "$FINISHED_EPOCH" \
  --gpu-release-passed "$GPU_RELEASE_PASSED" \
  --output "$OUTPUT_WIN\\final_status.json" \
  >"$OUTPUT/logs/summary.log" 2>&1 || SUMMARY_STATUS=$?
printf '%s\n' "$SUMMARY_STATUS" >"$OUTPUT/logs/summary.exit_code"
exit "$SUMMARY_STATUS"

