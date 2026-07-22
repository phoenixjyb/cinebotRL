#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly NAMESPACE="20260723_model_based_corrective_teacher_case23_pair_v1_exclusive"
readonly CONTRACT_RELATIVE="model_based_corrective_teacher_case23_pair_contract_v1.json"
readonly VALIDATOR_RELATIVE="validate_model_based_corrective_teacher_case23_pair.py"
readonly CONTRACT="$ROOT/scripts/two_wheel_balance/$CONTRACT_RELATIVE"
readonly VALIDATOR="$ROOT/scripts/two_wheel_balance/$VALIDATOR_RELATIVE"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly WIN_ROOT="G:\\wSpace\\cinebotRL-two-wheel-riser"
readonly OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
readonly OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$NAMESPACE"
readonly PLAN_DIR="$WIN_ROOT\\artifacts\\two_wheel_riser\\20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
readonly WRENCH_PROFILE="$WIN_ROOT\\scripts\\two_wheel_balance\\model_based_corrective_teacher_case23_wrench_profile_v1.json"
readonly CORRECTIVE_PROFILE="$WIN_ROOT\\scripts\\two_wheel_balance\\model_based_corrective_teacher_case23_profile_v1.json"
readonly GAINS="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
readonly PLAYBACK="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
readonly FINALIZER="$WIN_ROOT\\scripts\\two_wheel_balance\\summarize_model_based_corrective_teacher_case23_pair.py"
# Runtime authorization is intentionally absent from this CPU-only change.
readonly AUTHORIZATION_SHA256=""

reject() {
  printf '{"reason":"%s","runtime_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON \
  RISER_CORRECTIVE_CASE23_NAMESPACE RISER_CORRECTIVE_CASE23_CONTRACT \
  RISER_CORRECTIVE_CASE23_PROFILE RISER_CORRECTIVE_CASE23_PLAN \
  RISER_CORRECTIVE_CASE23_PERTURBATION RISER_CORRECTIVE_CASE23_OUTPUT; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || reject "unsupported_mode" 2
AUTHORIZATION_FILE="${RISER_CORRECTIVE_CASE23_AUTHORIZATION_FILE:-}"
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
python3 "$VALIDATOR" \
  --contract "$CONTRACT" \
  --repo-root "$ROOT" \
  --namespace "$NAMESPACE" \
  --output "$ADMISSION" >/dev/null
assert_gpu_free || reject "exclusive_gpu_ownership_failed" 5
if [[ "$MODE" == --preflight ]]; then
  cat "$ADMISSION"
  exit 0
fi

# This route remains unreachable until a later reviewed commit pins a token hash.
[[ ! -L "$AUTHORIZATION_FILE" ]] || reject "authorization_file_is_symlink" 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || {
  reject "authorization_file_mode_not_0600" 4
}
[[ "$(sha256sum "$AUTHORIZATION_FILE" | awk '{print $1}')" == "$AUTHORIZATION_SHA256" ]] || {
  reject "authorization_hash_mismatch" 4
}
[[ ! -e "$OUTPUT" ]] || reject "namespace_not_fresh" 5
mkdir -p "$OUTPUT/baseline" "$OUTPUT/candidate" "$OUTPUT/logs"
cp "$CONTRACT" "$OUTPUT/contract.json"
cp "$ADMISSION" "$OUTPUT/admission.json"
rm -f "$AUTHORIZATION_FILE"

COMMON_ARGS=(
  --gains "$GAINS"
  --plan-dir "$PLAN_DIR"
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz'
  --cases 23
  --controller-wz-kp 1.05
  --maximum-duration-scale 3.0
  --enable-camera-lever-arm-compensation
  --camera-lever-arm-compensation-gain 1.0
  --maximum-camera-lever-arm-correction-m 0.05
  --residual-action-scales 0.05,0.05,0.02
  --policy-command-base model_based_planner
  --zero-policy-action
  --deterministic-wrench-profile "$WRENCH_PROFILE"
  --headless
)

BASELINE_STATUS=0
timeout --signal=TERM --kill-after=30s 600 \
  "$PY" -u -X utf8 "$PLAYBACK" "${COMMON_ARGS[@]}" \
  --runtime-heartbeat "$OUTPUT_WIN\\baseline\\runtime_heartbeat.json" \
  --output "$OUTPUT_WIN\\baseline\\case_0023.json" \
  >"$OUTPUT/logs/baseline.log" 2>&1 || BASELINE_STATUS=$?
printf '%s\n' "$BASELINE_STATUS" >"$OUTPUT/logs/baseline.exit_code"
BASELINE_GPU_RELEASE_PASSED=1
wait_gpu_free || BASELINE_GPU_RELEASE_PASSED=0

CANDIDATE_STATUS=125
if [[ "$BASELINE_STATUS" == 0 && "$BASELINE_GPU_RELEASE_PASSED" == 1 ]] \
  && python3 - "$OUTPUT/baseline/case_0023.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
result = payload.get("results", [{}])[0]
raise SystemExit(0 if payload.get("passed") is True and result.get("dynamic_quality_passed") is True else 1)
PY
then
  CANDIDATE_STATUS=0
  timeout --signal=TERM --kill-after=30s 600 \
    "$PY" -u -X utf8 "$PLAYBACK" "${COMMON_ARGS[@]}" \
    --corrective-teacher-profile "$CORRECTIVE_PROFILE" \
    --runtime-heartbeat "$OUTPUT_WIN\\candidate\\runtime_heartbeat.json" \
    --output "$OUTPUT_WIN\\candidate\\case_0023.json" \
    >"$OUTPUT/logs/candidate.log" 2>&1 || CANDIDATE_STATUS=$?
fi
printf '%s\n' "$CANDIDATE_STATUS" >"$OUTPUT/logs/candidate.exit_code"
GPU_RELEASE_PASSED=1
wait_gpu_free || GPU_RELEASE_PASSED=0
if [[ "$BASELINE_GPU_RELEASE_PASSED" == 0 ]]; then
  GPU_RELEASE_PASSED=0
fi

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
FINAL_STATUS=0
"$PY" -X utf8 "$FINALIZER" \
  --root "$OUTPUT_WIN" \
  --admission "$OUTPUT_WIN\\admission.json" \
  --runtime-commit "$HEAD" \
  --baseline-exit-code "$BASELINE_STATUS" \
  --candidate-exit-code "$CANDIDATE_STATUS" \
  --gpu-release-passed "$GPU_RELEASE_PASSED" \
  --output "$OUTPUT_WIN\\final_status.json" \
  >"$OUTPUT/logs/finalizer.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalizer.exit_code"
exit "$FINAL_STATUS"
