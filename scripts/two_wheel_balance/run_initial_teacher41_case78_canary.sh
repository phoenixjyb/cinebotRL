#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
CASE=78
PADDED=0078
SOURCE_DURATION_S="135.487646"
EXECUTION_DURATION_S="192.29956737098348"
TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_v1"
NAMESPACE="20260722_initial_teacher41_masked_bc_case78_canary_v1_exclusive"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$NAMESPACE"
PLAN_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
TEACHER_ROOT="$ROOT/artifacts/two_wheel_riser/20260722_case78_shadow_label_measurement_v1_exclusive/gates"
TEACHER_GATE="$TEACHER_ROOT/case_${PADDED}.json"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/20260722_initial_teacher41_masked_bc_v1"
POLICY="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\20260722_initial_teacher41_masked_bc_v1\\residual_policy.torchscript.pt"
CPU_CONTRACT="$ROOT/artifacts/two_wheel_riser/20260722_initial_teacher41_masked_bc_case78_canary_contract_v1_cpu/contract.json"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
ROLLOUT_GATE="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
FINALIZER="$ROOT/scripts/two_wheel_balance/summarize_initial_teacher41_validation_canary.py"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"

REVIEWED_CPU_COMMIT="50975ea12e39b0e09a45da161d8cfd67169454de"
CPU_CONTRACT_SHA256="4eb446d8a205d79796c53aecc30ad4f20b6b00a90ed7f346c3b9f864bed7c334"
PLAN_SHA256="28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569"
TEACHER_GATE_SHA256="ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459"
POLICY_SHA256="0d796c600c6dca7dce176da555f4cd1f769163f41093d2b6313f4e6264888db7"
PLAYBACK_SHA256="ffe45cd5747f6e628caebafbc405d589f34df764df944ddc2b36a2efd0926b1d"
ROLLOUT_GATE_SHA256="e3e327f2b7bc7f3bdc5f7c27ba36edcf2f3660e5ed4c8b4e4324325764927f5b"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
AUTHORIZATION_SHA256="5084cbe95c7c13cb2153f1d5b67883f048bbde18f648ffce9f845c34293c84b4"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi

sha256() { sha256sum "$1" | awk '{print $1}'; }
identity_matches() { [[ -s "$1" && "$(sha256 "$1")" == "$2" ]]; }

gpu_owners() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true
  )"
  compute_owners="$(
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true
  )"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $ErrorActionPreference = "Stop"
      $queryProcessId = $PID
      Get-CimInstance Win32_Process |
        Where-Object {
          $_.ProcessId -ne $queryProcessId -and (
            $_.Name -eq "kit.exe" -or
            $_.CommandLine -match "smoke_.*playback|train_riser_residual_bc"
          )
        } |
        ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

wait_gpu_free() {
  local attempts=0
  while ! gpu_owners; do
    attempts=$((attempts + 1))
    (( attempts < 90 )) || return 1
    sleep 2
  done
}

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$HEAD" == "$UPSTREAM" ]] || { printf 'HEAD is not pushed\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is not clean\n' >&2
  exit 3
}
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_CPU_COMMIT" "$HEAD"
identity_matches "$CPU_CONTRACT" "$CPU_CONTRACT_SHA256"
identity_matches "$PLAN" "$PLAN_SHA256"
identity_matches "$TEACHER_GATE" "$TEACHER_GATE_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$ROLLOUT_GATE" "$ROLLOUT_GATE_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"

python3 - "$CPU_CONTRACT" "$HEAD" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
head = sys.argv[2]
source_commit = contract.get("source_commit", "")
checks = {
    "cpu_contract_ready": contract.get("cpu_contract_ready") is True,
    "case78_validation": contract.get("case") == 78
    and contract.get("split") == "validation",
    "camera_cap": contract.get("controller_contract", {}).get(
        "maximum_camera_lever_arm_correction_m"
    )
    == 0.1,
    "fresh_pair": contract.get("comparison_contract", {}).get("fresh_zero_required")
    is True
    and contract.get("comparison_contract", {}).get("fresh_learned_required") is True,
    "runtime_closed": contract.get("runtime_authorized") is False
    and contract.get("gpu_launch_authorized") is False
    and contract.get("dynamic_canary_authorized") is False,
    "learning_closed": contract.get("dataset_creation_authorized") is False
    and contract.get("bc_authorized") is False
    and contract.get("ppo_authorized") is False
    and contract.get("holdout_opened") is False,
    "runtime_descends_cpu_contract": bool(source_commit)
    and subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head], check=False
    ).returncode == 0,
}
if not all(checks.values()):
    raise SystemExit(f"case-78 CPU contract failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_case78_runtime_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 78,
    "split": "validation",
    "rollout_order": ["learned", "zero_if_learned_dynamic_passes"],
    "maximum_combined_timeout_s": 10800,
    "dataset_creation_authorized": False,
    "remaining_validation_cases_authorized": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "runtime_started": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_TEACHER41_CASE78_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT" >&2; exit 5; }
wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/learned" "$OUTPUT/zero" "$OUTPUT/logs"
python3 - "$OUTPUT/admission.json" "$HEAD" "$CPU_CONTRACT" "$PLAN" \
  "$TEACHER_GATE" "$POLICY" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(value):
    path = Path(value)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_case78_runtime_admission_v1",
    "runtime_commit": sys.argv[2],
    "cpu_contract": identity(sys.argv[3]),
    "plan": identity(sys.argv[4]),
    "teacher_gate": identity(sys.argv[5]),
    "policy": identity(sys.argv[6]),
    "authorization_sha256": sys.argv[7],
    "authorization_consumed_before_isaac": True,
    "case": 78,
    "split": "validation",
    "rollout_order": ["learned", "zero_if_learned_dynamic_passes"],
    "residual_action_scales": [0.35, 0.4, 0.1],
    "camera_lever_arm_cap_m": 0.1,
    "dataset_creation_authorized": False,
    "remaining_validation_cases_authorized": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

COMMON_ARGS=(
  --gains "$GAINS_WIN"
  --plan-dir "$PLAN_WIN"
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz'
  --cases "$CASE"
  --controller-wz-kp 1.05
  --maximum-duration-scale 3.0
  --enable-camera-lever-arm-compensation
  --camera-lever-arm-compensation-gain 1.0
  --maximum-camera-lever-arm-correction-m 0.10
  --residual-action-scales 0.35,0.40,0.10
  --runtime-heartbeat-interval-steps 2000
  --headless
)

LEARNED_STATUS=0
timeout --signal=TERM --kill-after=30s 5400 \
  "$PY" -u -X utf8 "$PLAYBACK_WIN" "${COMMON_ARGS[@]}" \
  --residual-policy "$POLICY_WIN" --residual-policy-device cuda \
  --runtime-heartbeat "$OUTPUT_WIN\\learned\\runtime_heartbeat.json" \
  --output "$OUTPUT_WIN\\learned\\case_${PADDED}.json" \
  >"$OUTPUT/logs/learned.log" 2>&1 || LEARNED_STATUS=$?
printf '%s\n' "$LEARNED_STATUS" >"$OUTPUT/logs/learned.exit_code"
wait_gpu_free || true

LEARNED_DYNAMIC_PASSED=0
if [[ -s "$OUTPUT/learned/case_${PADDED}.json" ]]; then
  LEARNED_DYNAMIC_PASSED="$(python3 - "$OUTPUT/learned/case_${PADDED}.json" <<'PY'
import json
from pathlib import Path
import sys
d=json.loads(Path(sys.argv[1]).read_text())
results=d.get('results', [])
r=results[0] if isinstance(results, list) and len(results) == 1 else {}
print(int(d.get('passed') is True and d.get('dynamic_quality_passed') is True and r.get('passed') is True))
PY
)"
fi

ZERO_STATUS=125
if [[ "$LEARNED_STATUS" == 0 && "$LEARNED_DYNAMIC_PASSED" == 1 ]] && wait_gpu_free; then
  ZERO_STATUS=0
  timeout --signal=TERM --kill-after=30s 5400 \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" "${COMMON_ARGS[@]}" \
    --zero-policy-action \
    --runtime-heartbeat "$OUTPUT_WIN\\zero\\runtime_heartbeat.json" \
    --output "$OUTPUT_WIN\\zero\\case_${PADDED}.json" \
    >"$OUTPUT/logs/zero.log" 2>&1 || ZERO_STATUS=$?
fi
printf '%s\n' "$ZERO_STATUS" >"$OUTPUT/logs/zero.exit_code"
wait_gpu_free || true

GATE_STATUS=125
if [[ -s "$OUTPUT/learned/case_${PADDED}.json" && -s "$OUTPUT/zero/case_${PADDED}.json" ]]; then
  GATE_STATUS=0
  python3 "$ROLLOUT_GATE" --mode validation_canary \
    --teacher-dir "$TEACHER_ROOT" --zero-dir "$OUTPUT/zero" \
    --learned-dir "$OUTPUT/learned" --cases "$CASE" --policy "$POLICY" \
    --expected-tracking-profile "$TRACKING_PROFILE" \
    --maximum-regression-fraction 0.05 \
    --minimum-zero-improvement-fraction 0.05 \
    --output "$OUTPUT/summary.json" >"$OUTPUT/logs/gate.log" 2>&1 || GATE_STATUS=$?
fi
printf '%s\n' "$GATE_STATUS" >"$OUTPUT/logs/gate.exit_code"

FINAL_STATUS=0
python3 "$FINALIZER" --root "$OUTPUT" --case "$CASE" \
  --source-duration-s "$SOURCE_DURATION_S" \
  --execution-duration-s "$EXECUTION_DURATION_S" \
  --tracking-profile "$TRACKING_PROFILE" --runtime-commit "$HEAD" \
  --learned-exit-code "$LEARNED_STATUS" --zero-exit-code "$ZERO_STATUS" \
  --gate-exit-code "$GATE_STATUS" --output "$OUTPUT/final_status.json" \
  >"$OUTPUT/logs/finalize.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalize.exit_code"
exit "$FINAL_STATUS"
