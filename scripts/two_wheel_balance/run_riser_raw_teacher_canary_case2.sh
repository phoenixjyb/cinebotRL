#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
AUTHORIZATION="AUTHORIZED_RISER_RAW_TEACHER_CASE2_CANARY_V2"
STAMP="20260720_initial_teacher_raw_canary_case2_v2_exclusive"
PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
PLANNER_COMMIT="0391190f536a29f65b4c97968b764f29444c9f43"
PLAN_SHA256="a2ad28cf4d353c59a9a642e39c8bbf484a0233df50a0b72b7ec18ca746c2cbe7"
SELECTION_SHA256="e0f1d2b44061aabfe64ad2ffa3d23f57bf9b3e51015b2e3fa0703ba24316bb06"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
TIMEOUT_SECONDS=480

if [[ "${RISER_RAW_TEACHER_CANARY_AUTHORIZATION:-}" != "$AUTHORIZATION" ]]; then
  printf 'raw-teacher case-2 canary authorization is absent or unknown\n' >&2
  exit 7
fi

for name in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON RISER_GATE_C_CASES \
  RISER_GATE_C_STAMP RISER_GATE_C_PORTFOLIO_STAMP RISER_DATASET_DIR \
  RISER_RESIDUAL_POLICY; do
  [[ -z "${!name:-}" ]] || {
    printf 'raw-teacher canary rejects environment override: %s\n' "$name" >&2
    exit 7
  }
done

PORTFOLIO="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"
SOURCE_MANIFEST="/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717/manifest.json"
SELECTION="$ROOT/artifacts/two_wheel_riser/20260720_initial_teacher42_selection_v1/selection.json"
SELECTION_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\20260720_initial_teacher42_selection_v1\\selection.json"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"
PLAN="$PORTFOLIO/case_0002_smoothed_riser_plan_v1.npz"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="${WIN_ROOT}\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="${WIN_ROOT}\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
AUDITOR="$ROOT/scripts/two_wheel_balance/audit_riser_raw_teacher_capture.py"
AUDITOR_WIN="${WIN_ROOT}\\scripts\\two_wheel_balance\\audit_riser_raw_teacher_capture.py"
RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_raw_teacher_canary_case2.sh"
DATASET_MODULE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_residual_dataset.py"
TRACKING="$ROOT/src/rl_platform/tasks/two_wheel_balance/whole_body_tracking.py"
RISER_CONTROL="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_control.py"

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(ps -ef | grep -E '[p]ython(\.exe)? .*smoke_.*playback\.py' || true)"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $ErrorActionPreference = "Stop"
      $queryProcessId = $PID
      Get-CimInstance Win32_Process |
        Where-Object {
          $_.ProcessId -ne $queryProcessId -and (
            $_.Name -eq "kit.exe" -or
            $_.CommandLine -match "smoke_.*playback|evaluate_cascade_robustness"
          )
        } |
        ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  if [[ -n "$wsl_owners" || -n "$compute_owners" || -n "$windows_owners" ]]; then
    printf 'raw-teacher canary GPU is not exclusive\n' >&2
    [[ -z "$wsl_owners" ]] || printf '%s\n' "$wsl_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    [[ -z "$windows_owners" ]] || printf '%s\n' "$windows_owners" >&2
    return 1
  fi
}

assert_no_competing_cpu() {
  ! ps -ef | grep -qE '[r]etarget_exact_source_v1_nonholonomic\.py' || {
    printf 'raw-teacher canary CPU/disk ownership is not exclusive\n' >&2
    return 1
  }
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    assert_gpu_free 2>/dev/null && return 0
    sleep 1
  done
  printf 'raw-teacher canary GPU did not release within 90 seconds\n' >&2
  return 1
}

[[ -x "$PY" && -x "$NVIDIA_SMI" && -x "$POWERSHELL" ]] || exit 2
[[ ! -e "$OUTPUT" ]] || {
  printf 'refusing existing raw-teacher namespace: %s\n' "$OUTPUT" >&2
  exit 2
}
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$SELECTION" | awk '{print $1}')" == "$SELECTION_SHA256" ]] || exit 2
[[ "$(sha256sum "$GAINS" | awk '{print $1}')" == "$GAINS_SHA256" ]] || exit 2
[[ "$(sha256sum "$ROBOT_USD" | awk '{print $1}')" == "$ROBOT_USD_SHA256" ]] || exit 2

git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make raw-teacher provenance ambiguous\n' >&2
  exit 2
}
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$COMMIT" == "$UPSTREAM" ]] || {
  printf 'raw-teacher runtime commit is not pushed\n' >&2
  exit 2
}
assert_gpu_free && assert_no_competing_cpu || exit 5

TEMP_ADMISSION="$(mktemp)"
trap 'rm -f "$TEMP_ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --manifest "$PORTFOLIO/manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_SHA256" \
  --expected-planner-commit "$PLANNER_COMMIT" \
  --expected-count 79 --minimum-candidates 70 --cases 2 \
  --output "$TEMP_ADMISSION" >/dev/null

python3 - "$TEMP_ADMISSION" "$COMMIT" "$STAMP" "$SELECTION_SHA256" \
  "$MANIFEST_SHA256" "$SOURCE_SHA256" "$PLAN_SHA256" \
  source_manifest "$SOURCE_MANIFEST" portfolio_manifest "$PORTFOLIO/manifest.json" \
  selected_plan "$PLAN" selection "$SELECTION" lqr_gains "$GAINS" \
  robot_usd "$ROBOT_USD" playback "$PLAYBACK" auditor "$AUDITOR" \
  raw_dataset_contract "$DATASET_MODULE" tracking_controller "$TRACKING" \
  riser_control "$RISER_CONTROL" wrapper "$RUNNER" validator "$VALIDATOR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update(
    {
        "runtime_commit": sys.argv[2],
        "upstream_commit": sys.argv[2],
        "namespace": sys.argv[3],
        "requested_cases": [2],
        "selection_sha256": sys.argv[4],
        "portfolio_manifest_sha256": sys.argv[5],
        "source_manifest_sha256": sys.argv[6],
        "plan_sha256": sys.argv[7],
        "trajectory_command_source": "deterministic_teacher",
        "raw_teacher_capture_authorized": True,
        "normalized_dataset_capture_authorized": False,
        "residual_action_application_authorized": False,
        "scale_freeze_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "runtime_authorized": payload.get("passed") is True,
    }
)
args = sys.argv[8:]
payload["runtime_identities"] = {
    args[index]: {
        "path": str(Path(args[index + 1]).resolve()),
        "sha256": hashlib.sha256(Path(args[index + 1]).read_bytes()).hexdigest(),
    }
    for index in range(0, len(args), 2)
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

mkdir -p "$OUTPUT/gates" "$OUTPUT/logs" "$OUTPUT/raw_cases"
mv "$TEMP_ADMISSION" "$OUTPUT/admission.json"
STATUS=0
timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" \
  "$PY" -u -X utf8 "$PLAYBACK_WIN" \
  --gains "$GAINS_WIN" --plan-dir "$PORTFOLIO_WIN" \
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
  --cases 2 --controller-wz-kp 1.05 --maximum-duration-scale 3.0 \
  --enable-camera-lever-arm-compensation \
  --camera-lever-arm-compensation-gain 1.0 \
  --maximum-camera-lever-arm-correction-m 0.05 \
  --raw-teacher-dir "${OUTPUT_WIN}\\raw_cases" \
  --output "${OUTPUT_WIN}\\gates\\case_0002.json" --headless \
  >"$OUTPUT/logs/case_0002.log" 2>&1 || STATUS=$?
printf '%s\n' "$STATUS" >"$OUTPUT/logs/case_0002.exit_code"
wait_for_gpu_release || STATUS=5

AUDIT_STATUS=6
RAW_CASE="$OUTPUT/raw_cases/case_0002_executed_raw_teacher_v1.npz"
if [[ "$STATUS" == 0 && -s "$OUTPUT/gates/case_0002.json" && -s "$RAW_CASE" ]]; then
  AUDIT_STATUS=0
  "$PY" -X utf8 "$AUDITOR_WIN" \
    --gate "${OUTPUT_WIN}\\gates\\case_0002.json" \
    --admission "${OUTPUT_WIN}\\admission.json" \
    --raw-case "${OUTPUT_WIN}\\raw_cases\\case_0002_executed_raw_teacher_v1.npz" \
    --selection "$SELECTION_WIN" --case 2 \
    --output "${OUTPUT_WIN}\\raw_capture_audit.json" \
    >"$OUTPUT/logs/raw_capture_audit.log" 2>&1 || AUDIT_STATUS=$?
fi

python3 - "$OUTPUT" "$COMMIT" "$STATUS" "$AUDIT_STATUS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
commit = sys.argv[2]
playback_status = int(sys.argv[3])
audit_status = int(sys.argv[4])

def identity(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } if path.is_file() else None

audit_path = root / "raw_capture_audit.json"
audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
passed = (
    playback_status == 0
    and audit_status == 0
    and audit.get("capture_admission_passed") is True
)
output = {
    "schema": "cinebotrl_two_wheel_riser_raw_teacher_canary_final_v1",
    "namespace": root.name,
    "runtime_commit": commit,
    "case": 2,
    "playback_exit_code": playback_status,
    "audit_exit_code": audit_status,
    "admission": identity(root / "admission.json"),
    "gate": identity(root / "gates/case_0002.json"),
    "raw_case": identity(root / "raw_cases/case_0002_executed_raw_teacher_v1.npz"),
    "raw_capture_audit": identity(audit_path),
    "dynamic_quality_passed": audit.get("checks", {}).get(
        "dynamic_quality_passed", False
    ),
    "raw_capture_admission_passed": audit.get("capture_admission_passed", False),
    "raw_residual_applied_to_commands": False,
    "action_scale_frozen": False,
    "valid_for_training": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "training_started": False,
    "passed": passed,
}
(root / "final_status.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
raise SystemExit(0 if passed else 4)
PY
