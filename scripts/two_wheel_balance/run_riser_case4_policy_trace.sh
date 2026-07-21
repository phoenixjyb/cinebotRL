#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
CASE=4
PADDED=0004
PLAN_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
CAPTURE_STAMP="20260721_initial_teacher42_raw_capture_v1_exclusive"
POLICY_STAMP="20260721_initial_teacher40_bc_previous_action_masked_v1"
OUTPUT_STAMP="20260721_case4_policy_rate_trace_teacher_masked_v1_exclusive"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
CAPTURE_ROOT="$ROOT/artifacts/two_wheel_riser/$CAPTURE_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$OUTPUT_STAMP"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$OUTPUT_STAMP"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
TEACHER_GATE="$CAPTURE_ROOT/gates/case_${PADDED}.json"
POLICY_FINAL="$POLICY_ROOT/final_status.json"
POLICY_REPORT="$POLICY_ROOT/report.json"
POLICY="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP\\residual_policy.torchscript.pt"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
PLAN_SHA256="16e962e57b906d18561cc8640c4788c719bf95817492896c252affe6920e3ddb"
TEACHER_GATE_SHA256="245f592ec81eee562a1e0dc4afb278c77c3b5d04046b1362701d75d4c995d79c"
POLICY_FINAL_SHA256="ded00f25dde299207dc0e3af0b611418e09d5368d4fc9e7cab53b57df9a36bba"
POLICY_REPORT_SHA256="3f0efb4a2707b343a775dd5dd8b0ad49d6506474da627d8449ca81556cbbcd3e"
POLICY_SHA256="34fa67192f8c66b879eb7d11a83c96ffd2320932e6807f2224cdfa2f74a4c0e4"
PLAYBACK_SHA256="8fff8f71f34b2be789b5ed243e59f4220a19b7b948191561598589447889722f"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
POLICY_COMMIT="7932a9efc35b99e5b87c3a7e8eb653647fce471b"
AUTHORIZATION_SHA256="478467eb076cca58f75e0c41d129766db16d4405d2f63bb2a6781c5e6cf9710e"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi
[[ -x "$PY" ]] || { printf 'missing Isaac Python\n' >&2; exit 2; }

sha256() { sha256sum "$1" | awk '{print $1}'; }
identity_matches() { [[ -s "$1" && "$(sha256 "$1")" == "$2" ]]; }

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true
  )"
  compute_owners="$(
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader
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

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$HEAD" == "$UPSTREAM" ]] || { printf 'HEAD is not pushed\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || exit 3
git -C "$ROOT" merge-base --is-ancestor "$POLICY_COMMIT" "$HEAD"
identity_matches "$PLAN" "$PLAN_SHA256"
identity_matches "$TEACHER_GATE" "$TEACHER_GATE_SHA256"
identity_matches "$POLICY_FINAL" "$POLICY_FINAL_SHA256"
identity_matches "$POLICY_REPORT" "$POLICY_REPORT_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"

python3 - "$POLICY_FINAL" "$POLICY_REPORT" <<'PY'
import json
from pathlib import Path
import sys

final, report = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
checks = {
    "policy_gate": final.get("passed") is True
    and final.get("learned_rollout_authorized") is True,
    "masked": report.get("masked_observation_indices") == [23, 24, 25],
    "validation_only": report.get("offline_gate_splits") == ["validation"],
    "holdout_closed": report.get("holdout_metrics_computed") is False,
    "ppo_closed": report.get("ppo_started") is False,
}
if not all(checks.values()):
    raise SystemExit(f"policy trace admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_case4_policy_trace_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 4,
    "runs": ["deterministic_teacher", "masked_policy"],
    "trace_only": True,
    "valid_for_training": False,
    "dagger_authorized": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "runtime_started": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_CASE4_POLICY_TRACE_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || { printf 'refusing to overwrite %s\n' "$OUTPUT" >&2; exit 5; }
assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/teacher" "$OUTPUT/learned" \
  "$OUTPUT/traces/teacher" "$OUTPUT/traces/learned" "$OUTPUT/logs"
python3 - "$OUTPUT/admission.json" "$HEAD" "$POLICY" "$PLAN" \
  "$TEACHER_GATE" "$AUTHORIZATION_SHA256" "$PLAYBACK" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_case4_policy_trace_admission_v1",
    "runtime_commit": sys.argv[2],
    "policy": identity(sys.argv[3]),
    "plan": identity(sys.argv[4]),
    "teacher_gate": identity(sys.argv[5]),
    "authorization_sha256": sys.argv[6],
    "playback": identity(sys.argv[7]),
    "case": 4,
    "split": "validation",
    "trace_only": True,
    "valid_for_training": False,
    "dagger_authorized": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

run_playback() {
  local mode="$1" status=0
  local policy_args=()
  if [[ "$mode" == learned ]]; then
    policy_args=(--residual-policy "$POLICY_WIN" --residual-policy-device cuda)
  fi
  assert_gpu_free || return 90
  timeout --signal=TERM --kill-after=30s 600 \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" --plan-dir "$PLAN_WIN" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$CASE" --controller-wz-kp 1.05 --maximum-duration-scale 3.0 \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain 1.0 \
    --maximum-camera-lever-arm-correction-m 0.05 \
    --residual-action-scales 0.35,0.40,0.10 \
    --policy-trace-dir "$OUTPUT_WIN\\traces\\$mode" \
    "${policy_args[@]}" \
    --output "$OUTPUT_WIN\\$mode\\case_${PADDED}.json" --headless \
    >"$OUTPUT/logs/$mode.log" 2>&1 || status=$?
  printf '%s\n' "$status" >"$OUTPUT/logs/$mode.exit_code"
  return "$status"
}

TEACHER_STATUS=0
run_playback teacher || TEACHER_STATUS=$?
LEARNED_STATUS=99
if [[ "$TEACHER_STATUS" == 0 ]]; then
  run_playback learned || LEARNED_STATUS=$?
fi
printf '%s\n' "$LEARNED_STATUS" >"$OUTPUT/logs/learned.exit_code"

"$PY" -X utf8 - "$OUTPUT_WIN" "$HEAD" "$TEACHER_STATUS" "$LEARNED_STATUS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
teacher_status, learned_status = map(int, sys.argv[3:])
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
def audit(mode):
    gate_path = root / mode / "case_0004.json"
    trace_path = root / "traces" / mode / "case_0004_policy_trace_v1.npz"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        count = len(data["observations"])
        checks = {
            "schema": metadata.get("schema") == "cinebotrl_two_wheel_riser_policy_trace_v1",
            "trace_only": metadata.get("trace_only") is True,
            "not_trainable": metadata.get("valid_for_training") is False,
            "no_teacher_labels": metadata.get("teacher_labels_present") is False,
            "no_residual_dataset": metadata.get("residual_dataset_present") is False,
            "dagger_closed": metadata.get("dagger_authorized") is False,
            "row_count": count == gate["results"][0]["completed_steps"],
            "observation_width": data["observations"].shape[1] == 65,
            "action_width": data["applied_residual_actions"].shape[1] == 3,
            "gate_trace_path": gate["results"][0]["executed_policy_trace"] == str(trace_path.resolve()),
            "runtime_no_training": gate.get("training_started") is False,
            "runtime_no_ppo": gate.get("ppo_authorized") is False,
            "runtime_no_dagger": gate.get("dagger_authorized") is False,
        }
    return {"checks": checks, "passed": all(checks.values()), "gate": identity(gate_path), "trace": identity(trace_path)}
teacher = audit("teacher") if teacher_status == 0 else None
learned = audit("learned") if learned_status == 0 else None
checks = {
    "teacher_exit_zero": teacher_status == 0,
    "teacher_trace_passed": teacher is not None and teacher["passed"],
    "learned_exit_zero": learned_status == 0,
    "learned_trace_passed": learned is not None and learned["passed"],
}
passed = all(checks.values())
(root / "final_status.json").write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_case4_policy_trace_final_v1",
    "runtime_commit": sys.argv[2],
    "checks": checks,
    "teacher": teacher,
    "learned": learned,
    "admission": identity(root / "admission.json"),
    "trace_only": True,
    "valid_for_training": False,
    "dagger_authorized": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "passed": passed,
}, indent=2) + "\n", encoding="utf-8")
raise SystemExit(0 if passed else 1)
PY
