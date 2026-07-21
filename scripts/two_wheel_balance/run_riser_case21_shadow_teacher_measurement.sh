#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
CASE=21
PADDED=0021
PLAN_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
POLICY_STAMP="20260721_initial_teacher40_bc_previous_action_masked_v1"
DATASET_STAMP="20260721_initial_teacher41_subset_30_5_5_v1"
PROPOSAL_STAMP="20260721_dagger_shadow_training_case_proposal_v1_cpu"
TEACHER_GATE_STAMP="20260721_initial_teacher42_raw_capture_v1_exclusive"
OUTPUT_STAMP="20260721_case21_shadow_teacher_measurement_v1_exclusive"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
DATASET_ROOT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP"
PROPOSAL_ROOT="$ROOT/artifacts/two_wheel_riser/$PROPOSAL_STAMP"
TEACHER_GATE_ROOT="$ROOT/artifacts/two_wheel_riser/$TEACHER_GATE_STAMP/gates"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$OUTPUT_STAMP"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$OUTPUT_STAMP"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
DIAGNOSIS="$ROOT/scripts/two_wheel_balance/diagnose_riser_shadow_teacher_gap.py"
DIAGNOSIS_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\diagnose_riser_shadow_teacher_gap.py"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
POLICY_FINAL="$POLICY_ROOT/final_status.json"
POLICY_REPORT="$POLICY_ROOT/report.json"
POLICY="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP\\residual_policy.torchscript.pt"
DATASET="$DATASET_ROOT/initial_teacher40_30_5_5_v1.npz"
DATASET_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$DATASET_STAMP\\initial_teacher40_30_5_5_v1.npz"
PROPOSAL="$PROPOSAL_ROOT/proposal.json"
RANKING="$PROPOSAL_ROOT/ranking.json"
TEACHER_GATE="$TEACHER_GATE_ROOT/case_${PADDED}.json"
TEACHER_GATE_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$TEACHER_GATE_STAMP\\gates\\case_${PADDED}.json"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
PLAN_SHA256="81c0da4be22d5b800978d1d46ca9705912f72007f7c615b31715c672dd86a1d4"
POLICY_FINAL_SHA256="ded00f25dde299207dc0e3af0b611418e09d5368d4fc9e7cab53b57df9a36bba"
POLICY_REPORT_SHA256="3f0efb4a2707b343a775dd5dd8b0ad49d6506474da627d8449ca81556cbbcd3e"
POLICY_SHA256="34fa67192f8c66b879eb7d11a83c96ffd2320932e6807f2224cdfa2f74a4c0e4"
DATASET_SHA256="53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44"
PROPOSAL_SHA256="5bc65581cda29cca7668a16d40d77df3cda717b43f87bc2744d4676dd0d3ef7d"
RANKING_SHA256="6019780884c6c4d9279742d918d99f32e426c737b6938735a46e46c100cf12db"
TEACHER_GATE_SHA256="35a72e768d12d162d7522d15500434268d020e90d58657fe27cef2c19d7068de"
PLAYBACK_SHA256="fa9fc87a1fef287b738dcefe3da2f6b450c5b6ca5140edaf4ade79a2dffee343"
DIAGNOSIS_SHA256="2655bc957c8524c4c9b0ae3fba964b2333f63e0b5ed20bafc2d8012a0a4fb870"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
POLICY_COMMIT="7932a9efc35b99e5b87c3a7e8eb653647fce471b"
AUTHORIZATION_SHA256="cc3c0f03b8140e37c27620a3755e20cbfc8ca22a1842a6ec83ddad166aa2b2fb"

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
identity_matches "$POLICY_FINAL" "$POLICY_FINAL_SHA256"
identity_matches "$POLICY_REPORT" "$POLICY_REPORT_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$DATASET" "$DATASET_SHA256"
identity_matches "$PROPOSAL" "$PROPOSAL_SHA256"
identity_matches "$RANKING" "$RANKING_SHA256"
identity_matches "$TEACHER_GATE" "$TEACHER_GATE_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$DIAGNOSIS" "$DIAGNOSIS_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"

python3 - "$POLICY_FINAL" "$POLICY_REPORT" "$PROPOSAL" "$RANKING" \
  "$TEACHER_GATE" "$PLAN_SHA256" "$TEACHER_GATE_SHA256" <<'PY'
import json
from pathlib import Path
import sys

policy_final, report, proposal, ranking, teacher_gate = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:6]
]
teacher_result = teacher_gate["results"][0]
candidate = proposal.get("candidate_artifacts", {}).get("21", {})
checks = {
    "policy_gate": policy_final.get("passed") is True
    and policy_final.get("learned_rollout_authorized") is True,
    "masked": report.get("masked_observation_indices") == [23, 24, 25],
    "holdout_closed": report.get("holdout_metrics_computed") is False,
    "proposal_schema": proposal.get("schema")
    == "cinebotrl_two_wheel_riser_dagger_shadow_measurement_proposal_v1",
    "proposal_case": proposal.get("proposed_cases") == [21]
    and proposal.get("proposed_case_split") == "train",
    "proposal_runtime_closed": proposal.get("runtime_authorized") is False
    and proposal.get("authorization_token_issued") is False,
    "proposal_training_closed": proposal.get("dataset_creation_authorized") is False
    and proposal.get("dagger_authorized") is False
    and proposal.get("bc_authorized") is False
    and proposal.get("ppo_authorized") is False,
    "ranking_case": ranking.get("selected_training_cases", [None])[0] == 21
    and 21 in ranking.get("train_cases", []),
    "ranking_split_exclusions": 21 not in ranking.get("validation_cases", [])
    and 21 not in ranking.get("holdout_cases", []),
    "candidate_plan": candidate.get("plan", {}).get("sha256") == sys.argv[6],
    "candidate_teacher_gate": candidate.get("teacher_gate", {}).get("sha256")
    == sys.argv[7],
    "teacher_gate": teacher_gate.get("passed") is True
    and teacher_result.get("case") == 21
    and teacher_result.get("dynamic_quality_passed") is True
    and teacher_result.get("thermal_admission_passed") is True
    and teacher_result.get("controller_evidence_passed") is True,
}
if not all(checks.values()):
    raise SystemExit(f"case-21 shadow admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_case21_shadow_teacher_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 21,
    "split": "train",
    "shadow_teacher_measurement_authorized": True,
    "shadow_teacher_applied_to_commands": False,
    "dataset_creation_authorized": False,
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

AUTHORIZATION_FILE="${RISER_CASE21_SHADOW_TEACHER_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || { printf 'refusing to overwrite %s\n' "$OUTPUT" >&2; exit 5; }
assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/learned" "$OUTPUT/traces" "$OUTPUT/diagnosis" "$OUTPUT/logs"
python3 - "$OUTPUT/admission.json" "$HEAD" "$POLICY" "$PLAN" "$DATASET" \
  "$PROPOSAL" "$RANKING" "$TEACHER_GATE" "$AUTHORIZATION_SHA256" \
  "$PLAYBACK" "$DIAGNOSIS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_case21_shadow_teacher_admission_v1",
    "runtime_commit": sys.argv[2],
    "policy": identity(sys.argv[3]),
    "plan": identity(sys.argv[4]),
    "teacher_dataset": identity(sys.argv[5]),
    "proposal": identity(sys.argv[6]),
    "ranking": identity(sys.argv[7]),
    "deterministic_teacher_gate": identity(sys.argv[8]),
    "authorization_sha256": sys.argv[9],
    "playback": identity(sys.argv[10]),
    "diagnosis": identity(sys.argv[11]),
    "case": 21,
    "split": "train",
    "residual_action_scales": [0.35, 0.4, 0.1],
    "shadow_teacher_measurement_authorized": True,
    "shadow_teacher_applied_to_commands": False,
    "dataset_creation_authorized": False,
    "valid_for_training": False,
    "dagger_authorized": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

PLAYBACK_STATUS=0
timeout --signal=TERM --kill-after=30s 600 \
  "$PY" -u -X utf8 "$PLAYBACK_WIN" \
  --gains "$GAINS_WIN" --plan-dir "$PLAN_WIN" \
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
  --cases "$CASE" --controller-wz-kp 1.05 --maximum-duration-scale 3.0 \
  --enable-camera-lever-arm-compensation \
  --camera-lever-arm-compensation-gain 1.0 \
  --maximum-camera-lever-arm-correction-m 0.05 \
  --residual-action-scales 0.35,0.40,0.10 \
  --residual-policy "$POLICY_WIN" --residual-policy-device cuda \
  --shadow-teacher-trace-dir "$OUTPUT_WIN\\traces" \
  --output "$OUTPUT_WIN\\learned\\case_${PADDED}.json" --headless \
  >"$OUTPUT/logs/playback.log" 2>&1 || PLAYBACK_STATUS=$?
printf '%s\n' "$PLAYBACK_STATUS" >"$OUTPUT/logs/playback.exit_code"

DIAGNOSIS_STATUS=99
if [[ "$PLAYBACK_STATUS" == 0 && -s "$OUTPUT/traces/case_${PADDED}_shadow_teacher_trace_v1.npz" ]]; then
  DIAGNOSIS_STATUS=0
  "$PY" -X utf8 "$DIAGNOSIS_WIN" \
    --shadow-trace "$OUTPUT_WIN\\traces\\case_${PADDED}_shadow_teacher_trace_v1.npz" \
    --teacher-dataset "$DATASET_WIN" --case "$CASE" \
    --output "$OUTPUT_WIN\\diagnosis\\shadow_teacher_gap.json" \
    >"$OUTPUT/logs/diagnosis.log" 2>&1 || DIAGNOSIS_STATUS=$?
fi
printf '%s\n' "$DIAGNOSIS_STATUS" >"$OUTPUT/logs/diagnosis.exit_code"

"$PY" -X utf8 - "$OUTPUT_WIN" "$HEAD" "$PLAYBACK_STATUS" \
  "$DIAGNOSIS_STATUS" "$TEACHER_GATE_WIN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
playback_status, diagnosis_status = map(int, sys.argv[3:5])
teacher_gate_path = Path(sys.argv[5])
teacher_result = json.loads(teacher_gate_path.read_text(encoding="utf-8"))["results"][0]
gate_path = root / "learned/case_0021.json"
trace_path = root / "traces/case_0021_shadow_teacher_trace_v1.npz"
diagnosis_path = root / "diagnosis/shadow_teacher_gap.json"

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
result = gate.get("results", [{}])[0]
diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8")) if diagnosis_path.is_file() else {}
checks = {
    "playback_exit_zero": playback_status == 0,
    "diagnosis_exit_zero": diagnosis_status == 0,
    "runtime_output_present": gate_path.is_file(),
    "runtime_passed": gate.get("passed") is True,
    "case": result.get("case") == 21,
    "dynamic_quality": result.get("dynamic_quality_passed") is True,
    "thermal_admission": result.get("thermal_admission_passed") is True,
    "controller_evidence": result.get("controller_evidence_passed") is True,
    "runtime_no_dataset": result.get("executed_residual_dataset") is None,
    "runtime_no_teacher_capture": result.get("executed_raw_teacher_capture") is None,
    "runtime_no_policy_trace": result.get("executed_policy_trace") is None,
    "trace_present": trace_path.is_file(),
    "diagnosis_present": diagnosis_path.is_file(),
    "diagnosis_no_dataset": diagnosis.get("dataset_created") is False,
    "diagnosis_training_closed": diagnosis.get("dagger_authorized") is False
    and diagnosis.get("bc_authorized") is False
    and diagnosis.get("ppo_authorized") is False,
}

if trace_path.is_file():
    with np.load(trace_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        count = len(data["observations"])
        scales = np.asarray(metadata["action_scales"])
        observations = data["observations"]
        applied = data["applied_residual_actions"]
        expected_commands = np.column_stack((
            observations[:, 18] + scales[0] * applied[:, 0],
            observations[:, 19] + scales[1] * applied[:, 1],
            observations[:, 15] + scales[2] * applied[:, 2],
        ))
        expected_commands[:, 0] = np.clip(expected_commands[:, 0], -0.4, 0.4)
        expected_commands[:, 1] = np.clip(expected_commands[:, 1], -0.4, 0.4)
        expected_commands[:, 2] = np.clip(expected_commands[:, 2], 0.0, 1.2)
        checks.update({
            "trace_schema": metadata.get("schema")
            == "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1",
            "trace_only": metadata.get("trace_only") is True,
            "not_trainable": metadata.get("valid_for_training") is False,
            "shadow_unapplied": metadata.get("shadow_teacher_applied_to_commands") is False,
            "labels_unadmitted": metadata.get(
                "shadow_teacher_labels_admitted_for_training"
            ) is False,
            "row_count": count == result.get("completed_steps"),
            "applied_command_reconstruction": np.allclose(
                expected_commands, data["final_high_level_commands"], atol=2e-6
            ),
            "trace_path": result.get("executed_shadow_teacher_trace")
            == str(trace_path.resolve()),
        })

metric_names = (
    "position_error_p95_m",
    "position_error_max_m",
    "attitude_error_p95_deg",
    "attitude_error_max_deg",
    "pitch_max_deg",
)
learned_minus_teacher = {
    name: result.get(name) - teacher_result.get(name)
    for name in metric_names
    if isinstance(result.get(name), (int, float))
    and isinstance(teacher_result.get(name), (int, float))
}
passed = all(checks.values())
payload = {
    "schema": "cinebotrl_two_wheel_riser_case21_shadow_teacher_final_v1",
    "runtime_commit": sys.argv[2],
    "case": 21,
    "split": "train",
    "playback_exit_code": playback_status,
    "diagnosis_exit_code": diagnosis_status,
    "checks": checks,
    "diagnosis_classification": diagnosis.get("classification"),
    "dagger_dataset_proposal_supported": diagnosis.get(
        "dagger_dataset_proposal_supported"
    ),
    "learned_minus_deterministic_teacher_metrics": learned_minus_teacher,
    "admission": identity(root / "admission.json"),
    "deterministic_teacher_gate": identity(teacher_gate_path),
    "shadow_teacher_applied_to_commands": False,
    "dataset_created": False,
    "valid_for_training": False,
    "dagger_authorized": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "passed": passed,
}
if gate_path.is_file():
    payload["learned_gate"] = identity(gate_path)
if trace_path.is_file():
    payload["shadow_trace"] = identity(trace_path)
if diagnosis_path.is_file():
    payload["diagnosis"] = identity(diagnosis_path)
(root / "final_status.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
raise SystemExit(0 if passed else 1)
PY
