#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
AUTHORIZATION_SHA256="7f3a058422c961a362b50b4ea8d9980664e5fab3b44739acac9668fa461fa370"
STAMP="20260721_initial_teacher42_raw_capture_v1_exclusive"
PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
PLANNER_COMMIT="0391190f536a29f65b4c97968b764f29444c9f43"
SELECTION_SHA256="e0f1d2b44061aabfe64ad2ffa3d23f57bf9b3e51015b2e3fa0703ba24316bb06"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
# Modern camera-lever-arm passes first, legacy-profile evidence follows, and the
# historical error-governor-only case remains last under the homogeneous profile.
CASES=(2 3 4 5 6 7 8 9 13 14 15 16 17 18 19 21 22 30 31 32 33 34 36 37 41 66 67 68 10 11 12 23 24 25 26 28 52 53 70 74 77 20)
CASES_CSV="$(IFS=,; printf '%s' "${CASES[*]}")"
PREFLIGHT_ONLY="${RISER_RAW_TEACHER_BATCH_PREFLIGHT:-0}"

for name in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON RISER_GATE_C_CASES \
  RISER_GATE_C_STAMP RISER_GATE_C_PORTFOLIO_STAMP RISER_DATASET_DIR \
  RISER_RESIDUAL_POLICY; do
  [[ -z "${!name:-}" ]] || {
    printf 'raw-teacher batch rejects environment override: %s\n' "$name" >&2
    exit 7
  }
done

if [[ "$PREFLIGHT_ONLY" != 1 ]]; then
  [[ -n "$AUTHORIZATION_SHA256" ]] || {
    printf 'raw-teacher batch has no issued runtime authorization\n' >&2
    exit 7
  }
  TOKEN_SHA256="$(printf '%s' "${RISER_RAW_TEACHER_BATCH_AUTHORIZATION:-}" | sha256sum | awk '{print $1}')"
  [[ "$TOKEN_SHA256" == "$AUTHORIZATION_SHA256" ]] || {
    printf 'raw-teacher batch authorization is absent or unknown\n' >&2
    exit 7
  }
fi

PORTFOLIO="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"
SOURCE_MANIFEST="/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717/manifest.json"
SELECTION="$ROOT/artifacts/two_wheel_riser/20260720_initial_teacher42_selection_v1/selection.json"
SELECTION_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\20260720_initial_teacher42_selection_v1\\selection.json"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="${WIN_ROOT}\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="${WIN_ROOT}\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
CASE_AUDITOR="$ROOT/scripts/two_wheel_balance/audit_riser_raw_teacher_capture.py"
CASE_AUDITOR_WIN="${WIN_ROOT}\\scripts\\two_wheel_balance\\audit_riser_raw_teacher_capture.py"
CORPUS_AUDITOR="$ROOT/scripts/two_wheel_balance/audit_riser_raw_teacher_corpus.py"
CORPUS_AUDITOR_WIN="${WIN_ROOT}\\scripts\\two_wheel_balance\\audit_riser_raw_teacher_corpus.py"
DATASET_BUILDER="$ROOT/scripts/two_wheel_balance/build_riser_raw_teacher_dataset.py"
DATASET_BUILDER_WIN="${WIN_ROOT}\\scripts\\two_wheel_balance\\build_riser_raw_teacher_dataset.py"
RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_raw_teacher42_capture.sh"
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
    printf 'raw-teacher batch GPU is not exclusive\n' >&2
    [[ -z "$wsl_owners" ]] || printf '%s\n' "$wsl_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    [[ -z "$windows_owners" ]] || printf '%s\n' "$windows_owners" >&2
    return 1
  fi
}

assert_no_competing_cpu() {
  ! ps -ef | grep -qE '[r]etarget_exact_source_v1_nonholonomic\.py' || {
    printf 'raw-teacher batch CPU/disk ownership is not exclusive\n' >&2
    return 1
  }
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    assert_gpu_free 2>/dev/null && return 0
    sleep 1
  done
  printf 'raw-teacher batch GPU did not release within 90 seconds\n' >&2
  return 1
}

write_campaign_status() {
  local reason="$1" stopped_case="$2" exit_code="$3"
  python3 - "$OUTPUT" "$COMMIT" "$CASES_CSV" "$reason" "$stopped_case" "$exit_code" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
cases = [int(item) for item in sys.argv[3].split(",")]
completed = []
for case in cases:
    path = root / f"case_audits/case_{case:04d}.json"
    if not path.is_file():
        continue
    audit = json.loads(path.read_text())
    if audit.get("capture_admission_passed") is True:
        completed.append(case)
pending = [case for case in cases if case not in completed]
admission = root / "admission.json"
output = {
    "schema": "cinebotrl_two_wheel_riser_raw_teacher42_progress_v1",
    "runtime_commit": sys.argv[2],
    "admission_sha256": (
        hashlib.sha256(admission.read_bytes()).hexdigest()
        if admission.is_file() else None
    ),
    "capture_order": cases,
    "completed_cases": completed,
    "completed_case_count": len(completed),
    "pending_cases": pending,
    "next_case": pending[0] if pending else None,
    "reason": sys.argv[4],
    "stopped_case": int(sys.argv[5]) if sys.argv[5] else None,
    "exit_code": int(sys.argv[6]),
    "capture_admission_passed": sys.argv[4] == "corpus_and_dataset_admitted",
    "bc_authorized": False,
    "ppo_authorized": False,
    "training_started": False,
}
temporary = root / "progress_status.json.tmp"
temporary.write_text(json.dumps(output, indent=2) + "\n")
temporary.replace(root / "progress_status.json")
PY
}

[[ -x "$PY" && -x "$NVIDIA_SMI" && -x "$POWERSHELL" ]] || exit 2
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || exit 2
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
  printf 'raw-teacher batch commit is not pushed\n' >&2
  exit 2
}

TEMP_ADMISSION="$(mktemp)"
trap 'rm -f "$TEMP_ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --manifest "$PORTFOLIO/manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_SHA256" \
  --expected-planner-commit "$PLANNER_COMMIT" \
  --expected-count 79 --minimum-candidates 70 --cases "$CASES_CSV" \
  --output "$TEMP_ADMISSION" >/dev/null

python3 - "$TEMP_ADMISSION" "$COMMIT" "$STAMP" "$SELECTION_SHA256" \
  "$MANIFEST_SHA256" "$SOURCE_SHA256" "$CASES_CSV" "$PREFLIGHT_ONLY" \
  source_manifest "$SOURCE_MANIFEST" portfolio_manifest "$PORTFOLIO/manifest.json" \
  selection "$SELECTION" lqr_gains "$GAINS" robot_usd "$ROBOT_USD" \
  playback "$PLAYBACK" case_auditor "$CASE_AUDITOR" \
  corpus_auditor "$CORPUS_AUDITOR" dataset_builder "$DATASET_BUILDER" \
  raw_dataset_contract "$DATASET_MODULE" tracking_controller "$TRACKING" \
  riser_control "$RISER_CONTROL" wrapper "$RUNNER" validator "$VALIDATOR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
cases = [int(item) for item in sys.argv[7].split(",")]
payload.update(
    {
        "runtime_commit": sys.argv[2],
        "upstream_commit": sys.argv[2],
        "namespace": sys.argv[3],
        "selection_sha256": sys.argv[4],
        "portfolio_manifest_sha256": sys.argv[5],
        "source_manifest_sha256": sys.argv[6],
        "requested_cases": cases,
        "capture_order": cases,
        "homogeneous_tracking_profile": (
            "riser_recovery_direction_v4_camera_lever_arm_v1"
        ),
        "controller_wz_kp": 1.05,
        "maximum_duration_scale": 3.0,
        "camera_lever_arm_compensation_enabled": True,
        "camera_lever_arm_compensation_gain": 1.0,
        "maximum_camera_lever_arm_correction_m": 0.05,
        "trajectory_command_source": "deterministic_teacher",
        "raw_teacher_capture_authorized": sys.argv[8] != "1",
        "normalized_dataset_capture_authorized": False,
        "residual_action_application_authorized": False,
        "scale_freeze_authorized_after_corpus_gate_only": True,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "runtime_authorized": payload.get("passed") is True and sys.argv[8] != "1",
    }
)
args = sys.argv[9:]
payload["runtime_identities"] = {
    args[index]: {
        "path": str(Path(args[index + 1]).resolve()),
        "sha256": hashlib.sha256(Path(args[index + 1]).read_bytes()).hexdigest(),
    }
    for index in range(0, len(args), 2)
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  python3 - "$SELECTION" "$TEMP_ADMISSION" <<'PY'
import json
from pathlib import Path
import sys

selection = json.loads(Path(sys.argv[1]).read_text())
admission = json.loads(Path(sys.argv[2]).read_text())
rows = {int(row["case"]): row for row in selection["rows"]}
cases = admission["capture_order"]
steps = sum(int(rows[case]["completed_steps"]) for case in cases)
execution = sum(float(rows[case]["execution_duration_s"]) for case in cases)
nominal_hours = steps / 29.87 / 3600.0
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_raw_teacher42_preflight_v1",
    "runtime_commit": admission["runtime_commit"],
    "namespace": admission["namespace"],
    "case_count": len(cases),
    "capture_order": cases,
    "historical_policy_steps": steps,
    "execution_duration_total_s": execution,
    "estimated_nominal_wall_hours": nominal_hours,
    "estimated_bounded_wall_hours": nominal_hours * 1.20,
    "homogeneous_tracking_profile": admission["homogeneous_tracking_profile"],
    "runtime_authorized": False,
    "output_namespace_created": False,
    "bc_authorized": False,
    "ppo_authorized": False,
}, indent=2))
PY
  exit 0
fi

assert_gpu_free && assert_no_competing_cpu || exit 5
if [[ ! -e "$OUTPUT" ]]; then
  mkdir -p "$OUTPUT/gates" "$OUTPUT/logs" "$OUTPUT/raw_cases" "$OUTPUT/case_audits"
  mv "$TEMP_ADMISSION" "$OUTPUT/admission.json"
  ADMISSION_SHA256="$(sha256sum "$OUTPUT/admission.json" | awk '{print $1}')"
  python3 - "$OUTPUT/campaign_contract.json" "$COMMIT" "$ADMISSION_SHA256" "$CASES_CSV" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_raw_teacher42_campaign_contract_v1",
    "runtime_commit": sys.argv[2],
    "admission_sha256": sys.argv[3],
    "capture_order": [int(item) for item in sys.argv[4].split(",")],
    "resume_requires_same_commit_and_hashes": True,
    "overwrite_permitted": False,
    "stop_on_first_reject": True,
    "bc_authorized": False,
    "ppo_authorized": False,
}, indent=2) + "\n")
PY
else
  python3 - "$OUTPUT" "$COMMIT" "$CASES_CSV" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
contract = json.loads((root / "campaign_contract.json").read_text())
admission = root / "admission.json"
checks = {
    "runtime_commit": contract.get("runtime_commit") == sys.argv[2],
    "capture_order": contract.get("capture_order")
    == [int(item) for item in sys.argv[3].split(",")],
    "admission_hash": contract.get("admission_sha256")
    == hashlib.sha256(admission.read_bytes()).hexdigest(),
    "no_overwrite": contract.get("overwrite_permitted") is False,
    "fail_fast": contract.get("stop_on_first_reject") is True,
}
if not all(checks.values()):
    raise SystemExit(f"resume contract mismatch: {checks}")
PY
fi

ADMISSION_SHA256="$(sha256sum "$OUTPUT/admission.json" | awk '{print $1}')"
write_campaign_status ready "" 0
for CASE in "${CASES[@]}"; do
  PADDED="$(printf '%04d' "$CASE")"
  GATE="$OUTPUT/gates/case_${PADDED}.json"
  RAW="$OUTPUT/raw_cases/case_${PADDED}_executed_raw_teacher_v1.npz"
  AUDIT="$OUTPUT/case_audits/case_${PADDED}.json"
  if [[ -s "$AUDIT" ]]; then
    python3 - "$AUDIT" "$GATE" "$RAW" "$OUTPUT/admission.json" "$CASE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
audit = json.loads(Path(sys.argv[1]).read_text())
checks = {
    "passed": audit.get("capture_admission_passed") is True,
    "case": audit.get("case") == int(sys.argv[5]),
    "gate": audit.get("gate_sha256") == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "raw": audit.get("raw_case_sha256") == hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),
    "admission": audit.get("admission_sha256") == hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "not_trainable": audit.get("valid_for_training") is False,
}
if not all(checks.values()):
    raise SystemExit(f"completed case audit mismatch: {checks}")
PY
    printf 'raw-teacher case %s already audited; resume skip\n' "$PADDED"
    continue
  fi
  if [[ -e "$GATE" || -e "$RAW" ]]; then
    write_campaign_status partial_evidence "$CASE" 6
    printf 'partial case evidence requires manual quarantine: %s\n' "$PADDED" >&2
    exit 6
  fi
  assert_gpu_free && assert_no_competing_cpu || exit 5
  TIMEOUT_SECONDS="$(python3 - "$SELECTION" "$CASE" <<'PY'
import json, math
from pathlib import Path
import sys
rows = {int(row["case"]): row for row in json.loads(Path(sys.argv[1]).read_text())["rows"]}
steps = int(rows[int(sys.argv[2])]["completed_steps"])
print(min(1800, max(480, math.ceil(steps / 20.0 + 180.0))))
PY
)"
  STATUS=0
  timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" --plan-dir "$PORTFOLIO_WIN" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$CASE" --controller-wz-kp 1.05 --maximum-duration-scale 3.0 \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain 1.0 \
    --maximum-camera-lever-arm-correction-m 0.05 \
    --raw-teacher-dir "${OUTPUT_WIN}\\raw_cases" \
    --output "${OUTPUT_WIN}\\gates\\case_${PADDED}.json" --headless \
    >"$OUTPUT/logs/case_${PADDED}.log" 2>&1 || STATUS=$?
  printf '%s\n' "$STATUS" >"$OUTPUT/logs/case_${PADDED}.exit_code"
  wait_for_gpu_release || STATUS=5
  if [[ "$STATUS" != 0 || ! -s "$GATE" || ! -s "$RAW" ]]; then
    write_campaign_status runtime_or_physical_reject "$CASE" 4
    printf 'raw-teacher batch stopped on case %s runtime/evidence failure\n' "$PADDED" >&2
    exit 4
  fi
  AUDIT_STATUS=0
  "$PY" -X utf8 "$CASE_AUDITOR_WIN" \
    --gate "${OUTPUT_WIN}\\gates\\case_${PADDED}.json" \
    --admission "${OUTPUT_WIN}\\admission.json" \
    --raw-case "${OUTPUT_WIN}\\raw_cases\\case_${PADDED}_executed_raw_teacher_v1.npz" \
    --selection "$SELECTION_WIN" --case "$CASE" \
    --output "${OUTPUT_WIN}\\case_audits\\case_${PADDED}.json" \
    >"$OUTPUT/logs/case_${PADDED}_audit.log" 2>&1 || AUDIT_STATUS=$?
  if [[ "$AUDIT_STATUS" != 0 ]]; then
    write_campaign_status case_audit_reject "$CASE" 4
    printf 'raw-teacher batch stopped on case %s admission failure\n' "$PADDED" >&2
    exit 4
  fi
  write_campaign_status case_admitted "" 0
done

"$PY" -X utf8 "$CORPUS_AUDITOR_WIN" \
  --selection "$SELECTION_WIN" --admission "${OUTPUT_WIN}\\admission.json" \
  --gate-dir "${OUTPUT_WIN}\\gates" --raw-dir "${OUTPUT_WIN}\\raw_cases" \
  --expected-count 42 --output "${OUTPUT_WIN}\\corpus_audit.json" \
  >"$OUTPUT/logs/corpus_audit.log" 2>&1
"$PY" -X utf8 "$DATASET_BUILDER_WIN" \
  --corpus-audit "${OUTPUT_WIN}\\corpus_audit.json" \
  --output "${OUTPUT_WIN}\\initial_teacher40_30_5_5_v1.npz" \
  >"$OUTPUT/logs/dataset_build.log" 2>&1

python3 - "$OUTPUT" "$COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
corpus = json.loads((root / "corpus_audit.json").read_text())
summary = json.loads((root / "initial_teacher40_30_5_5_v1.summary.json").read_text())
output = {
    "schema": "cinebotrl_two_wheel_riser_raw_teacher42_campaign_final_v1",
    "runtime_commit": sys.argv[2],
    "case_count": corpus["case_count"],
    "capture_admission_passed": corpus["capture_admission_passed"],
    "dataset_admission_passed": summary["dataset_admission_passed"],
    "corpus_audit": identity(root / "corpus_audit.json"),
    "dataset": identity(root / "initial_teacher40_30_5_5_v1.npz"),
    "dataset_summary": identity(root / "initial_teacher40_30_5_5_v1.summary.json"),
    "valid_for_bc_initialization": summary["valid_for_bc_initialization"],
    "bc_authorized": False,
    "ppo_authorized": False,
    "training_started": False,
    "passed": corpus["passed"] is True and summary["dataset_admission_passed"] is True,
}
(root / "final_status.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
PY
write_campaign_status corpus_and_dataset_admitted "" 0
