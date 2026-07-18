#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
AUTHORIZATION="AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE68_66_CAMERA_LEVER_ARM_V1"
CASES="68,66"
STAMP="20260718_gate_c_smoothed_case68_66_camera_lever_arm_v1_exclusive"
PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v7_case74_relief_cpu"
MANIFEST_SHA256="0fe4b517d2629a1bca413162378708c2985cf5a42a1da8746de0a662f2fab00c"
SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
PLANNER_COMMIT="b0b0f300543bbc0e140f472ee4c9d3142284a906"
CASE68_PLAN_SHA256="4f4fc302402c53533f4bdbed33682bf52971a6f0cb93af3b42bd6da5ffeed142"
CASE66_PLAN_SHA256="ebdaf9a2e60e66c6231931bec6087c0b36a0895e22d4ee659e2b056b9b21bc37"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
TIMEOUT_SECONDS=480
MAXIMUM_DURATION_SCALE="3.00"
CONTROLLER_WZ_KP="1.05"
CAMERA_LEVER_ARM_GAIN="1.00"
MAXIMUM_CAMERA_LEVER_ARM_CORRECTION_M="0.05"
TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_v1"

if [[ "${RISER_CAMERA_LEVER_ARM_GATE_C_AUTHORIZATION:-}" != "$AUTHORIZATION" ]]; then
  printf 'camera lever-arm Gate C authorization is absent or unknown\n' >&2
  exit 7
fi

PORTFOLIO="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"
SOURCE_MANIFEST="/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717/manifest.json"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\scripts\two_wheel_balance\smoke_riser_reference_playback.py"
SUMMARIZER="$ROOT/scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"
RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_smoothed_gate_c_camera_lever_arm.sh"
LOADER="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_playback.py"
TRACKING="$ROOT/src/rl_platform/tasks/two_wheel_balance/whole_body_tracking.py"
RISER_CONTROL="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_control.py"
RECOVERY_EVIDENCE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_recovery_evidence.py"

assert_gpu_free() {
  local playback_owners compute_owners
  playback_owners="$(ps -ef | grep -E '[p]ython(\.exe)? .*smoke_.*playback\.py' || true)"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  if [[ -n "$playback_owners" || -n "$compute_owners" ]]; then
    printf 'camera lever-arm Gate C GPU is not free\n' >&2
    [[ -z "$playback_owners" ]] || printf '%s\n' "$playback_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    return 1
  fi
}

assert_no_competing_cpu() {
  ! ps -ef | grep -qE '[r]etarget_exact_source_v1_nonholonomic\.py' || {
    printf 'camera lever-arm Gate C CPU/disk ownership is not exclusive\n' >&2
    return 1
  }
}

assert_exclusive_resources() {
  assert_gpu_free && assert_no_competing_cpu
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    assert_gpu_free 2>/dev/null && return 0
    sleep 1
  done
  printf 'camera lever-arm Gate C GPU did not release within 90 seconds\n' >&2
  return 1
}

case_gate_passed() {
  python3 - "$1" "$2" <<'PY'
import json
import math
from pathlib import Path
import sys

gate = json.loads(Path(sys.argv[1]).read_text())
case = int(sys.argv[2])
result = gate.get("results", [{}])[0]
correction_max = result.get("camera_lever_arm_correction_max_m")
raw_max = result.get("camera_lever_arm_raw_correction_max_m")
saturation_ratio = result.get("camera_lever_arm_correction_saturation_ratio")
numeric = (correction_max, raw_max, saturation_ratio)
ok = (
    gate.get("cases") == [case]
    and len(gate.get("results", [])) == 1
    and gate.get("dynamic_quality_passed") is True
    and result.get("dynamic_quality_passed") is True
    and gate.get("thermal_admission_passed") is True
    and result.get("thermal_admission_passed") is True
    and gate.get("controller_evidence_passed") is True
    and result.get("controller_evidence_passed") is True
    and gate.get("controller_overrides") == {"wz_kp": 1.05}
    and gate.get("tracking_profile")
    == "riser_recovery_direction_v4_camera_lever_arm_v1"
    and gate.get("camera_lever_arm_compensation_contract")
    == "measured_camera_to_base_xy_offset_v1"
    and gate.get("camera_lever_arm_compensation_enabled") is True
    and gate.get("camera_lever_arm_compensation_gain") == 1.0
    and gate.get("maximum_camera_lever_arm_correction_m") == 0.05
    and result.get("camera_lever_arm_compensation_enabled") is True
    and result.get("camera_lever_arm_compensation_gain") == 1.0
    and result.get("maximum_camera_lever_arm_correction_m") == 0.05
    and result.get("camera_lever_arm_telemetry_observed") is True
    and result.get("camera_lever_arm_telemetry_sample_count")
    == result.get("completed_steps")
    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric)
    and 0.0 <= correction_max <= 0.05 + 1e-9
    and raw_max + 1e-12 >= correction_max
    and 0.0 <= saturation_ratio <= 1.0
    and gate.get("trajectory_command_source") == "deterministic_teacher"
    and gate.get("residual_policy") is None
    and result.get("executed_residual_dataset") is None
    and result.get("raw_residual_label_applied_to_commands") is False
    and gate.get("training_started") is False
    and gate.get("ppo_authorized") is False
    and isinstance(result.get("residual_label_envelope_passed"), bool)
)
raise SystemExit(0 if ok else 6)
PY
}

[[ -x "$PY" && -x "$NVIDIA_SMI" ]] || exit 2
[[ ! -e "$OUTPUT" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO/case_0068_smoothed_riser_plan_v1.npz" | awk '{print $1}')" == "$CASE68_PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO/case_0066_smoothed_riser_plan_v1.npz" | awk '{print $1}')" == "$CASE66_PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$GAINS" | awk '{print $1}')" == "$GAINS_SHA256" ]] || exit 2
[[ "$(sha256sum "$ROBOT_USD" | awk '{print $1}')" == "$ROBOT_USD_SHA256" ]] || exit 2

git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make runtime provenance ambiguous\n' >&2
  exit 2
}
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')"
[[ "$COMMIT" == "$UPSTREAM" ]] || { printf 'runtime commit is not pushed\n' >&2; exit 2; }
assert_exclusive_resources || exit 5

TEMP_ADMISSION="$(mktemp)"
trap 'rm -f "$TEMP_ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --manifest "$PORTFOLIO/manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_SHA256" \
  --expected-planner-commit "$PLANNER_COMMIT" \
  --expected-count 79 --minimum-candidates 70 --cases "$CASES" \
  --output "$TEMP_ADMISSION" >/dev/null

python3 - "$TEMP_ADMISSION" "$COMMIT" "$STAMP" \
  source_manifest "$SOURCE_MANIFEST" portfolio_manifest "$PORTFOLIO/manifest.json" \
  case68_plan "$PORTFOLIO/case_0068_smoothed_riser_plan_v1.npz" \
  case66_plan "$PORTFOLIO/case_0066_smoothed_riser_plan_v1.npz" \
  lqr_gains "$GAINS" robot_usd "$ROBOT_USD" playback "$PLAYBACK" \
  tracking_controller "$TRACKING" riser_control "$RISER_CONTROL" \
  recovery_evidence "$RECOVERY_EVIDENCE" playback_loader "$LOADER" \
  wrapper "$RUNNER" summarizer "$SUMMARIZER" validator "$VALIDATOR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["runtime_commit"] = payload["upstream_commit"] = sys.argv[2]
payload["namespace"] = sys.argv[3]
args = sys.argv[4:]
payload["runtime_identities"] = {
    args[index]: {
        "path": str(Path(args[index + 1]).resolve()),
        "sha256": hashlib.sha256(Path(args[index + 1]).read_bytes()).hexdigest(),
    }
    for index in range(0, len(args), 2)
}
payload["tracking_profile"] = "riser_recovery_direction_v4_camera_lever_arm_v1"
payload["camera_lever_arm_compensation_contract"] = "measured_camera_to_base_xy_offset_v1"
payload["camera_lever_arm_compensation_gain"] = 1.0
payload["maximum_camera_lever_arm_correction_m"] = 0.05
payload["runtime_authorized"] = payload["passed"] is True
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

mkdir -p "$OUTPUT/gates" "$OUTPUT/logs"
mv "$TEMP_ADMISSION" "$OUTPUT/admission.json"

for CASE in 68 66; do
  assert_exclusive_resources || exit 5
  STATUS=0
  timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" --plan-dir "$PORTFOLIO_WIN" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$CASE" --controller-wz-kp "$CONTROLLER_WZ_KP" \
    --maximum-duration-scale "$MAXIMUM_DURATION_SCALE" \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain "$CAMERA_LEVER_ARM_GAIN" \
    --maximum-camera-lever-arm-correction-m "$MAXIMUM_CAMERA_LEVER_ARM_CORRECTION_M" \
    --output "$OUTPUT_WIN\gates\case_$(printf '%04d' "$CASE").json" --headless \
    >"$OUTPUT/logs/case_$(printf '%04d' "$CASE").log" 2>&1 || STATUS=$?
  printf '%s\n' "$STATUS" >"$OUTPUT/logs/case_$(printf '%04d' "$CASE").exit_code"
  wait_for_gpu_release || exit 5
  if [[ ! -s "$OUTPUT/gates/case_$(printf '%04d' "$CASE").json" ]] \
    || ! case_gate_passed "$OUTPUT/gates/case_$(printf '%04d' "$CASE").json" "$CASE"; then
    python3 "$SUMMARIZER" --root "$OUTPUT" --git-commit "$COMMIT" --cases "$CASES" \
      --expected-tracking-profile "$TRACKING_PROFILE" \
      --require-camera-lever-arm-compensation --output "$OUTPUT/summary.json" >/dev/null
    printf 'camera lever-arm Gate C stopped on case %s\n' "$CASE" >&2
    exit 4
  fi
done

python3 "$SUMMARIZER" --root "$OUTPUT" --git-commit "$COMMIT" --cases "$CASES" \
  --expected-tracking-profile "$TRACKING_PROFILE" \
  --require-camera-lever-arm-compensation --output "$OUTPUT/summary.json" >/dev/null
python3 - "$OUTPUT/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
ok = (
    summary.get("requested_cases") == [68, 66]
    and summary.get("dynamically_passed_cases") == [68, 66]
    and summary.get("first_dynamic_reject") is None
    and summary.get("dynamic_quality_passed") is True
    and summary.get("thermal_admission_passed") is True
    and summary.get("controller_evidence_passed") is True
    and summary.get("runtime_contract_passed") is True
    and summary.get("residual_capture_started") is False
    and summary.get("bc_started") is False
    and summary.get("ppo_started") is False
    and summary.get("valid_for_final_gate_c") is True
    and summary.get("valid_for_training") is False
)
raise SystemExit(0 if ok else 6)
PY
printf 'camera lever-arm Gate C closed: %s\n' "$OUTPUT"
