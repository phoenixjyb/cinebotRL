#!/usr/bin/env bash
# Validate only accepted exact-source-v1 candidates under Isaac Lab.
# This is fail-fast and resumable; it never starts PPO or promotes rejected or
# runtime-quarantined candidates.

set -uo pipefail

ROOT=${CINEBOTRL_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-balance}
RETARGET_DIR=${RETARGET_DIR:-$ROOT/evaluation_results/two_wheel_exact_source_v1/gate2_all79_offline}
OUT=${1:-$ROOT/evaluation_results/two_wheel_exact_source_v1/gate4_accepted_dynamic}
PYTHON=${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}
RESULT_SCHEMA=recomo_two_wheel_corrected_full_pose_playback_smoke_v3
EXACT_SOURCE_MANIFEST_SHA256=f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d

mkdir -p "$OUT"
rm -f "$OUT/COMPLETE" "$OUT/FAILED.json"
echo "$$" > "$OUT/runner.pid"

fail() {
  local reason=$1
  local case_number=${2:-null}
  local interop_exit=${3:-null}
  python3 - "$OUT/FAILED.json" "$reason" "$case_number" "$interop_exit" <<'PY'
import json
import sys

path, reason, case_number, interop_exit = sys.argv[1:]
payload = {
    "reason": reason,
    "case": None if case_number == "null" else int(case_number),
    "interop_exit_code": None if interop_exit == "null" else int(interop_exit),
    "training_started": False,
}
with open(path, "w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
  printf '%s state=failed reason=%s case=%s interop_exit=%s\n' \
    "$(date -Iseconds)" "$reason" "$case_number" "$interop_exit"
  exit 2
}

validate_result() {
  python3 - "$1" "$2" "$RESULT_SCHEMA" <<'PY'
import json
import sys

path, expected_case, expected_schema = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
except (OSError, ValueError):
    raise SystemExit(1)
valid = (
    result.get("schema") == expected_schema
    and result.get("training_started") is False
    and result.get("cases") == [expected_case]
    and result.get("passed_case_count") == 1
    and result.get("passed") is True
    and result.get("phase_governor_enabled") is True
    and result.get("phase_feedforward_scaled_by_progress") is True
    and result.get("phase_governor_position_stop_m") == 0.15
    and result.get("phase_governor_attitude_start_deg") == 4.0
    and result.get("phase_governor_attitude_stop_deg") == 8.0
    and result.get("physical_gimbal_joint_labels_learned") is False
    and result.get("camera_observation_and_reward_link") == "cam_link"
    and result.get("camera_frame_conversion")
    == "R_world_cam = R_world_DFR * Rz(+pi/2)"
)
raise SystemExit(0 if valid else 1)
PY
}

python3 - \
  "$RETARGET_DIR/summary.json" \
  "$RETARGET_DIR" \
  "$OUT/accepted_cases.txt" \
  "$EXACT_SOURCE_MANIFEST_SHA256" <<'PY' || fail invalid_retarget_corpus
import json
from pathlib import Path
import sys

summary_path, candidate_dir, output_path, expected_sha = sys.argv[1:]
candidate_dir = Path(candidate_dir)
with open(summary_path, encoding="utf-8") as stream:
    result = json.load(stream)
rows = result.get("results", [])
accepted = sorted(int(row["case"]) for row in rows if row.get("passed") is True)
rejected = sorted(int(row["case"]) for row in rows if row.get("passed") is not True)
valid = (
    result.get("schema")
    == "cinebotrl_two_wheel_exact_source_retarget_batch_v1"
    and result.get("source_manifest_sha256") == expected_sha
    and result.get("trajectory_integrity_contract") == "exact_source_v1"
    and result.get("source_reference_quality_qualified_teacher") is False
    and result.get("source_reference_valid_for_training") is False
    and result.get("training_started") is False
    and result.get("valid_for_training") is False
    and result.get("requested_cases") == list(range(1, 80))
    and len(rows) == 79
    and accepted == result.get("passed_cases")
    and rejected == result.get("rejected_cases")
    and accepted
    and all((candidate_dir / f"case_{case:04d}.npz").is_file() for case in accepted)
    and all(not (candidate_dir / f"case_{case:04d}.npz").exists() for case in rejected)
)
if not valid:
    raise SystemExit(1)
Path(output_path).write_text(
    "\n".join(str(case) for case in accepted) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY

while read -r case; do
  [[ -n "$case" ]] || continue
  tag=$(printf '%04d' "$case")
  result="$OUT/case_${tag}.json"
  console="$OUT/case_${tag}.console.log"

  candidate_win=$(wslpath -w "$RETARGET_DIR/case_${tag}.npz")
  "$PYTHON" - "$candidate_win" <<'PY' || fail invalid_candidate_contract "$case"
import sys
sys.path.insert(0, r'G:\wSpace\cinebotRL-two-wheel-balance\src')
from pathlib import Path
from rl_platform.tasks.two_wheel_balance.exact_source_reference import (
    validate_exact_source_candidate,
)
validate_exact_source_candidate(Path(sys.argv[1]))
PY

  if [[ -f "$result" ]] && validate_result "$result" "$case"; then
    printf '%s case=%s resume=passed\n' "$(date -Iseconds)" "$case"
    continue
  fi

  rm -f "$result"
  printf '%s case=%s state=start\n' "$(date -Iseconds)" "$case"
  result_win=$(wslpath -w "$result")
  env PYTHONPATH='G:\wSpace\cinebotRL-two-wheel-balance\src' \
    "$PYTHON" \
    'G:\wSpace\cinebotRL-two-wheel-balance\scripts\two_wheel_balance\smoke_all79_whole_body_playback.py' \
    --gains 'G:\wSpace\cinebotRL-two-wheel-balance\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json' \
    --retarget-dir "$(wslpath -w "$RETARGET_DIR")" \
    --urdf 'G:\wSpace\cinebotRL-two-wheel-balance\assets_own\recomoProto2_two_wheel_whole_body_attitude\recomoProto2_two_wheel_whole_body_attitude.urdf' \
    --cases "$case" \
    --enable-phase-governor \
    --output "$result_win" \
    --headless > "$console" 2>&1
  status=$?

  if ! validate_result "$result" "$case"; then
    fail invalid_or_failed_case_result "$case" "$status"
  fi
  printf '%s case=%s state=passed interop_exit=%s\n' \
    "$(date -Iseconds)" "$case" "$status"
done < "$OUT/accepted_cases.txt"

python3 - "$OUT" "$RESULT_SCHEMA" <<'PY' || fail aggregate_failed
import json
from pathlib import Path
import sys

output_dir, expected_schema = Path(sys.argv[1]), sys.argv[2]
accepted = [
    int(line)
    for line in (output_dir / "accepted_cases.txt").read_text().splitlines()
    if line.strip()
]
case_results = []
for case in accepted:
    with (output_dir / f"case_{case:04d}.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != expected_schema or payload.get("passed") is not True:
        raise SystemExit(f"invalid case result while aggregating: {case}")
    case_results.append(payload["results"][0])

summary = {
    "schema": "recomo_two_wheel_corrected_accepted_dynamic_gate_summary_v1",
    "training_started": False,
    "controller_profile": "structural_robust_v1",
    "phase_governor_enabled": True,
    "phase_governor_pitch_start_deg": 10.5,
    "phase_governor_pitch_stop_deg": 11.5,
    "phase_governor_position_stop_m": 0.15,
    "phase_governor_attitude_start_deg": 4.0,
    "phase_governor_attitude_stop_deg": 8.0,
    "phase_feedforward_scaled_by_progress": True,
    "position_target_link": "ee1_tool",
    "camera_observation_and_reward_link": "cam_link",
    "camera_frame_conversion": "R_world_cam = R_world_DFR * Rz(+pi/2)",
    "physical_gimbal_joint_labels_learned": False,
    "accepted_cases": accepted,
    "requested_case_count": len(accepted),
    "passed_case_count": sum(item["passed"] for item in case_results),
    "maximum_peak_pitch_deg": max(item["peak_pitch_deg"] for item in case_results),
    "maximum_peak_arm_error_deg": max(item["peak_arm_error_deg"] for item in case_results),
    "maximum_tool_position_p95_m": max(item["position_error_p95_m"] for item in case_results),
    "maximum_tool_position_error_m": max(item["position_error_max_m"] for item in case_results),
    "maximum_camera_attitude_p95_deg": max(item["attitude_error_p95_deg"] for item in case_results),
    "maximum_camera_attitude_error_deg": max(item["attitude_error_max_deg"] for item in case_results),
    "maximum_wheel_action_saturation_ratio": max(item["action_saturation_ratio"] for item in case_results),
    "maximum_arm_effort_saturation_ratio": max(item["arm_effort_saturation_ratio"] for item in case_results),
    "maximum_gimbal_effort_saturation_ratio": max(item["gimbal_effort_saturation_ratio"] for item in case_results),
    "cases": case_results,
    "passed": all(item["passed"] for item in case_results),
}
with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
PY

touch "$OUT/COMPLETE"
printf '%s state=complete accepted_cases=%s\n' \
  "$(date -Iseconds)" "$(wc -l < "$OUT/accepted_cases.txt")"
