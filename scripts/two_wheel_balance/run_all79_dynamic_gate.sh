#!/usr/bin/env bash
set -uo pipefail

ROOT=${CINEBOTRL_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-balance}
OUT=${1:-$ROOT/evaluation_results/two_wheel_all79_playback/all79_promoted_defaults_v2}
RETARGET_DIR=${RETARGET_DIR:-$ROOT/evaluation_results/two_wheel_all79_playback/retargeted_all79_v2_case7_scale125}
PYTHON=${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}
RESULT_SCHEMA=recomo_two_wheel_all79_whole_body_playback_smoke_v2

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
)
raise SystemExit(0 if valid else 1)
PY
}

for case in $(seq 1 79); do
  tag=$(printf '%04d' "$case")
  [[ -f "$RETARGET_DIR/case_${tag}.npz" ]] || fail missing_retarget_input "$case"
done

python3 - "$RETARGET_DIR/summary.json" <<'PY' || fail invalid_retarget_summary
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
valid = result.get("passed") is True and result.get("passed_case_count") == 79
raise SystemExit(0 if valid else 1)
PY

for case in $(seq 1 79); do
  tag=$(printf '%04d' "$case")
  result="$OUT/case_${tag}.json"
  console="$OUT/case_${tag}.console.log"

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
    --urdf 'G:\wSpace\cinebotRL-two-wheel-balance\assets_own\recomoProto2_two_wheel_whole_body\recomoProto2_two_wheel_whole_body.urdf' \
    --cases "$case" \
    --output "$result_win" \
    --headless > "$console" 2>&1
  status=$?

  if ! validate_result "$result" "$case"; then
    fail invalid_or_failed_case_result "$case" "$status"
  fi
  printf '%s case=%s state=passed interop_exit=%s\n' \
    "$(date -Iseconds)" "$case" "$status"
done

python3 - "$OUT" "$RESULT_SCHEMA" <<'PY' || fail aggregate_failed
import json
from pathlib import Path
import sys

output_dir, expected_schema = Path(sys.argv[1]), sys.argv[2]
case_results = []
for case in range(1, 80):
    with (output_dir / f"case_{case:04d}.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != expected_schema or payload.get("passed") is not True:
        raise SystemExit(f"invalid case result while aggregating: {case}")
    case_results.append(payload["results"][0])

summary = {
    "schema": "recomo_two_wheel_all79_dynamic_gate_summary_v1",
    "training_started": False,
    "controller_profile": "structural_robust_v1",
    "task_space_feedback_enabled": True,
    "com_pitch_feedforward_enabled": True,
    "arm_gravity_feedforward_enabled": True,
    "phase_governor_enabled": False,
    "arm_stiffness": 400.0,
    "arm_damping": 40.0,
    "requested_case_count": 79,
    "passed_case_count": sum(item["passed"] for item in case_results),
    "maximum_peak_pitch_deg": max(item["peak_pitch_deg"] for item in case_results),
    "maximum_peak_arm_error_deg": max(item["peak_arm_error_deg"] for item in case_results),
    "maximum_peak_arm_effort_nm": max(item["peak_arm_effort_nm"] for item in case_results),
    "maximum_tool_position_p95_m": max(item["position_error_p95_m"] for item in case_results),
    "maximum_tool_position_error_m": max(item["position_error_max_m"] for item in case_results),
    "maximum_wheel_action_saturation_ratio": max(item["action_saturation_ratio"] for item in case_results),
    "maximum_arm_effort_saturation_ratio": max(item["arm_effort_saturation_ratio"] for item in case_results),
    "cases": case_results,
    "passed": all(item["passed"] for item in case_results),
}
with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
PY

touch "$OUT/COMPLETE"
printf '%s state=complete cases=79\n' "$(date -Iseconds)"
