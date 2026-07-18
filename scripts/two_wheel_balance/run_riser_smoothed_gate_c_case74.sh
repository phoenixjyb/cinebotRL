#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
AUTHORIZATION="AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE74_RELIEF_V2"
CASE=74
STAMP="20260718_gate_c_smoothed_case74_relief_v2_exclusive"
PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v7_case74_relief_cpu"
MANIFEST_SHA256="0fe4b517d2629a1bca413162378708c2985cf5a42a1da8746de0a662f2fab00c"
SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
PLANNER_COMMIT="b0b0f300543bbc0e140f472ee4c9d3142284a906"
PLAN_SHA256="0acc088a695ff53f9eccfde73107b0748e5de12ffbb6b048efa467455071bf90"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
TIMEOUT_SECONDS=360
MAXIMUM_DURATION_SCALE="2.05"

if [[ "${RISER_SMOOTHED_GATE_C_AUTHORIZATION:-}" != "$AUTHORIZATION" ]]; then
  printf 'smoothed Gate C case74 authorization is absent or unknown\n' >&2
  exit 7
fi

PORTFOLIO_WSL="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PORTFOLIO_STAMP"
SOURCE_MANIFEST_WSL="/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717/manifest.json"
OUTPUT_WSL="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$STAMP"
GAINS_WSL="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
SUMMARIZER="$ROOT/scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"
RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_smoothed_gate_c_case74.sh"
LOADER="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_playback.py"
TRACKING="$ROOT/src/rl_platform/tasks/two_wheel_balance/whole_body_tracking.py"
RISER_CONTROL="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_control.py"
RECOVERY_EVIDENCE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_recovery_evidence.py"

assert_exclusive_gpu() {
  local playback_owners compute_owners
  playback_owners="$(ps -ef | grep -E '[p]ython(\.exe)? .*smoke_.*playback\.py' || true)"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  if [[ -n "$playback_owners" || -n "$compute_owners" ]]; then
    printf 'refusing shared-GPU smoothed Gate C launch\n' >&2
    [[ -z "$playback_owners" ]] || printf '%s\n' "$playback_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    return 1
  fi
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    if assert_exclusive_gpu 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  printf 'GPU owner did not release within 90 seconds\n' >&2
  assert_exclusive_gpu
}

[[ -x "$PY" ]] || { printf 'missing Isaac Python\n' >&2; exit 2; }
[[ -x "$NVIDIA_SMI" ]] || { printf 'missing WSL NVIDIA ownership probe\n' >&2; exit 2; }
[[ -s "$PORTFOLIO_WSL/manifest.json" ]] || { printf 'missing smoothed manifest\n' >&2; exit 2; }
[[ ! -e "$OUTPUT_WSL" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT_WSL" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_MANIFEST_WSL" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO_WSL/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO_WSL/case_0074_smoothed_riser_plan_v1.npz" | awk '{print $1}')" == "$PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$GAINS_WSL" | awk '{print $1}')" == "$GAINS_SHA256" ]] || exit 2
[[ "$(sha256sum "$ROBOT_USD" | awk '{print $1}')" == "$ROBOT_USD_SHA256" ]] || exit 2
"$PY" -X utf8 - "$PORTFOLIO_WIN\\case_0074_smoothed_riser_plan_v1.npz" <<'PY'
import json
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata_json"].item()))
    source_time = np.asarray(data["source_time_s"], dtype=np.float64)
    execution_time = np.asarray(data["execution_time_s"], dtype=np.float64)
    time_alias = np.asarray(data["time_s"], dtype=np.float64)
relief = metadata.get("localized_heading_relief", {})
checks = {
    "schema": metadata.get("schema") == "cinebotrl_two_wheel_riser_smoothed_plan_v1",
    "case": metadata.get("case") == 74,
    "smoothed_target_schema": metadata.get("smoothed_target", {}).get("schema")
    == "derived_smoothed_target_v1",
    "vertical_shift": metadata.get("smoothed_target", {}).get("vertical_shift_m")
    == 0.0,
    "planning_strategy": metadata.get("smoothed_target", {}).get(
        "planning_strategy"
    )
    == "case74_localized_heading_relief_v1",
    "smoothing_sigma": metadata.get("smoothed_target", {}).get(
        "smoothing_sigma_samples"
    )
    == 24.0,
    "smoothing_blend": metadata.get("smoothed_target", {}).get(
        "smoothing_blend_factor"
    )
    == 0.75,
    "localized_heading_relief": relief.get("schema")
    == "case74_localized_heading_relief_v1"
    and relief.get("applied") is True
    and relief.get("start_anchor") == 394
    and relief.get("end_anchor") == 572
    and relief.get("source_geometry_changed") is False
    and relief.get("outside_window_parent_geometry_unchanged") is True
    and relief.get("controller_changed") is False
    and relief.get("phase_governor_changed") is False
    and relief.get("thresholds_changed") is False,
    "source_clock": len(source_time) == 590
    and source_time[0] == 0.0
    and abs(float(source_time[-1]) - 11.373883) <= 1e-9
    and bool(np.all(np.diff(source_time) > 0.0)),
    "execution_clock": len(execution_time) == 590
    and execution_time[0] == 0.0
    and abs(float(execution_time[-1]) - 22.29452723780125) <= 1e-9
    and bool(np.all(np.diff(execution_time) > 0.0)),
    "time_alias_unambiguous": np.array_equal(time_alias, execution_time),
    "dynamic_margin_retime_not_applied": metadata.get(
        "dynamic_margin_retime", {}
    ).get("applied") is False,
}
if not all(checks.values()):
    raise SystemExit(f"invalid case74 playback payload: {checks}")
PY
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make runtime provenance ambiguous\n' >&2
  exit 2
}
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')"
[[ "$COMMIT" == "$UPSTREAM" ]] || { printf 'runtime commit is not pushed\n' >&2; exit 2; }
assert_exclusive_gpu || exit 5

TEMP_ADMISSION="$(mktemp)"
trap 'rm -f "$TEMP_ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --manifest "$PORTFOLIO_WSL/manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_SHA256" \
  --expected-planner-commit "$PLANNER_COMMIT" \
  --expected-count 79 \
  --minimum-candidates 70 \
  --case "$CASE" \
  --output "$TEMP_ADMISSION" >/dev/null

python3 - "$TEMP_ADMISSION" "$COMMIT" "$STAMP" \
  source_manifest "$SOURCE_MANIFEST_WSL" \
  portfolio_manifest "$PORTFOLIO_WSL/manifest.json" \
  selected_plan "$PORTFOLIO_WSL/case_0074_smoothed_riser_plan_v1.npz" \
  lqr_gains "$GAINS_WSL" robot_usd "$ROBOT_USD" \
  tracking_controller "$TRACKING" riser_control "$RISER_CONTROL" \
  recovery_evidence "$RECOVERY_EVIDENCE" playback_loader "$LOADER" \
  playback_runner "$PLAYBACK" wrapper "$RUNNER" validator "$VALIDATOR" \
  summarizer "$SUMMARIZER" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["runtime_commit"] = sys.argv[2]
payload["upstream_commit"] = sys.argv[2]
payload["namespace"] = sys.argv[3]
identity_args = sys.argv[4:]
if len(identity_args) % 2:
    raise SystemExit("runtime identity arguments must be label/path pairs")
payload["runtime_identities"] = {
    identity_args[index]: {
        "path": str(Path(identity_args[index + 1]).resolve()),
        "sha256": hashlib.sha256(Path(identity_args[index + 1]).read_bytes()).hexdigest(),
    }
    for index in range(0, len(identity_args), 2)
}
payload["runtime_authorized"] = payload["passed"] is True
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

mkdir -p "$OUTPUT_WSL/gates" "$OUTPUT_WSL/logs"
mv "$TEMP_ADMISSION" "$OUTPUT_WSL/admission.json"
if ! timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" \
  "$PY" -u -X utf8 "$PLAYBACK_WIN" \
  --gains "$GAINS_WIN" \
  --plan-dir "$PORTFOLIO_WIN" \
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
  --cases "$CASE" \
  --maximum-duration-scale "$MAXIMUM_DURATION_SCALE" \
  --output "$OUTPUT_WIN\\gates\\case_0074.json" \
  --headless >"$OUTPUT_WSL/logs/case_0074.log" 2>&1; then
  python3 "$SUMMARIZER" --root "$OUTPUT_WSL" --git-commit "$COMMIT" \
    --cases "$CASE" --output "$OUTPUT_WSL/summary.json" >/dev/null
  wait_for_gpu_release || exit 5
  printf 'smoothed Gate C stopped on case74\n' >&2
  exit 4
fi

wait_for_gpu_release || exit 5
python3 "$SUMMARIZER" --root "$OUTPUT_WSL" --git-commit "$COMMIT" \
  --cases "$CASE" --output "$OUTPUT_WSL/summary.json" >/dev/null
python3 - "$OUTPUT_WSL/gates/case_0074.json" "$OUTPUT_WSL/summary.json" <<'PY'
import json
from pathlib import Path
import sys

gate = json.loads(Path(sys.argv[1]).read_text())
summary = json.loads(Path(sys.argv[2]).read_text())
result = gate.get("results", [{}])[0]
ok = (
    gate.get("cases") == [74]
    and gate.get("maximum_duration_scale") == 2.05
    and gate.get("completion_horizon_contract")
    == "bounded_execution_duration_scale_v1"
    and gate.get("trajectory_command_source") == "deterministic_teacher"
    and gate.get("residual_policy") is None
    and result.get("executed_residual_dataset") is None
    and gate.get("training_started") is False
    and gate.get("ppo_authorized") is False
    and summary.get("residual_capture_started") is False
    and summary.get("bc_started") is False
    and summary.get("ppo_started") is False
    and result.get("maximum_duration_scale") == 2.05
    and abs(
        result.get("maximum_runtime_s", 0.0)
        - result.get("execution_duration_s", 0.0) * 2.05
    )
    <= 1e-9
)
raise SystemExit(0 if ok else 6)
PY
printf 'smoothed Gate C case74 closed: %s\n' "$OUTPUT_WSL"
