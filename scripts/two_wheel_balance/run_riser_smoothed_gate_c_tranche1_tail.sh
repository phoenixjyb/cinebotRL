#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
AUTHORIZATION="AUTHORIZED_RISER_SMOOTHED_GATE_C_TRANCHE1_TAIL_12_11_23_HORIZON300_V1"
CASES="12,11,23"
STAMP="20260718_gate_c_smoothed_tranche1_tail_12_11_23_horizon300_wzkp105_v1_exclusive"
PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v7_case74_relief_cpu"
MANIFEST_SHA256="0fe4b517d2629a1bca413162378708c2985cf5a42a1da8746de0a662f2fab00c"
SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
PLANNER_COMMIT="b0b0f300543bbc0e140f472ee4c9d3142284a906"
CASE12_PLAN_SHA256="4f4f4ed45e618ce2ae350aba430e6e20e78d3d63b631dbed8a742a726023097b"
CASE11_PLAN_SHA256="538ddf56b161f93388040284626a9eae01fadbc88cfac8405a5e7848654292b2"
CASE23_PLAN_SHA256="ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
TIMEOUT_SECONDS=480
MAXIMUM_DURATION_SCALE="3.00"
CONTROLLER_WZ_KP="1.05"

if [[ "${RISER_SMOOTHED_GATE_C_TRANCHE1_TAIL_AUTHORIZATION:-}" != "$AUTHORIZATION" ]]; then
  printf 'tranche-1 tail Gate C authorization is absent or unknown\n' >&2
  exit 7
fi

PORTFOLIO_WSL="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"
SOURCE_MANIFEST_WSL="/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717/manifest.json"
OUTPUT_WSL="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"
GAINS_WSL="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\scripts\two_wheel_balance\smoke_riser_reference_playback.py"
SUMMARIZER="$ROOT/scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"
RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_smoothed_gate_c_tranche1_tail.sh"
LOADER="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_playback.py"
TRACKING="$ROOT/src/rl_platform/tasks/two_wheel_balance/whole_body_tracking.py"
RISER_CONTROL="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_control.py"
RECOVERY_EVIDENCE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_recovery_evidence.py"

assert_gpu_free() {
  local playback_owners compute_owners
  playback_owners="$(ps -ef | grep -E '[p]ython(\.exe)? .*smoke_.*playback\.py' || true)"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  if [[ -n "$playback_owners" || -n "$compute_owners" ]]; then
    printf 'tranche-1 tail GPU is not free\n' >&2
    [[ -z "$playback_owners" ]] || printf '%s\n' "$playback_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    return 1
  fi
}

assert_no_competing_cpu() {
  local competing_cpu
  competing_cpu="$(ps -ef | grep -E '[r]etarget_exact_source_v1_nonholonomic\.py' || true)"
  if [[ -n "$competing_cpu" ]]; then
    printf 'tranche-1 tail CPU/disk ownership is not exclusive\n%s\n' "$competing_cpu" >&2
    return 1
  fi
}

assert_exclusive_resources() {
  assert_gpu_free && assert_no_competing_cpu
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    if assert_gpu_free 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  printf 'tranche-1 tail GPU did not release within 90 seconds\n' >&2
  assert_gpu_free
}

case_gate_passed() {
  python3 - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys

gate = json.loads(Path(sys.argv[1]).read_text())
case = int(sys.argv[2])
result = gate.get("results", [{}])[0]
ok = (
    gate.get("cases") == [case]
    and len(gate.get("results", [])) == 1
    and gate.get("dynamic_quality_passed") is True
    and result.get("dynamic_quality_passed") is True
    and gate.get("thermal_admission_passed") is True
    and result.get("thermal_admission_passed") is True
    and gate.get("controller_overrides") == {"wz_kp": 1.05}
    and gate.get("trajectory_command_source") == "deterministic_teacher"
    and gate.get("residual_policy") is None
    and result.get("executed_residual_dataset") is None
    and result.get("raw_residual_label_applied_to_commands") is False
    and gate.get("training_started") is False
    and gate.get("ppo_authorized") is False
    and isinstance(result.get("residual_label_envelope_passed"), bool)
    and isinstance(result.get("residual_label_admission_passed"), bool)
)
raise SystemExit(0 if ok else 6)
PY
}

[[ -x "$PY" ]] || { printf 'missing Isaac Python\n' >&2; exit 2; }
[[ -x "$NVIDIA_SMI" ]] || { printf 'missing WSL NVIDIA ownership probe\n' >&2; exit 2; }
[[ -s "$PORTFOLIO_WSL/manifest.json" ]] || { printf 'missing smoothed manifest\n' >&2; exit 2; }
[[ ! -e "$OUTPUT_WSL" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT_WSL" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_MANIFEST_WSL" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO_WSL/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO_WSL/case_0012_smoothed_riser_plan_v1.npz" | awk '{print $1}')" == "$CASE12_PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO_WSL/case_0011_smoothed_riser_plan_v1.npz" | awk '{print $1}')" == "$CASE11_PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO_WSL/case_0023_smoothed_riser_plan_v1.npz" | awk '{print $1}')" == "$CASE23_PLAN_SHA256" ]] || exit 2
[[ "$(sha256sum "$GAINS_WSL" | awk '{print $1}')" == "$GAINS_SHA256" ]] || exit 2
[[ "$(sha256sum "$ROBOT_USD" | awk '{print $1}')" == "$ROBOT_USD_SHA256" ]] || exit 2

"$PY" -X utf8 - \
  "$PORTFOLIO_WIN\case_0012_smoothed_riser_plan_v1.npz" \
  "$PORTFOLIO_WIN\case_0011_smoothed_riser_plan_v1.npz" \
  "$PORTFOLIO_WIN\case_0023_smoothed_riser_plan_v1.npz" <<'PY'
import json
import sys

import numpy as np

expected = {
    12: (223, 4.235739, 8.080625132264984),
    11: (319, 6.270896, 9.008324292935859),
    23: (506, 9.929694, 9.929693999999989),
}
for path, case in zip(sys.argv[1:], (12, 11, 23), strict=True):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        source_time = np.asarray(data["source_time_s"], dtype=np.float64)
        execution_time = np.asarray(data["execution_time_s"], dtype=np.float64)
        time_alias = np.asarray(data["time_s"], dtype=np.float64)
    states, source_duration, execution_duration = expected[case]
    checks = {
        "schema": metadata.get("schema")
        == "cinebotrl_two_wheel_riser_smoothed_plan_v1",
        "case": metadata.get("case") == case,
        "strategy": metadata.get("smoothed_target", {}).get("planning_strategy")
        == "smoothed_preview_0.05m_g2.75",
        "states": len(source_time) == len(execution_time) == states,
        "source_clock": source_time[0] == 0.0
        and abs(float(source_time[-1]) - source_duration) <= 1e-9
        and bool(np.all(np.diff(source_time) > 0.0)),
        "execution_clock": execution_time[0] == 0.0
        and abs(float(execution_time[-1]) - execution_duration) <= 1e-9
        and bool(np.all(np.diff(execution_time) > 0.0)),
        "time_alias_unambiguous": np.array_equal(time_alias, execution_time),
    }
    if not all(checks.values()):
        raise SystemExit(f"invalid tranche-1 tail case {case}: {checks}")
PY

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
  --manifest "$PORTFOLIO_WSL/manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_SHA256" \
  --expected-planner-commit "$PLANNER_COMMIT" \
  --expected-count 79 \
  --minimum-candidates 70 \
  --cases "$CASES" \
  --output "$TEMP_ADMISSION" >/dev/null

python3 - "$TEMP_ADMISSION" "$COMMIT" "$STAMP" \
  source_manifest "$SOURCE_MANIFEST_WSL" \
  portfolio_manifest "$PORTFOLIO_WSL/manifest.json" \
  case12_plan "$PORTFOLIO_WSL/case_0012_smoothed_riser_plan_v1.npz" \
  case11_plan "$PORTFOLIO_WSL/case_0011_smoothed_riser_plan_v1.npz" \
  case23_plan "$PORTFOLIO_WSL/case_0023_smoothed_riser_plan_v1.npz" \
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

for CASE in 12 11 23; do
  if ! assert_exclusive_resources; then
    python3 "$SUMMARIZER" --root "$OUTPUT_WSL" --git-commit "$COMMIT" \
      --cases "$CASES" --output "$OUTPUT_WSL/summary.json" >/dev/null
    exit 5
  fi
  PLAYBACK_STATUS=0
  timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" \
    --plan-dir "$PORTFOLIO_WIN" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$CASE" \
    --controller-wz-kp "$CONTROLLER_WZ_KP" \
    --maximum-duration-scale "$MAXIMUM_DURATION_SCALE" \
    --output "$OUTPUT_WIN\gates\case_$(printf '%04d' "$CASE").json" \
    --headless >"$OUTPUT_WSL/logs/case_$(printf '%04d' "$CASE").log" 2>&1 \
    || PLAYBACK_STATUS=$?
  printf '%s\n' "$PLAYBACK_STATUS" \
    >"$OUTPUT_WSL/logs/case_$(printf '%04d' "$CASE").exit_code"
  wait_for_gpu_release || exit 5
  if [[ ! -s "$OUTPUT_WSL/gates/case_$(printf '%04d' "$CASE").json" ]] \
    || ! case_gate_passed \
    "$OUTPUT_WSL/gates/case_$(printf '%04d' "$CASE").json" "$CASE"; then
    python3 "$SUMMARIZER" --root "$OUTPUT_WSL" --git-commit "$COMMIT" \
      --cases "$CASES" --output "$OUTPUT_WSL/summary.json" >/dev/null
    printf 'tranche-1 tail Gate C stopped on case %s\n' "$CASE" >&2
    exit 4
  fi
  if ! assert_no_competing_cpu; then
    python3 "$SUMMARIZER" --root "$OUTPUT_WSL" --git-commit "$COMMIT" \
      --cases "$CASES" --output "$OUTPUT_WSL/summary.json" >/dev/null
    exit 5
  fi
done

python3 "$SUMMARIZER" --root "$OUTPUT_WSL" --git-commit "$COMMIT" \
  --cases "$CASES" --output "$OUTPUT_WSL/summary.json" >/dev/null
python3 - "$OUTPUT_WSL" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
ok = (
    summary.get("requested_cases") == [12, 11, 23]
    and summary.get("dynamically_passed_cases") == [12, 11, 23]
    and summary.get("first_dynamic_reject") is None
    and summary.get("dynamic_quality_passed") is True
    and summary.get("thermal_admission_passed") is True
    and summary.get("runtime_contract_passed") is True
    and summary.get("residual_capture_started") is False
    and summary.get("bc_started") is False
    and summary.get("ppo_started") is False
    and summary.get("valid_for_final_gate_c") is True
    and summary.get("valid_for_training") is False
)
for case in (12, 11, 23):
    gate = json.loads((root / "gates" / f"case_{case:04d}.json").read_text())
    result = gate.get("results", [{}])[0]
    ok = ok and (
        gate.get("cases") == [case]
        and gate.get("controller_overrides") == {"wz_kp": 1.05}
        and gate.get("maximum_duration_scale") == 3.0
        and gate.get("trajectory_command_source") == "deterministic_teacher"
        and gate.get("residual_policy") is None
        and result.get("executed_residual_dataset") is None
        and result.get("raw_residual_label_applied_to_commands") is False
        and gate.get("training_started") is False
        and gate.get("ppo_authorized") is False
    )
raise SystemExit(0 if ok else 6)
PY
printf 'tranche-1 tail Gate C closed: %s\n' "$OUTPUT_WSL"
