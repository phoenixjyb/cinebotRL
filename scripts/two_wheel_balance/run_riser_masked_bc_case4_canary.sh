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
PROFILE="riser_recovery_direction_v4_camera_lever_arm_v1"
PLAN_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
CAPTURE_STAMP="20260721_initial_teacher42_raw_capture_v1_exclusive"
ZERO_STAMP="20260721_initial_teacher40_bc_case4_rendered_canary_v3"
POLICY_STAMP="20260721_initial_teacher40_bc_previous_action_masked_v1"
OUTPUT_STAMP="20260721_initial_teacher40_bc_previous_action_masked_case4_canary_v1"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
CAPTURE_ROOT="$ROOT/artifacts/two_wheel_riser/$CAPTURE_STAMP"
ZERO_ROOT="$ROOT/artifacts/two_wheel_riser/$ZERO_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$OUTPUT_STAMP"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$OUTPUT_STAMP"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
ROLLOUT_GATE="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
TEACHER_GATE="$CAPTURE_ROOT/gates/case_${PADDED}.json"
ZERO_GATE="$ZERO_ROOT/zero/case_${PADDED}.json"
POLICY_FINAL="$POLICY_ROOT/final_status.json"
POLICY_REPORT="$POLICY_ROOT/report.json"
POLICY="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP\\residual_policy.torchscript.pt"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
PLAN_SHA256="16e962e57b906d18561cc8640c4788c719bf95817492896c252affe6920e3ddb"
TEACHER_GATE_SHA256="245f592ec81eee562a1e0dc4afb278c77c3b5d04046b1362701d75d4c995d79c"
ZERO_GATE_SHA256="bc0c16c1a594021e87c4eeed1ee9396c4a95b8b79946eddb1d9b9dc319a349d1"
POLICY_FINAL_SHA256="ded00f25dde299207dc0e3af0b611418e09d5368d4fc9e7cab53b57df9a36bba"
POLICY_REPORT_SHA256="3f0efb4a2707b343a775dd5dd8b0ad49d6506474da627d8449ca81556cbbcd3e"
POLICY_SHA256="34fa67192f8c66b879eb7d11a83c96ffd2320932e6807f2224cdfa2f74a4c0e4"
PLAYBACK_SHA256="ff078ee6e6ed6cbb23d814547f4c4cb275c238a995c4cb1110796c49dc4e4904"
ROLLOUT_GATE_SHA256="e3e327f2b7bc7f3bdc5f7c27ba36edcf2f3660e5ed4c8b4e4324325764927f5b"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
POLICY_COMMIT="7932a9efc35b99e5b87c3a7e8eb653647fce471b"
AUTHORIZATION_SHA256="06402a55b1cd109cdbe6802e849f093c9743b84578e766d5da69284e286f9f2b"

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
identity_matches "$ZERO_GATE" "$ZERO_GATE_SHA256"
identity_matches "$POLICY_FINAL" "$POLICY_FINAL_SHA256"
identity_matches "$POLICY_REPORT" "$POLICY_REPORT_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$ROLLOUT_GATE" "$ROLLOUT_GATE_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"

python3 - "$POLICY_FINAL" "$POLICY_REPORT" <<'PY'
import json
from pathlib import Path
import sys

final, report = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
checks = {
    "final": final.get("passed") is True
    and final.get("learned_rollout_authorized") is True,
    "masked": report.get("masked_observation_indices") == [23, 24, 25]
    and report.get("previous_action_observation_contract")
    == "masked_after_normalization_v1",
    "validation_only": report.get("offline_gate_splits") == ["validation"],
    "holdout_closed": report.get("holdout_metrics_computed") is False,
    "rollout_not_prestarted": report.get("learned_rollout_started") is False,
    "ppo_closed": report.get("ppo_started") is False,
}
if not all(checks.values()):
    raise SystemExit(f"masked case-4 admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_masked_bc_case4_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 4,
    "split": "validation",
    "rendering": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "runtime_started": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_MASKED_CASE4_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || { printf 'refusing to overwrite %s\n' "$OUTPUT" >&2; exit 5; }
assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/learned" "$OUTPUT/logs"
python3 - "$OUTPUT/admission.json" "$HEAD" "$POLICY" "$PLAN" \
  "$TEACHER_GATE" "$ZERO_GATE" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_masked_bc_case4_admission_v1",
    "runtime_commit": sys.argv[2],
    "policy": identity(sys.argv[3]),
    "plan": identity(sys.argv[4]),
    "teacher_gate": identity(sys.argv[5]),
    "zero_gate": identity(sys.argv[6]),
    "authorization_sha256": sys.argv[7],
    "case": 4,
    "split": "validation",
    "residual_action_scales": [0.35, 0.4, 0.1],
    "holdout_opened": False,
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
  --output "$OUTPUT_WIN\\learned\\case_${PADDED}.json" --headless \
  >"$OUTPUT/logs/playback.log" 2>&1 || PLAYBACK_STATUS=$?
printf '%s\n' "$PLAYBACK_STATUS" >"$OUTPUT/logs/playback.exit_code"

GATE_STATUS=0
python3 "$ROLLOUT_GATE" --mode validation_canary \
  --teacher-dir "$CAPTURE_ROOT/gates" --zero-dir "$ZERO_ROOT/zero" \
  --learned-dir "$OUTPUT/learned" --cases "$CASE" --policy "$POLICY" \
  --expected-tracking-profile "$PROFILE" --maximum-regression-fraction 0.05 \
  --minimum-zero-improvement-fraction 0.05 --output "$OUTPUT/summary.json" \
  >"$OUTPUT/logs/gate.log" 2>&1 || GATE_STATUS=$?
printf '%s\n' "$GATE_STATUS" >"$OUTPUT/logs/gate.exit_code"

python3 - "$OUTPUT" "$HEAD" "$PLAYBACK_STATUS" "$GATE_STATUS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
playback_status, gate_status = map(int, sys.argv[3:])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
learned = json.loads((root / "learned/case_0004.json").read_text(encoding="utf-8"))
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
checks = {
    "playback_exit_zero": playback_status == 0,
    "dynamic_quality": learned.get("dynamic_quality_passed") is True,
    "gate_exit_zero": gate_status == 0,
    "gate_passed": summary.get("passed") is True,
    "case_only": summary.get("cases") == [4],
}
passed = all(checks.values())
(root / "final_status.json").write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_masked_bc_case4_final_v1",
    "runtime_commit": sys.argv[2],
    "checks": checks,
    "admission": identity(root / "admission.json"),
    "learned_gate": identity(root / "learned/case_0004.json"),
    "summary": identity(root / "summary.json"),
    "holdout_opened": False,
    "ppo_authorized": False,
    "ppo_started": False,
    "passed": passed,
}, indent=2) + "\n", encoding="utf-8")
PY

exit "$GATE_STATUS"
