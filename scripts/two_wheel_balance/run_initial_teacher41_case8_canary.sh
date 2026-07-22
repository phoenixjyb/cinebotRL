#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
CASE=8
PADDED=0008
NAMESPACE="20260722_initial_teacher41_masked_bc_case8_canary_v1_exclusive"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$NAMESPACE"
PLAN_STAMP="20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
TEACHER_ROOT="$ROOT/artifacts/two_wheel_riser/20260718_gate_c_smoothed_case8_dynamic_retime_v1_exclusive/gates"
TEACHER_GATE="$TEACHER_ROOT/case_${PADDED}.json"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/20260722_initial_teacher41_masked_bc_v1"
POLICY="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\20260722_initial_teacher41_masked_bc_v1\\residual_policy.torchscript.pt"
CPU_ADMISSION="$ROOT/artifacts/two_wheel_riser/20260722_initial_teacher41_masked_bc_case8_canary_contract_v1_cpu/admission.json"
CPU_CONTRACT="$ROOT/scripts/two_wheel_balance/initial_teacher41_case8_canary_cpu_contract_v1.json"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
ROLLOUT_GATE="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
FINALIZER="$ROOT/scripts/two_wheel_balance/summarize_initial_teacher41_case8_canary.py"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"

REVIEWED_CPU_COMMIT="b0fa9621df3b53ae224aa4b7baafb83889fdd8c8"
CPU_CONTRACT_SHA256="18c48f566fe4ed04977f601cebccb0bbea062359695d541aef9eaf65353025f6"
CPU_ADMISSION_SHA256="59b270beb7ef6e9b6a919ef85ec54599465a0ff10a2b5842c1f268326e7ff057"
PLAN_SHA256="f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655"
TEACHER_GATE_SHA256="19506045f9b6ec04cee58efa1b5d2d5600824ce166b1534db05a4895596cf1e0"
POLICY_SHA256="0d796c600c6dca7dce176da555f4cd1f769163f41093d2b6313f4e6264888db7"
PLAYBACK_SHA256="ffe45cd5747f6e628caebafbc405d589f34df764df944ddc2b36a2efd0926b1d"
ROLLOUT_GATE_SHA256="e3e327f2b7bc7f3bdc5f7c27ba36edcf2f3660e5ed4c8b4e4324325764927f5b"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
AUTHORIZATION_SHA256="a9d54710e6be2f84f5ecfc79a00f7f745257781baf14be9cd8db800ba7dcbdb7"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi

sha256() { sha256sum "$1" | awk '{print $1}'; }
identity_matches() { [[ -s "$1" && "$(sha256 "$1")" == "$2" ]]; }

gpu_owners() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true
  )"
  compute_owners="$(
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true
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

wait_gpu_free() {
  local attempts=0
  while ! gpu_owners; do
    attempts=$((attempts + 1))
    (( attempts < 60 )) || return 1
    sleep 2
  done
}

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$HEAD" == "$UPSTREAM" ]] || { printf 'HEAD is not pushed\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is not clean\n' >&2
  exit 3
}
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_CPU_COMMIT" "$HEAD"
identity_matches "$CPU_CONTRACT" "$CPU_CONTRACT_SHA256"
identity_matches "$CPU_ADMISSION" "$CPU_ADMISSION_SHA256"
identity_matches "$PLAN" "$PLAN_SHA256"
identity_matches "$TEACHER_GATE" "$TEACHER_GATE_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$ROLLOUT_GATE" "$ROLLOUT_GATE_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"

python3 - "$CPU_ADMISSION" "$HEAD" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

admission = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
head = sys.argv[2]
source_commit = admission.get("runtime_commit", "")
checks = {
    "cpu_contract_passed": admission.get("passed") is True
    and admission.get("cpu_contract_ready") is True,
    "case8_validation_only": admission.get("case") == 8
    and admission.get("split") == "validation",
    "runtime_closed": admission.get("runtime_authorized") is False
    and admission.get("gpu_launch_authorized") is False
    and admission.get("dynamic_canary_authorized") is False,
    "learning_closed": admission.get("dataset_creation_authorized") is False
    and admission.get("bc_authorized") is False
    and admission.get("ppo_authorized") is False
    and admission.get("holdout_opened") is False,
    "runtime_descends_cpu_admission": bool(source_commit)
    and subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head], check=False
    ).returncode == 0,
}
if not all(checks.values()):
    raise SystemExit(f"case-8 CPU admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_case8_runtime_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 8,
    "split": "validation",
    "zero_then_learned": True,
    "dataset_creation_authorized": False,
    "case78_authorized": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "runtime_started": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_TEACHER41_CASE8_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT" >&2; exit 5; }
wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/zero" "$OUTPUT/learned" "$OUTPUT/logs"
python3 - "$OUTPUT/admission.json" "$HEAD" "$CPU_ADMISSION" "$CPU_CONTRACT" \
  "$PLAN" "$TEACHER_GATE" "$POLICY" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(value):
    path = Path(value)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_case8_runtime_admission_v1",
    "runtime_commit": sys.argv[2],
    "cpu_admission": identity(sys.argv[3]),
    "cpu_contract": identity(sys.argv[4]),
    "plan": identity(sys.argv[5]),
    "teacher_gate": identity(sys.argv[6]),
    "policy": identity(sys.argv[7]),
    "authorization_sha256": sys.argv[8],
    "authorization_consumed_before_isaac": True,
    "case": 8,
    "split": "validation",
    "rollout_order": ["zero", "learned"],
    "residual_action_scales": [0.35, 0.4, 0.1],
    "dataset_creation_authorized": False,
    "case78_authorized": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

COMMON_ARGS=(
  --gains "$GAINS_WIN"
  --plan-dir "$PLAN_WIN"
  --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz'
  --cases "$CASE"
  --controller-wz-kp 1.05
  --maximum-duration-scale 3.0
  --enable-camera-lever-arm-compensation
  --camera-lever-arm-compensation-gain 1.0
  --maximum-camera-lever-arm-correction-m 0.05
  --residual-action-scales 0.35,0.40,0.10
  --runtime-heartbeat-interval-steps 1000
  --headless
)

ZERO_STATUS=0
timeout --signal=TERM --kill-after=30s 600 \
  "$PY" -u -X utf8 "$PLAYBACK_WIN" "${COMMON_ARGS[@]}" \
  --zero-policy-action \
  --runtime-heartbeat "$OUTPUT_WIN\\zero\\runtime_heartbeat.json" \
  --output "$OUTPUT_WIN\\zero\\case_${PADDED}.json" \
  >"$OUTPUT/logs/zero.log" 2>&1 || ZERO_STATUS=$?
printf '%s\n' "$ZERO_STATUS" >"$OUTPUT/logs/zero.exit_code"

if [[ ! -s "$OUTPUT/zero/case_${PADDED}.json" ]] || ! wait_gpu_free; then
  LEARNED_STATUS=125
else
  LEARNED_STATUS=0
  timeout --signal=TERM --kill-after=30s 600 \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" "${COMMON_ARGS[@]}" \
    --residual-policy "$POLICY_WIN" --residual-policy-device cuda \
    --runtime-heartbeat "$OUTPUT_WIN\\learned\\runtime_heartbeat.json" \
    --output "$OUTPUT_WIN\\learned\\case_${PADDED}.json" \
    >"$OUTPUT/logs/learned.log" 2>&1 || LEARNED_STATUS=$?
fi
printf '%s\n' "$LEARNED_STATUS" >"$OUTPUT/logs/learned.exit_code"
wait_gpu_free || true

GATE_STATUS=125
if [[ -s "$OUTPUT/zero/case_${PADDED}.json" && -s "$OUTPUT/learned/case_${PADDED}.json" ]]; then
  GATE_STATUS=0
  python3 "$ROLLOUT_GATE" --mode validation_canary \
    --teacher-dir "$TEACHER_ROOT" --zero-dir "$OUTPUT/zero" \
    --learned-dir "$OUTPUT/learned" --cases "$CASE" --policy "$POLICY" \
    --expected-tracking-profile riser_recovery_direction_v4_camera_lever_arm_v1 \
    --maximum-regression-fraction 0.05 \
    --minimum-zero-improvement-fraction 0.05 \
    --output "$OUTPUT/summary.json" >"$OUTPUT/logs/gate.log" 2>&1 || GATE_STATUS=$?
fi
printf '%s\n' "$GATE_STATUS" >"$OUTPUT/logs/gate.exit_code"

FINAL_STATUS=0
python3 "$FINALIZER" --root "$OUTPUT" --runtime-commit "$HEAD" \
  --zero-exit-code "$ZERO_STATUS" --learned-exit-code "$LEARNED_STATUS" \
  --gate-exit-code "$GATE_STATUS" --output "$OUTPUT/final_status.json" \
  >"$OUTPUT/logs/finalize.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalize.exit_code"
exit "$FINAL_STATUS"
