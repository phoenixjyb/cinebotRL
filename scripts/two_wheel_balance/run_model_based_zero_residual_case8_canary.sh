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
NAMESPACE="20260722_model_based_zero_residual_case8_canary_v1_exclusive"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$NAMESPACE"
PLAN_STAMP="20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
TEACHER="$ROOT/artifacts/two_wheel_riser/20260718_gate_c_smoothed_case8_dynamic_retime_v1_exclusive/gates/case_${PADDED}.json"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/20260722_model_based_zero_residual_policy_v1_cpu"
POLICY="$POLICY_ROOT/model_based_zero_residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\20260722_model_based_zero_residual_policy_v1_cpu\\model_based_zero_residual_policy.torchscript.pt"
CPU_ADMISSION="$ROOT/artifacts/two_wheel_riser/20260722_model_based_zero_residual_case8_contract_v1_cpu/admission.json"
CPU_CONTRACT="$ROOT/scripts/two_wheel_balance/model_based_zero_residual_case8_cpu_contract_v1.json"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
FINALIZER="$ROOT/scripts/two_wheel_balance/summarize_model_based_zero_residual_case8_canary.py"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"

REVIEWED_CPU_COMMIT="28709a2c1abaccb3a459fa4fa46c9239f04502ef"
CPU_CONTRACT_SHA256="47d4de888b0eeba0dbebad066f7ef106f3042fe9d63bf174f551dfddfbc1185b"
CPU_ADMISSION_SHA256="001a27c14305d3c04f502a203da7f8474c451a842f11b3f0848d1b8da2f4a0de"
PLAN_SHA256="f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655"
TEACHER_SHA256="19506045f9b6ec04cee58efa1b5d2d5600824ce166b1534db05a4895596cf1e0"
POLICY_SHA256="b1494f7af219d44cf966d7ba7781370afc1e8fe9575dd4e414d6ec0b7ea1ab19"
PLAYBACK_SHA256="320019f164343d113bed74c4352686bcb12eb68404bfe911d23594f5f4fc81a3"
FINALIZER_SHA256="bf17e18934a72c32c0a49b005728e1666058dc7ede9e6d28b0801efee233623e"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
AUTHORIZATION_SHA256="af3cf7748bafb522acaa2827553d49a939771e135c61c367c42d492ecc5a96c0"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi
AUTHORIZATION_FILE="${RISER_MODEL_BASED_ZERO_CASE8_AUTHORIZATION_FILE:-}"
if [[ "$MODE" == --execute ]]; then
  [[ -n "$AUTHORIZATION_SHA256" ]] || {
    printf 'model-based zero-residual case-8 runtime authorization is not issued\n' >&2
    exit 4
  }
  [[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
    printf 'model-based zero-residual case-8 one-use token is absent\n' >&2
    exit 4
  }
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
identity_matches "$TEACHER" "$TEACHER_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$FINALIZER" "$FINALIZER_SHA256"
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
    "namespace_exact": admission.get("namespace")
    == "20260722_model_based_zero_residual_case8_canary_v1_exclusive",
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
    ).returncode
    == 0,
}
if not all(checks.values()):
    raise SystemExit(f"zero-residual case-8 CPU admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 8,
    "split": "validation",
    "policy_command_base": "model_based_planner",
    "residual_action_scales": [0.05, 0.05, 0.02],
    "rollout_order": ["explicit_zero", "zero_checkpoint"],
    "runtime_authorization_hash_issued": True,
    "runtime_token_consumed": False,
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

[[ ! -L "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT" >&2; exit 5; }
wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/explicit_zero" "$OUTPUT/zero_checkpoint" "$OUTPUT/logs"
python3 - "$OUTPUT/admission.json" "$HEAD" "$CPU_ADMISSION" "$CPU_CONTRACT" \
  "$PLAN" "$TEACHER" "$POLICY" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(value):
    path = Path(value)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_runtime_admission_v1",
    "runtime_commit": sys.argv[2],
    "cpu_admission": identity(sys.argv[3]),
    "cpu_contract": identity(sys.argv[4]),
    "plan": identity(sys.argv[5]),
    "teacher": identity(sys.argv[6]),
    "policy": identity(sys.argv[7]),
    "authorization_sha256": sys.argv[8],
    "authorization_consumed_before_isaac": True,
    "case": 8,
    "split": "validation",
    "rollout_order": ["explicit_zero", "zero_checkpoint"],
    "policy_command_base": "model_based_planner",
    "residual_action_scales": [0.05, 0.05, 0.02],
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
  --policy-command-base model_based_planner
  --residual-action-scales 0.05,0.05,0.02
  --runtime-heartbeat-interval-steps 1000
  --headless
)

EXPLICIT_STATUS=0
timeout --signal=TERM --kill-after=30s 600 \
  "$PY" -u -X utf8 "$PLAYBACK_WIN" "${COMMON_ARGS[@]}" \
  --zero-policy-action \
  --runtime-heartbeat "$OUTPUT_WIN\\explicit_zero\\runtime_heartbeat.json" \
  --output "$OUTPUT_WIN\\explicit_zero\\case_${PADDED}.json" \
  >"$OUTPUT/logs/explicit_zero.log" 2>&1 || EXPLICIT_STATUS=$?
printf '%s\n' "$EXPLICIT_STATUS" >"$OUTPUT/logs/explicit_zero.exit_code"

if [[ ! -s "$OUTPUT/explicit_zero/case_${PADDED}.json" ]] || ! wait_gpu_free; then
  CHECKPOINT_STATUS=125
else
  CHECKPOINT_STATUS=0
  timeout --signal=TERM --kill-after=30s 600 \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" "${COMMON_ARGS[@]}" \
    --residual-policy "$POLICY_WIN" --residual-policy-device cuda \
    --runtime-heartbeat "$OUTPUT_WIN\\zero_checkpoint\\runtime_heartbeat.json" \
    --output "$OUTPUT_WIN\\zero_checkpoint\\case_${PADDED}.json" \
    >"$OUTPUT/logs/zero_checkpoint.log" 2>&1 || CHECKPOINT_STATUS=$?
fi
printf '%s\n' "$CHECKPOINT_STATUS" >"$OUTPUT/logs/zero_checkpoint.exit_code"
wait_gpu_free || true

FINAL_STATUS=0
PYTHONPATH="$ROOT" python3 "$FINALIZER" --root "$OUTPUT" \
  --cpu-admission "$CPU_ADMISSION" --policy "$POLICY" \
  --explicit-zero-exit-code "$EXPLICIT_STATUS" \
  --zero-checkpoint-exit-code "$CHECKPOINT_STATUS" \
  --output "$OUTPUT/final_status.json" \
  >"$OUTPUT/logs/finalize.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalize.exit_code"
exit "$FINAL_STATUS"
