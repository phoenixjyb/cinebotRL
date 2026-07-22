#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
CASE=78
PADDED=0078
NAMESPACE="20260722_model_based_zero_residual_case78_canary_v1_exclusive"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$NAMESPACE"
PLAN_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
TEACHER="$ROOT/artifacts/two_wheel_riser/20260722_case78_shadow_label_measurement_v1_exclusive/gates/case_${PADDED}.json"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/20260722_model_based_zero_residual_policy_v1_cpu"
POLICY="$POLICY_ROOT/model_based_zero_residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\20260722_model_based_zero_residual_policy_v1_cpu\\model_based_zero_residual_policy.torchscript.pt"
CPU_CONTRACT="$ROOT/artifacts/two_wheel_riser/20260722_model_based_zero_residual_case78_contract_v1_cpu/contract.json"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
FINALIZER="$ROOT/scripts/two_wheel_balance/summarize_model_based_zero_residual_case78_canary.py"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"

REVIEWED_CPU_COMMIT="1f1ea03312a8cfd77014f94ed6e6498915709bfe"
CPU_CONTRACT_SHA256="cc958c9e6cb182509d4cf787891ca9586aefa08b7928f30f358d0ab5f0a4014c"
PLAN_SHA256="28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569"
TEACHER_SHA256="ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459"
POLICY_SHA256="b1494f7af219d44cf966d7ba7781370afc1e8fe9575dd4e414d6ec0b7ea1ab19"
PLAYBACK_SHA256="320019f164343d113bed74c4352686bcb12eb68404bfe911d23594f5f4fc81a3"
FINALIZER_SHA256="636975efc4e7758ffd75864b7fb159cd1c32c49b139ff8e5ec259ad61464eb20"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
AUTHORIZATION_SHA256="ed4f0c0e4336da2afcd1469774467bec624517f27904403678689608fb01b60b"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi
AUTHORIZATION_FILE="${RISER_MODEL_BASED_ZERO_CASE78_AUTHORIZATION_FILE:-}"
if [[ "$MODE" == --execute ]]; then
  [[ -n "$AUTHORIZATION_SHA256" ]] || {
    printf 'model-based zero-residual case-78 runtime authorization is not issued\n' >&2
    exit 4
  }
  [[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
    printf 'model-based zero-residual case-78 one-use token is absent\n' >&2
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
    (( attempts < 90 )) || return 1
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
identity_matches "$PLAN" "$PLAN_SHA256"
identity_matches "$TEACHER" "$TEACHER_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$FINALIZER" "$FINALIZER_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"

python3 - "$CPU_CONTRACT" "$HEAD" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
head = sys.argv[2]
source_commit = contract.get("source_commit", "")
checks = {
    "cpu_contract_ready": contract.get("cpu_contract_ready") is True,
    "case78_validation": contract.get("case") == 78
    and contract.get("split") == "validation",
    "namespace_exact": contract.get("namespace")
    == "20260722_model_based_zero_residual_case78_canary_v1_exclusive",
    "model_based_zero_residual": contract.get("controller_contract", {}).get(
        "policy_command_base"
    )
    == "model_based_planner"
    and contract.get("controller_contract", {}).get("residual_action_scales")
    == [0.05, 0.05, 0.02],
    "camera_cap_exact": contract.get("controller_contract", {}).get(
        "maximum_camera_lever_arm_correction_m"
    )
    == 0.1,
    "runtime_closed": contract.get("runtime_authorization_token_issued") is False
    and contract.get("runtime_authorized") is False
    and contract.get("gpu_launch_authorized") is False
    and contract.get("dynamic_canary_authorized") is False,
    "learning_closed": contract.get("dataset_creation_authorized") is False
    and contract.get("bc_authorized") is False
    and contract.get("ppo_authorized") is False
    and contract.get("holdout_opened") is False,
    "runtime_descends_cpu_contract": bool(source_commit)
    and subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head], check=False
    ).returncode
    == 0,
}
if not all(checks.values()):
    raise SystemExit(f"zero-residual case-78 CPU contract failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  wait_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_case78_preflight_v1",
    "runtime_commit": sys.argv[1],
    "output": sys.argv[2],
    "case": 78,
    "split": "validation",
    "policy_command_base": "model_based_planner",
    "residual_action_scales": [0.05, 0.05, 0.02],
    "camera_lever_arm_cap_m": 0.1,
    "rollout_order": ["explicit_zero", "zero_checkpoint"],
    "maximum_combined_timeout_s": 10800,
    "runtime_authorization_hash_issued": True,
    "runtime_token_consumed": False,
    "dataset_creation_authorized": False,
    "case16_22_32_authorized": False,
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
python3 - "$OUTPUT/admission.json" "$HEAD" "$CPU_CONTRACT" "$PLAN" \
  "$TEACHER" "$POLICY" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(value):
    path = Path(value)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_case78_runtime_admission_v1",
    "runtime_commit": sys.argv[2],
    "cpu_contract": identity(sys.argv[3]),
    "plan": identity(sys.argv[4]),
    "teacher": identity(sys.argv[5]),
    "policy": identity(sys.argv[6]),
    "authorization_sha256": sys.argv[7],
    "authorization_consumed_before_isaac": True,
    "case": 78,
    "split": "validation",
    "rollout_order": ["explicit_zero", "zero_checkpoint"],
    "policy_command_base": "model_based_planner",
    "residual_action_scales": [0.05, 0.05, 0.02],
    "camera_lever_arm_cap_m": 0.1,
    "dataset_creation_authorized": False,
    "case16_22_32_authorized": False,
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
  --maximum-camera-lever-arm-correction-m 0.10
  --policy-command-base model_based_planner
  --residual-action-scales 0.05,0.05,0.02
  --runtime-heartbeat-interval-steps 2000
  --headless
)

EXPLICIT_STATUS=0
timeout --signal=TERM --kill-after=30s 5400 \
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
  timeout --signal=TERM --kill-after=30s 5400 \
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
  --cpu-contract "$CPU_CONTRACT" --policy "$POLICY" \
  --explicit-zero-exit-code "$EXPLICIT_STATUS" \
  --zero-checkpoint-exit-code "$CHECKPOINT_STATUS" \
  --output "$OUTPUT/final_status.json" \
  >"$OUTPUT/logs/finalize.log" 2>&1 || FINAL_STATUS=$?
printf '%s\n' "$FINAL_STATUS" >"$OUTPUT/logs/finalize.exit_code"
exit "$FINAL_STATUS"
