#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${RISER_TRAIN_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
DATASET_STAMP="20260721_initial_teacher41_subset_30_5_5_v1"
ORIGINAL_STAMP="20260721_initial_teacher40_bc_v1"
MASKED_STAMP="20260721_initial_teacher40_bc_previous_action_masked_v1"
SCALAR_STAMP="20260721_initial_teacher40_bc_previous_action_gain010_v1"
POLICY_STAMP="20260721_initial_teacher40_bc_previous_action_gains010_000_010_v1"
DATASET="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP/initial_teacher40_30_5_5_v1.npz"
ORIGINAL_POLICY="$ROOT/artifacts/two_wheel_riser/$ORIGINAL_STAMP/residual_policy.torchscript.pt"
MASKED_POLICY="$ROOT/artifacts/two_wheel_riser/$MASKED_STAMP/residual_policy.torchscript.pt"
SCALAR_ROOT="$ROOT/artifacts/two_wheel_riser/$SCALAR_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
TRAINER="$ROOT/scripts/two_wheel_balance/train_riser_residual_bc.py"
COMPARATOR="$ROOT/scripts/two_wheel_balance/compare_riser_bc_previous_action.py"
POLICY_MODULE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_residual_policy.py"
DATASET_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$DATASET_STAMP\\initial_teacher40_30_5_5_v1.npz"
ORIGINAL_POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$ORIGINAL_STAMP\\residual_policy.torchscript.pt"
MASKED_POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$MASKED_STAMP\\residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP"
TRAINER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\train_riser_residual_bc.py"
COMPARATOR_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\compare_riser_bc_previous_action.py"
DATASET_SHA256="53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44"
ORIGINAL_POLICY_SHA256="6d86812d3ef63093e00d938e8aa4120146dd30f52dce007af569c3ece989d1dd"
MASKED_POLICY_SHA256="34fa67192f8c66b879eb7d11a83c96ffd2320932e6807f2224cdfa2f74a4c0e4"
SCALAR_FINAL_SHA256="628380b9f9ae8f7a837ccea30906bfb5f9f3518d01124ea6dd4eb92e40f685bf"
SCALAR_COMPARISON_SHA256="f2dfbfb631890a477ab31a705759ee4f27bce6dbd3183771d9b89e7c98ee9949"
TRAINER_SHA256="a9c54e6d913d4fde12bc145dbbd512e3de11abe0877dd1a6ab992c154849f74a"
COMPARATOR_SHA256="deec342e2f1e6ceb9b7ecee817e42c782cf7cf759fb1809e55073388af7acf22"
POLICY_MODULE_SHA256="24ac42885bafa9237a4e9b1bfd7d2bef374c374f3b6f4ef87f5d705837c1bee1"
REVIEWED_COMMIT="ae30ea6ad2fcc476f5c2d420382e9625568fc5d9"
AUTHORIZATION_SHA256="f60a3f278892af1f94e76f8285e4c362e5730edfe3c04747aab48645387110c0"

[[ "$MODE" == --preflight || "$MODE" == --execute ]] || exit 2
[[ -x "$PY" ]] || exit 2
sha256() { sha256sum "$1" | awk '{print $1}'; }
identity_matches() { [[ -s "$1" && "$(sha256 "$1")" == "$2" ]]; }

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true)"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $queryProcessId = $PID
      Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $queryProcessId -and (
          $_.Name -eq "kit.exe" -or
          $_.CommandLine -match "smoke_.*playback|train_riser_residual_bc"
        )
      } | ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$HEAD" == "$(git -C "$ROOT" rev-parse '@{upstream}')" ]] || exit 3
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || exit 3
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_COMMIT" "$HEAD"
identity_matches "$DATASET" "$DATASET_SHA256"
identity_matches "$ORIGINAL_POLICY" "$ORIGINAL_POLICY_SHA256"
identity_matches "$MASKED_POLICY" "$MASKED_POLICY_SHA256"
identity_matches "$SCALAR_ROOT/final_status.json" "$SCALAR_FINAL_SHA256"
identity_matches "$SCALAR_ROOT/case4_previous_action_comparison.json" \
  "$SCALAR_COMPARISON_SHA256"
identity_matches "$TRAINER" "$TRAINER_SHA256"
identity_matches "$COMPARATOR" "$COMPARATOR_SHA256"
identity_matches "$POLICY_MODULE" "$POLICY_MODULE_SHA256"

python3 - "$SCALAR_ROOT/final_status.json" \
  "$SCALAR_ROOT/case4_previous_action_comparison.json" <<'PY'
import json
from pathlib import Path
import sys

final, comparison = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
masked = comparison["results"]["masked"]["recursive_previous_action"]["mse_per_action"]
candidate = comparison["results"]["candidate"]["recursive_previous_action"]["mse_per_action"]
checks = {
    "scalar_rejected": final.get("passed") is False,
    "longitudinal_improved": candidate[0] < masked[0],
    "yaw_degraded": candidate[1] > masked[1],
    "riser_improved": candidate[2] < masked[2],
    "runtime_closed": final.get("learned_rollout_started") is False,
    "holdout_closed": final.get("holdout_opened") is False,
    "ppo_closed": final.get("ppo_started") is False,
}
if not all(checks.values()):
    raise SystemExit(f"channel-selective rationale failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  python3 - "$HEAD" "$POLICY_ROOT" <<'PY'
import json, sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_channel_selective_previous_action_bc_preflight_v1",
    "training_commit": sys.argv[1],
    "policy_output": sys.argv[2],
    "previous_action_observation_gains": [0.10, 0.00, 0.10],
    "validation_only_model_selection": True,
    "holdout_opened": False,
    "rollout_authorized": False,
    "ppo_authorized": False,
    "training_started": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_CHANNEL_GAINS_BC_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$POLICY_ROOT" ]] || exit 5
assert_gpu_free || exit 5

mkdir -p "$POLICY_ROOT"
python3 - "$POLICY_ROOT/admission.json" "$HEAD" "$DATASET" \
  "$SCALAR_ROOT/final_status.json" "$SCALAR_ROOT/case4_previous_action_comparison.json" \
  "$TRAINER" "$COMPARATOR" "$POLICY_MODULE" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_channel_selective_previous_action_bc_admission_v1",
    "training_commit": sys.argv[2],
    "dataset": identity(sys.argv[3]),
    "scalar_reject": identity(sys.argv[4]),
    "scalar_comparison": identity(sys.argv[5]),
    "trainer": identity(sys.argv[6]),
    "comparator": identity(sys.argv[7]),
    "policy_module": identity(sys.argv[8]),
    "authorization_sha256": sys.argv[9],
    "previous_action_observation_gains": [0.10, 0.00, 0.10],
    "validation_only_model_selection": True,
    "holdout_opened": False,
    "learned_rollout_authorized": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n")
PY
rm -f "$AUTHORIZATION_FILE"

TRAIN_STATUS=0
"$PY" -u -X utf8 "$TRAINER_WIN" \
  --dataset "$DATASET_WIN" --output-dir "$POLICY_WIN" --source-commit "$HEAD" \
  --epochs 80 --batch-size 4096 --patience 10 \
  --state-hidden-sizes 128,128 --lookahead-hidden-sizes 64,64 \
  --fusion-hidden-sizes 256,128 --seed 20260721 --device cuda \
  --minimum-improvement-fraction 0.05 \
  --previous-action-observation-gains 0.10,0.00,0.10 \
  >"$POLICY_ROOT/train.log" 2>&1 || TRAIN_STATUS=$?
printf '%s\n' "$TRAIN_STATUS" >"$POLICY_ROOT/train.exit_code"

COMPARE_STATUS=99
if [[ "$TRAIN_STATUS" == 0 && -s "$POLICY_ROOT/residual_policy.torchscript.pt" ]]; then
  COMPARE_STATUS=0
  "$PY" -u -X utf8 "$COMPARATOR_WIN" --dataset "$DATASET_WIN" \
    --original-policy "$ORIGINAL_POLICY_WIN" --masked-policy "$MASKED_POLICY_WIN" \
    --candidate-policy "$POLICY_WIN\\residual_policy.torchscript.pt" \
    --expected-dataset-sha256 "$DATASET_SHA256" \
    --expected-original-sha256 "$ORIGINAL_POLICY_SHA256" \
    --expected-masked-sha256 "$MASKED_POLICY_SHA256" --case 4 \
    --output "$POLICY_WIN\\case4_previous_action_comparison.json" \
    >"$POLICY_ROOT/compare.log" 2>&1 || COMPARE_STATUS=$?
fi
printf '%s\n' "$COMPARE_STATUS" >"$POLICY_ROOT/compare.exit_code"

python3 - "$POLICY_ROOT" "$HEAD" "$TRAIN_STATUS" "$COMPARE_STATUS" \
  "$DATASET_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
train_status, compare_status = map(int, sys.argv[3:5])
report_path = root / "report.json"
comparison_path = root / "case4_previous_action_comparison.json"
report = json.loads(report_path.read_text()) if report_path.is_file() else {}
comparison = json.loads(comparison_path.read_text()) if comparison_path.is_file() else {}

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

checks = {
    "trainer_exit_zero": train_status == 0,
    "comparison_exit_zero": compare_status == 0,
    "source_commit": report.get("source_commit") == sys.argv[2],
    "dataset_identity": report.get("dataset_sha256") == sys.argv[5],
    "architecture": report.get("policy_architecture")
    == "state_shared_lookahead_fusion_previous_action_attenuated_v1",
    "channel_gains": report.get("previous_action_observation_gains")
    == [0.10, 0.00, 0.10],
    "offline_gate": report.get("offline_gate_passed") is True,
    "all_channels_improved": report.get("improvement_checks", {}).get("validation")
    == [True, True, True],
    "case4_comparison": comparison.get("passed") is True,
    "holdout_unopened": report.get("holdout_metrics_computed") is False
    and comparison.get("holdout_opened") is False,
    "runtime_closed": report.get("learned_rollout_started") is False
    and comparison.get("isaac_launched") is False,
    "ppo_closed": report.get("ppo_started") is False
    and comparison.get("ppo_started") is False,
    "checkpoint": (root / "residual_policy.pt").is_file(),
    "torchscript": (root / "residual_policy.torchscript.pt").is_file(),
}
passed = all(checks.values())
(root / "final_status.json").write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_channel_selective_previous_action_bc_final_v1",
    "training_commit": sys.argv[2],
    "checks": checks,
    "admission": identity(root / "admission.json"),
    "report": identity(report_path) if report_path.is_file() else None,
    "comparison": identity(comparison_path) if comparison_path.is_file() else None,
    "checkpoint": identity(root / "residual_policy.pt") if (root / "residual_policy.pt").is_file() else None,
    "torchscript": identity(root / "residual_policy.torchscript.pt") if (root / "residual_policy.torchscript.pt").is_file() else None,
    "learned_rollout_authorized": passed,
    "learned_rollout_started": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "ppo_started": False,
    "passed": passed,
}, indent=2) + "\n")
if not passed:
    raise SystemExit(f"channel-selective BC gate failed: {checks}")
PY

printf 'channel-selective previous-action BC gate passed: %s\n' "$POLICY_ROOT"
