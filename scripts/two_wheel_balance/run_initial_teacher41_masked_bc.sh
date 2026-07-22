#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly WIN_ROOT="G:\\wSpace\\cinebotRL-two-wheel-riser"
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
readonly MODE="${1:---preflight}"
readonly DATASET_STAMP="20260722_initial_teacher41_case78_31_5_5_v2_resealed_cpu"
readonly CONTRACT_STAMP="20260722_initial_teacher41_masked_bc_contract_v1_cpu"
readonly POLICY_STAMP="20260722_initial_teacher41_masked_bc_v1"
readonly DATASET="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP/initial_teacher41_case78_31_5_5_v2_resealed.npz"
readonly SUMMARY="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP/initial_teacher41_case78_31_5_5_v2_resealed.summary.json"
readonly LOADER_AUDIT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP/loader_audit.json"
readonly CONTRACT="$ROOT/artifacts/two_wheel_riser/$CONTRACT_STAMP/contract.json"
readonly TRAINER="$ROOT/scripts/two_wheel_balance/train_riser_residual_bc.py"
readonly POLICY_MODULE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_residual_policy.py"
readonly POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
readonly TRAINER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\train_riser_residual_bc.py"
readonly DATASET_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$DATASET_STAMP\\initial_teacher41_case78_31_5_5_v2_resealed.npz"
readonly POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP"
readonly DATASET_SHA256="03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497"
readonly SUMMARY_SHA256="2b7b177f481fdc632aca2134d9eea69cec66814581a5e39d9c6a099e3d8bcbfb"
readonly LOADER_AUDIT_SHA256="7a764d9cc41e9d43dd808251e2b8466e7ef0940bd356cbebeee38ffcd88e34cb"
readonly CONTRACT_SHA256="41bef3b5b39eb216ecc69cead67b4668424ab79a28a8b8da9df54b58e653dd84"
readonly TRAINER_SHA256="a9c54e6d913d4fde12bc145dbbd512e3de11abe0877dd1a6ab992c154849f74a"
readonly POLICY_MODULE_SHA256="24ac42885bafa9237a4e9b1bfd7d2bef374c374f3b6f4ef87f5d705837c1bee1"
readonly REVIEWED_DATASET_COMMIT="301194698b2d3a7b684afe9e1478763b4891571b"
readonly AUTHORIZATION_SHA256="3198199ccd6b7504c63b9e369aa33565a834281f18c00562cef6da5c75b2ca80"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi
[[ -x "$PY" ]] || { printf 'missing training Python: %s\n' "$PY" >&2; exit 2; }

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
  if [[ -n "$wsl_owners" || -n "$compute_owners" || -n "$windows_owners" ]]; then
    printf 'GPU has an existing playback/training owner\n' >&2
    [[ -z "$wsl_owners" ]] || printf '%s\n' "$wsl_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    [[ -z "$windows_owners" ]] || printf '%s\n' "$windows_owners" >&2
    return 1
  fi
}

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$HEAD" == "$UPSTREAM" ]] || { printf 'HEAD is not pushed\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is not clean\n' >&2
  exit 3
}
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_DATASET_COMMIT" "$HEAD"
identity_matches "$DATASET" "$DATASET_SHA256"
identity_matches "$SUMMARY" "$SUMMARY_SHA256"
identity_matches "$LOADER_AUDIT" "$LOADER_AUDIT_SHA256"
identity_matches "$CONTRACT" "$CONTRACT_SHA256"
identity_matches "$TRAINER" "$TRAINER_SHA256"
identity_matches "$POLICY_MODULE" "$POLICY_MODULE_SHA256"

python3 - "$CONTRACT" "$SUMMARY" "$LOADER_AUDIT" <<'PY'
import json
from pathlib import Path
import sys

contract, summary, loader = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]
]
checks = {
    "contract_ready_not_authorized": contract.get("cpu_contract_ready") is True
    and contract.get("bc_training_authorized") is False
    and contract.get("runtime_authorization_token_issued") is False,
    "masked_architecture": contract.get("architecture_decision", {}).get(
        "policy_architecture"
    ) == "state_shared_lookahead_fusion_previous_action_masked_v1"
    and contract.get("architecture_decision", {}).get(
        "mask_previous_action_observations"
    ) is True,
    "training_contract": contract.get("training_contract", {}).get("epochs_max") == 80
    and contract.get("training_contract", {}).get("patience") == 10
    and contract.get("training_contract", {}).get("batch_size") == 4096
    and contract.get("training_contract", {}).get("seed") == 20260722
    and contract.get("training_contract", {}).get("model_selection_splits")
    == ["validation"],
    "dataset_shape": summary.get("case_count") == 41
    and summary.get("row_count") == 486619
    and summary.get("split_case_counts")
    == {"train": 31, "validation": 5, "holdout": 5},
    "loader": loader.get("passed") is True
    and loader.get("dataset_sha256")
    == "03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497",
    "learning_closed": summary.get("bc_authorized") is False
    and summary.get("ppo_authorized") is False
    and summary.get("training_started") is False,
    "holdout_closed": contract.get("holdout_opened") is False
    and loader.get("holdout_metrics_computed") is False,
}
if not all(checks.values()):
    raise SystemExit(f"teacher-41 masked BC preflight failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  assert_gpu_free || exit 5
  python3 - "$HEAD" "$POLICY_ROOT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_masked_bc_preflight_v1",
    "training_commit": sys.argv[1],
    "policy_output": sys.argv[2],
    "architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
    "masked_observation_indices": [23, 24, 25],
    "validation_only_model_selection": True,
    "holdout_opened": False,
    "runtime_authorization_token_issued": False,
    "training_started": False,
    "learned_rollout_authorized": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_TEACHER41_MASKED_BC_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
  printf 'missing one-use teacher-41 masked BC authorization file\n' >&2
  exit 4
}
[[ ! -L "$AUTHORIZATION_FILE" && "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || {
  printf 'invalid teacher-41 BC authorization file\n' >&2
  exit 4
}
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || {
  printf 'teacher-41 BC authorization hash mismatch\n' >&2
  exit 4
}
[[ ! -e "$POLICY_ROOT" ]] || {
  printf 'refusing to overwrite teacher-41 BC output: %s\n' "$POLICY_ROOT" >&2
  exit 5
}
assert_gpu_free || exit 5

mkdir -p "$POLICY_ROOT"
python3 - "$POLICY_ROOT/admission.json" "$HEAD" "$DATASET" "$SUMMARY" \
  "$LOADER_AUDIT" "$CONTRACT" "$TRAINER" "$POLICY_MODULE" \
  "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_masked_bc_admission_v1",
    "training_commit": sys.argv[2],
    "dataset": identity(sys.argv[3]),
    "dataset_summary": identity(sys.argv[4]),
    "loader_audit": identity(sys.argv[5]),
    "cpu_contract": identity(sys.argv[6]),
    "trainer": identity(sys.argv[7]),
    "policy_module": identity(sys.argv[8]),
    "authorization_sha256": sys.argv[9],
    "authorization_consumed_before_cuda": True,
    "architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
    "masked_observation_indices": [23, 24, 25],
    "epochs_max": 80,
    "patience": 10,
    "batch_size": 4096,
    "seed": 20260722,
    "validation_only_model_selection": True,
    "holdout_opened": False,
    "learned_rollout_authorized": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

STATUS=0
"$PY" -u -X utf8 "$TRAINER_WIN" \
  --dataset "$DATASET_WIN" --output-dir "$POLICY_WIN" \
  --source-commit "$HEAD" --epochs 80 --batch-size 4096 --patience 10 \
  --state-hidden-sizes 128,128 --lookahead-hidden-sizes 64,64 \
  --fusion-hidden-sizes 256,128 --seed 20260722 --device cuda \
  --minimum-improvement-fraction 0.05 --mask-previous-action-observations \
  >"$POLICY_ROOT/train.log" 2>&1 || STATUS=$?
printf '%s\n' "$STATUS" >"$POLICY_ROOT/train.exit_code"

GPU_RELEASED=0
assert_gpu_free && GPU_RELEASED=1
python3 - "$POLICY_ROOT" "$HEAD" "$STATUS" "$DATASET_SHA256" \
  "$GPU_RELEASED" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
status = int(sys.argv[3])
gpu_released = bool(int(sys.argv[5]))
report_path = root / "report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

checks = {
    "trainer_exit_zero": status == 0,
    "source_commit": report.get("source_commit") == sys.argv[2],
    "dataset_identity": report.get("dataset_sha256") == sys.argv[4]
    and report.get("dataset_case_count") == 41
    and report.get("dataset_row_count") == 486619,
    "architecture": report.get("policy_architecture")
    == "state_shared_lookahead_fusion_previous_action_masked_v1",
    "masked_indices": report.get("masked_observation_indices") == [23, 24, 25],
    "masked_contract": report.get("previous_action_observation_contract")
    == "masked_after_normalization_v1",
    "offline_gate": report.get("offline_gate_passed") is True,
    "all_channels_improved": report.get("improvement_checks", {}).get("validation")
    == [True, True, True],
    "validation_only": report.get("offline_gate_splits") == ["validation"],
    "holdout_unopened": report.get("holdout_metrics_computed") is False
    and report.get("holdout_used_for_model_selection") is False,
    "rollout_not_started": report.get("learned_rollout_started") is False,
    "ppo_not_started": report.get("ppo_started") is False,
    "checkpoint": (root / "residual_policy.pt").is_file(),
    "torchscript": (root / "residual_policy.torchscript.pt").is_file(),
    "checkpoint_hash": (root / "residual_policy.pt").is_file()
    and report.get("checkpoint_sha256")
    == hashlib.sha256((root / "residual_policy.pt").read_bytes()).hexdigest(),
    "torchscript_hash": (root / "residual_policy.torchscript.pt").is_file()
    and report.get("torchscript_sha256")
    == hashlib.sha256((root / "residual_policy.torchscript.pt").read_bytes()).hexdigest(),
    "gpu_released": gpu_released,
}
passed = all(checks.values())
payload = {
    "schema": "cinebotrl_two_wheel_riser_initial_teacher41_masked_bc_final_v1",
    "training_commit": sys.argv[2],
    "checks": checks,
    "admission": identity(root / "admission.json"),
    "report": identity(report_path) if report_path.is_file() else None,
    "checkpoint": identity(root / "residual_policy.pt") if (root / "residual_policy.pt").is_file() else None,
    "torchscript": identity(root / "residual_policy.torchscript.pt") if (root / "residual_policy.torchscript.pt").is_file() else None,
    "bc_training_completed": status == 0,
    "offline_gate_passed": report.get("offline_gate_passed") is True,
    "case8_canary_proposal_ready": passed,
    "learned_rollout_authorized": False,
    "learned_rollout_started": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "ppo_started": False,
    "passed": passed,
}
(root / "final_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if not passed:
    raise SystemExit(f"teacher-41 masked BC final gate failed: {checks}")
PY

printf 'teacher-41 masked BC offline gate passed: %s\n' "$POLICY_ROOT"
