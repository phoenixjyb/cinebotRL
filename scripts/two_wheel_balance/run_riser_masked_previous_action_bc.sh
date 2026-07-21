#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${RISER_TRAIN_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
DATASET_STAMP="20260721_initial_teacher41_subset_30_5_5_v1"
DIAGNOSIS_STAMP="20260721_initial_teacher40_bc_case4_cpu_diagnosis_v1"
POLICY_STAMP="20260721_initial_teacher40_bc_previous_action_masked_v1"
DATASET="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP/initial_teacher40_30_5_5_v1.npz"
DIAGNOSIS="$ROOT/artifacts/two_wheel_riser/$DIAGNOSIS_STAMP/report.json"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
TRAINER="$ROOT/scripts/two_wheel_balance/train_riser_residual_bc.py"
POLICY_MODULE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_residual_policy.py"
TRAINER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\train_riser_residual_bc.py"
DATASET_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$DATASET_STAMP\\initial_teacher40_30_5_5_v1.npz"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP"
DATASET_SHA256="53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44"
DIAGNOSIS_SHA256="9c81451eeb4549da1504f4ae6baa141aa67aa415ababb481c9b37e202917766c"
TRAINER_SHA256="05f0041bf126133ba87b82adb32f2b069474398745720266e2653f1c6a319b73"
POLICY_MODULE_SHA256="4a75d4d1c0c1f41a11b3702a32b57a695dcb754ff3bcd4b4b0a8d6080b8d5546"
DIAGNOSIS_COMMIT="6f5940c2fb19bfcfad1472a6bfb4bbf17db2cf8d"
AUTHORIZATION_SHA256="1a89756cdfeed525d0e412be9a642edc659920b403fea809afd79e83f77bfe46"

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
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$HEAD" == "$UPSTREAM" ]] || { printf 'HEAD is not pushed\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is not clean\n' >&2
  exit 3
}
git -C "$ROOT" merge-base --is-ancestor "$DIAGNOSIS_COMMIT" "$HEAD"
identity_matches "$DATASET" "$DATASET_SHA256"
identity_matches "$DIAGNOSIS" "$DIAGNOSIS_SHA256"
identity_matches "$TRAINER" "$TRAINER_SHA256"
identity_matches "$POLICY_MODULE" "$POLICY_MODULE_SHA256"

python3 - "$DIAGNOSIS" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "case": report.get("case") == 4 and report.get("split") == "validation",
    "diagnosis": report.get("diagnosis", {}).get("classification")
    == "autoregressive_previous_action_exposure_bias",
    "fit": report.get("diagnosis", {}).get("teacher_state_fit_passed") is True,
    "recursive_failure": report.get("diagnosis", {}).get(
        "recursive_previous_action_stability_passed"
    ) is False,
    "runtime_closed": report.get("isaac_launched") is False,
    "holdout_closed": report.get("holdout_opened") is False,
    "ppo_closed": report.get("ppo_started") is False,
    "passed": report.get("passed") is True,
}
if not all(checks.values()):
    raise SystemExit(f"masked BC diagnosis admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  python3 - "$HEAD" "$POLICY_ROOT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_masked_previous_action_bc_preflight_v1",
    "training_commit": sys.argv[1],
    "policy_output": sys.argv[2],
    "masked_observation_indices": [23, 24, 25],
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

AUTHORIZATION_FILE="${RISER_MASKED_BC_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
  printf 'missing one-use masked BC authorization file\n' >&2
  exit 4
}
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$POLICY_ROOT" ]] || {
  printf 'refusing to overwrite masked BC output: %s\n' "$POLICY_ROOT" >&2
  exit 5
}
assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$POLICY_ROOT"
python3 - "$POLICY_ROOT/admission.json" "$HEAD" "$DATASET" "$DIAGNOSIS" \
  "$TRAINER" "$POLICY_MODULE" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_masked_previous_action_bc_admission_v1",
    "training_commit": sys.argv[2],
    "dataset": identity(sys.argv[3]),
    "diagnosis": identity(sys.argv[4]),
    "trainer": identity(sys.argv[5]),
    "policy_module": identity(sys.argv[6]),
    "authorization_sha256": sys.argv[7],
    "masked_observation_indices": [23, 24, 25],
    "epochs_max": 80,
    "patience": 10,
    "batch_size": 4096,
    "seed": 20260721,
    "validation_only_model_selection": True,
    "holdout_opened": False,
    "rollout_authorized": False,
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
  --fusion-hidden-sizes 256,128 --seed 20260721 --device cuda \
  --minimum-improvement-fraction 0.05 --mask-previous-action-observations \
  >"$POLICY_ROOT/train.log" 2>&1 || STATUS=$?
printf '%s\n' "$STATUS" >"$POLICY_ROOT/train.exit_code"

python3 - "$POLICY_ROOT" "$HEAD" "$STATUS" "$DATASET_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
status = int(sys.argv[3])
report_path = root / "report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
checks = {
    "trainer_exit_zero": status == 0,
    "source_commit": report.get("source_commit") == sys.argv[2],
    "dataset_identity": report.get("dataset_sha256") == sys.argv[4],
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
}
passed = all(checks.values())
(root / "final_status.json").write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_masked_previous_action_bc_final_v1",
    "training_commit": sys.argv[2],
    "checks": checks,
    "admission": identity(root / "admission.json"),
    "report": identity(report_path) if report_path.is_file() else None,
    "checkpoint": (
        identity(root / "residual_policy.pt")
        if (root / "residual_policy.pt").is_file()
        else None
    ),
    "torchscript": (
        identity(root / "residual_policy.torchscript.pt")
        if (root / "residual_policy.torchscript.pt").is_file()
        else None
    ),
    "learned_rollout_authorized": passed,
    "learned_rollout_started": False,
    "holdout_opened": False,
    "ppo_authorized": False,
    "ppo_started": False,
    "passed": passed,
}, indent=2) + "\n", encoding="utf-8")
if not passed:
    raise SystemExit(f"masked BC final gate failed: {checks}")
PY

printf 'masked previous-action BC gate passed: %s\n' "$POLICY_ROOT"
