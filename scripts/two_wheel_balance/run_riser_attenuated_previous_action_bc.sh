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
SCHEDULED_STAMP="20260721_initial_teacher40_bc_scheduled_previous_action_v1"
POLICY_STAMP="20260721_initial_teacher40_bc_previous_action_gain010_v1"
DATASET="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP/initial_teacher40_30_5_5_v1.npz"
ORIGINAL_ROOT="$ROOT/artifacts/two_wheel_riser/$ORIGINAL_STAMP"
MASKED_ROOT="$ROOT/artifacts/two_wheel_riser/$MASKED_STAMP"
SCHEDULED_ROOT="$ROOT/artifacts/two_wheel_riser/$SCHEDULED_STAMP"
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
ORIGINAL_REPORT_SHA256="44437ed005aa69718c244fda8c8fddb58ebf95c308c9387d11d85e4ff62ce104"
ORIGINAL_POLICY_SHA256="6d86812d3ef63093e00d938e8aa4120146dd30f52dce007af569c3ece989d1dd"
MASKED_REPORT_SHA256="3f0efb4a2707b343a775dd5dd8b0ad49d6506474da627d8449ca81556cbbcd3e"
MASKED_POLICY_SHA256="34fa67192f8c66b879eb7d11a83c96ffd2320932e6807f2224cdfa2f74a4c0e4"
SCHEDULED_FINAL_SHA256="abab8f8726dc0ac545ae3f45f147e3e10e590ef507682ebe5197d0d80b5f447e"
SCHEDULED_COMPARISON_SHA256="1173da5c72ca3fe0d2f75059a0571af6333b6f1d8bf36ed93f4229f8d1571009"
TRAINER_SHA256="e76cefc1f248bec3f7098ad3f22f3de4922096a13f50ff3af805e15bb50e932a"
COMPARATOR_SHA256="deec342e2f1e6ceb9b7ecee817e42c782cf7cf759fb1809e55073388af7acf22"
POLICY_MODULE_SHA256="b07de795b20a8bac6e1a6aace51f0b5ed34abf7999d1002782d4d790ab9cddfb"
REVIEWED_ATTENUATION_COMMIT="5401fa3f31ea7416595b87bf4541b5f685a7007a"
AUTHORIZATION_SHA256="17f206486c02a1a2f44e91e63b62d7fac97ef212b7ed19143e338834c47092d9"

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
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
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
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_ATTENUATION_COMMIT" "$HEAD"
identity_matches "$DATASET" "$DATASET_SHA256"
identity_matches "$ORIGINAL_ROOT/report.json" "$ORIGINAL_REPORT_SHA256"
identity_matches "$ORIGINAL_ROOT/residual_policy.torchscript.pt" "$ORIGINAL_POLICY_SHA256"
identity_matches "$MASKED_ROOT/report.json" "$MASKED_REPORT_SHA256"
identity_matches "$MASKED_ROOT/residual_policy.torchscript.pt" "$MASKED_POLICY_SHA256"
identity_matches "$SCHEDULED_ROOT/final_status.json" "$SCHEDULED_FINAL_SHA256"
identity_matches "$SCHEDULED_ROOT/case4_previous_action_comparison.json" \
  "$SCHEDULED_COMPARISON_SHA256"
identity_matches "$TRAINER" "$TRAINER_SHA256"
identity_matches "$COMPARATOR" "$COMPARATOR_SHA256"
identity_matches "$POLICY_MODULE" "$POLICY_MODULE_SHA256"

python3 - "$SCHEDULED_ROOT/final_status.json" \
  "$SCHEDULED_ROOT/case4_previous_action_comparison.json" <<'PY'
import json
from pathlib import Path
import sys

final, comparison = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
checks = {
    "scheduled_rejected": final.get("passed") is False
    and final.get("checks", {}).get("trainer_exit_zero") is True
    and final.get("checks", {}).get("case4_comparison") is False,
    "specific_recursive_failure": comparison.get("passed") is False
    and comparison.get("checks", {}).get("recursive_aggregate_beats_original") is True
    and comparison.get("checks", {}).get(
        "recursive_aggregate_beats_masked_by_one_percent"
    ) is False,
    "runtime_closed": final.get("checks", {}).get("runtime_closed") is True,
    "holdout_closed": final.get("holdout_opened") is False,
    "ppo_closed": final.get("ppo_started") is False,
}
if not all(checks.values()):
    raise SystemExit(f"attenuated BC rationale admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  python3 - "$HEAD" "$POLICY_ROOT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_attenuated_previous_action_bc_preflight_v1",
    "training_commit": sys.argv[1],
    "policy_output": sys.argv[2],
    "previous_action_observation_gain": 0.10,
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

AUTHORIZATION_FILE="${RISER_ATTENUATED_BC_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || exit 4
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$POLICY_ROOT" ]] || exit 5
assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$POLICY_ROOT"
python3 - "$POLICY_ROOT/admission.json" "$HEAD" "$DATASET" \
  "$SCHEDULED_ROOT/final_status.json" "$SCHEDULED_ROOT/case4_previous_action_comparison.json" \
  "$TRAINER" "$COMPARATOR" "$POLICY_MODULE" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_attenuated_previous_action_bc_admission_v1",
    "training_commit": sys.argv[2],
    "dataset": identity(sys.argv[3]),
    "scheduled_reject": identity(sys.argv[4]),
    "scheduled_comparison": identity(sys.argv[5]),
    "trainer": identity(sys.argv[6]),
    "comparator": identity(sys.argv[7]),
    "policy_module": identity(sys.argv[8]),
    "authorization_sha256": sys.argv[9],
    "training_contract": {
        "method": "offline_behavior_cloning",
        "previous_action_observation_gain": 0.10,
        "epochs_max": 80,
        "patience": 10,
        "batch_size": 4096,
        "seed": 20260721,
        "validation_only_model_selection": True,
        "holdout_opened": False,
    },
    "bc_training_authorized": True,
    "learned_rollout_authorized": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

TRAIN_STATUS=0
"$PY" -u -X utf8 "$TRAINER_WIN" \
  --dataset "$DATASET_WIN" --output-dir "$POLICY_WIN" \
  --source-commit "$HEAD" --epochs 80 --batch-size 4096 --patience 10 \
  --state-hidden-sizes 128,128 --lookahead-hidden-sizes 64,64 \
  --fusion-hidden-sizes 256,128 --seed 20260721 --device cuda \
  --minimum-improvement-fraction 0.05 --previous-action-observation-gain 0.10 \
  >"$POLICY_ROOT/train.log" 2>&1 || TRAIN_STATUS=$?
printf '%s\n' "$TRAIN_STATUS" >"$POLICY_ROOT/train.exit_code"

COMPARE_STATUS=99
if [[ "$TRAIN_STATUS" == 0 && -s "$POLICY_ROOT/residual_policy.torchscript.pt" ]]; then
  COMPARE_STATUS=0
  "$PY" -u -X utf8 "$COMPARATOR_WIN" \
    --dataset "$DATASET_WIN" --original-policy "$ORIGINAL_POLICY_WIN" \
    --masked-policy "$MASKED_POLICY_WIN" \
    --candidate-policy "$POLICY_WIN\\residual_policy.torchscript.pt" \
    --expected-dataset-sha256 "$DATASET_SHA256" \
    --expected-original-sha256 "$ORIGINAL_POLICY_SHA256" \
    --expected-masked-sha256 "$MASKED_POLICY_SHA256" \
    --case 4 --output "$POLICY_WIN\\case4_previous_action_comparison.json" \
    >"$POLICY_ROOT/compare.log" 2>&1 || COMPARE_STATUS=$?
fi
printf '%s\n' "$COMPARE_STATUS" >"$POLICY_ROOT/compare.exit_code"

python3 - "$POLICY_ROOT" "$HEAD" "$TRAIN_STATUS" "$COMPARE_STATUS" \
  "$DATASET_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
train_status, compare_status = map(int, sys.argv[3:5])
report_path = root / "report.json"
comparison_path = root / "case4_previous_action_comparison.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {}

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
    "attenuation_contract": report.get("previous_action_observation_gain") == 0.10
    and report.get("previous_action_observation_contract")
    == "attenuated_after_normalization_v1",
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
    "schema": "cinebotrl_two_wheel_riser_attenuated_previous_action_bc_final_v1",
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
}, indent=2) + "\n", encoding="utf-8")
if not passed:
    raise SystemExit(f"attenuated previous-action BC gate failed: {checks}")
PY

printf 'attenuated previous-action BC gate passed: %s\n' "$POLICY_ROOT"
