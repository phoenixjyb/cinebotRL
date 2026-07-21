#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${RISER_TRAIN_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
DATASET_STAMP="20260721_initial_teacher41_subset_30_5_5_v1"
POLICY_STAMP="20260721_initial_teacher40_bc_v1"
DATASET_ROOT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
DATASET="$DATASET_ROOT/initial_teacher40_30_5_5_v1.npz"
SUMMARY="$DATASET_ROOT/initial_teacher40_30_5_5_v1.summary.json"
FINAL="$DATASET_ROOT/final_status.json"
CORPUS="$DATASET_ROOT/corpus_audit.json"
LOADER_AUDIT="$DATASET_ROOT/loader_audit.json"
TRAINER="$ROOT/scripts/two_wheel_balance/train_riser_residual_bc.py"
TRAINER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\train_riser_residual_bc.py"
DATASET_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$DATASET_STAMP\\initial_teacher40_30_5_5_v1.npz"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP"
DATASET_SHA256="53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44"
SUMMARY_SHA256="815463ffa133addbaec4f09a453fd9dae8e63eb690b37f56fd0a5c1877879542"
FINAL_SHA256="5d0771afb8359ca53d7e33ac6d597e652f17b025ce77a9397405ea00f4e45186"
CORPUS_SHA256="b5b19d4426185610ab66f2a4d755c8b076a0143f3642a1de9ab848a5b35f5308"
LOADER_AUDIT_SHA256="d38bddf8310c18e09eb168072c5449b765470202ab347d5fe4a3705cb12f3914"
REVIEWED_DATASET_LOADER_COMMIT="aec787b3ca5b180c30a9d0b5292dfa0e3f89252c"
AUTHORIZATION_SHA256="c73e9521f1bf6f92b3d2299742f0a02b2667008153a09c8f7bc77802c0838f0e"

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
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_DATASET_LOADER_COMMIT" "$HEAD" || {
  printf 'reviewed v3 dataset loader is not an ancestor of HEAD\n' >&2
  exit 3
}
identity_matches "$DATASET" "$DATASET_SHA256"
identity_matches "$SUMMARY" "$SUMMARY_SHA256"
identity_matches "$FINAL" "$FINAL_SHA256"
identity_matches "$CORPUS" "$CORPUS_SHA256"
identity_matches "$LOADER_AUDIT" "$LOADER_AUDIT_SHA256"

python3 - "$SUMMARY" "$FINAL" "$CORPUS" "$LOADER_AUDIT" <<'PY'
import json
from pathlib import Path
import sys

summary, final, corpus, loader = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]
]
checks = {
    "dataset_schema": summary.get("schema")
    == "cinebotrl_two_wheel_riser_residual_merged_v3",
    "dataset_admitted": summary.get("passed") is True
    and summary.get("dataset_admission_passed") is True
    and summary.get("valid_for_bc_initialization") is True,
    "dataset_shape": summary.get("case_count") == 40
    and summary.get("captured_case_count") == 41
    and summary.get("row_count") == 403569,
    "split_exact": summary.get("split_case_counts")
    == {"train": 30, "validation": 5, "holdout": 5},
    "coverage_only": summary.get("coverage_only_cases") == [77],
    "zero_clip": summary.get("action_clip_ratio") == [0.0, 0.0, 0.0],
    "previous_action": summary.get("previous_action_rebuilt") is True,
    "final_status": final.get("passed") is True
    and final.get("valid_for_bc_initialization") is True,
    "corpus_frozen": corpus.get("passed") is True
    and corpus.get("frozen_action_scales") == [0.35000000000000003, 0.4, 0.1],
    "production_loader": loader.get("passed") is True
    and loader.get("loaded_rows") == 403569,
    "learning_not_prestarted": summary.get("bc_authorized") is False
    and summary.get("ppo_authorized") is False
    and summary.get("training_started") is False,
}
if not all(checks.values()):
    raise SystemExit(f"initial BC dataset admission failed: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  python3 - "$HEAD" "$DATASET" "$POLICY_ROOT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_bc_preflight_v1",
    "training_commit": sys.argv[1],
    "dataset": sys.argv[2],
    "policy_output": sys.argv[3],
    "epochs_max": 80,
    "patience": 10,
    "batch_size": 4096,
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

AUTHORIZATION_FILE="${RISER_INITIAL_BC_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
  printf 'missing one-use BC authorization file\n' >&2
  exit 4
}
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || {
  printf 'BC authorization file must have mode 600\n' >&2
  exit 4
}
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || {
  printf 'BC authorization hash mismatch\n' >&2
  exit 4
}
[[ ! -e "$POLICY_ROOT" ]] || {
  printf 'refusing to overwrite BC output: %s\n' "$POLICY_ROOT" >&2
  exit 5
}

assert_gpu_free || exit 5

mkdir -p "$POLICY_ROOT"
python3 - "$POLICY_ROOT/admission.json" "$HEAD" "$DATASET" "$SUMMARY" \
  "$FINAL" "$CORPUS" "$LOADER_AUDIT" "$TRAINER" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_bc_admission_v1",
    "training_commit": sys.argv[2],
    "dataset": identity(sys.argv[3]),
    "dataset_summary": identity(sys.argv[4]),
    "dataset_final_status": identity(sys.argv[5]),
    "corpus_audit": identity(sys.argv[6]),
    "loader_audit": identity(sys.argv[7]),
    "trainer": identity(sys.argv[8]),
    "authorization_sha256": sys.argv[9],
    "training_contract": {
        "method": "offline_behavior_cloning",
        "epochs_max": 80,
        "patience": 10,
        "batch_size": 4096,
        "seed": 20260721,
        "minimum_improvement_fraction": 0.05,
        "validation_only_model_selection": True,
        "holdout_opened": False,
    },
    "bc_training_authorized": True,
    "learned_rollout_authorized": False,
    "ppo_authorized": False,
    "training_started": False,
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
  --minimum-improvement-fraction 0.05 \
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
    "report_schema": report.get("schema")
    == "cinebotrl_two_wheel_riser_residual_bc_gate_v2",
    "source_commit": report.get("source_commit") == sys.argv[2],
    "dataset_schema": report.get("dataset_schema")
    == "cinebotrl_two_wheel_riser_residual_merged_v3",
    "dataset_identity": report.get("dataset_sha256") == sys.argv[4]
    and report.get("dataset_case_count") == 40
    and report.get("dataset_row_count") == 403569,
    "architecture": report.get("policy_architecture")
    == "state_shared_lookahead_fusion_v1",
    "offline_gate": report.get("offline_gate_passed") is True,
    "all_channels_improved": report.get("improvement_checks", {}).get(
        "validation"
    )
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
    == hashlib.sha256(
        (root / "residual_policy.torchscript.pt").read_bytes()
    ).hexdigest(),
}
passed = all(checks.values())
payload = {
    "schema": "cinebotrl_two_wheel_riser_initial_bc_final_v1",
    "training_commit": sys.argv[2],
    "checks": checks,
    "admission": identity(root / "admission.json"),
    "report": identity(report_path) if report_path.is_file() else None,
    "checkpoint": identity(root / "residual_policy.pt") if (root / "residual_policy.pt").is_file() else None,
    "torchscript": identity(root / "residual_policy.torchscript.pt") if (root / "residual_policy.torchscript.pt").is_file() else None,
    "bc_training_completed": status == 0,
    "offline_gate_passed": report.get("offline_gate_passed") is True,
    "learned_rollout_authorized": passed,
    "learned_rollout_started": False,
    "ppo_authorized": False,
    "ppo_started": False,
    "passed": passed,
}
(root / "final_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if not passed:
    raise SystemExit(f"initial BC final gate failed: {checks}")
PY

printf 'initial BC gate passed: %s\n' "$POLICY_ROOT"
