#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${RISER_TRAIN_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
DATASET_STAMP="${RISER_DATASET_STAMP:-20260717_residual_all79_exact_source_v1}"
POLICY_STAMP="${RISER_POLICY_STAMP:-20260717_residual_bc_exact_source_v1}"
DATASET_ROOT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP"
DATASET_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$DATASET_STAMP\\all79_residual_dataset_v1.npz"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP"
SUMMARY="$DATASET_ROOT/summary.json"
DATASET="$DATASET_ROOT/all79_residual_dataset_v1.npz"
TRAINER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\train_riser_residual_bc.py"

[[ -x "$PY" ]] || { printf 'missing training Python: %s\n' "$PY" >&2; exit 2; }
[[ -s "$SUMMARY" && -s "$DATASET" ]] || {
  printf 'all-79 dataset gate is incomplete: %s\n' "$DATASET_ROOT" >&2
  exit 2
}

DATASET_COMMIT="$({ python3 - "$SUMMARY" "$DATASET" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
dataset = summary.get("dataset", {})
plan_manifest = Path(summary.get("plan_manifest", ""))
capture_admission = Path(summary.get("capture_admission", ""))
source_manifest = Path(summary.get("exact_source_manifest", ""))
source_audit = Path(summary.get("exact_source_admission", ""))
source_commit = summary.get("git_commit")
admission = (
    json.loads(capture_admission.read_text(encoding="utf-8"))
    if capture_admission.is_file()
    else {}
)
exact_source_audit = (
    json.loads(source_audit.read_text(encoding="utf-8"))
    if source_audit.is_file()
    else {}
)
checks = {
    "schema": summary.get("schema")
    == "cinebotrl_two_wheel_riser_all79_dynamic_dataset_gate_v1",
    "passed": summary.get("passed") is True,
    "case_count": summary.get("case_count") == 79,
    "passed_case_count": summary.get("passed_case_count") == 79,
    "tracking_profile": summary.get("tracking_profiles")
    == ["riser_phase_consistent_v2"],
    "phase_governor": summary.get("phase_governor_enabled") is True,
    "source_commit": isinstance(source_commit, str) and len(source_commit) == 40,
    "plan_manifest_hash": plan_manifest.is_file()
    and summary.get("plan_manifest_sha256")
    == hashlib.sha256(plan_manifest.read_bytes()).hexdigest(),
    "capture_admission_hash": capture_admission.is_file()
    and summary.get("capture_admission_sha256")
    == hashlib.sha256(capture_admission.read_bytes()).hexdigest(),
    "capture_admission_commit": admission.get("git_commit") == source_commit,
    "capture_admission_plan": admission.get("plan_manifest_sha256")
    == summary.get("plan_manifest_sha256"),
    "capture_admission_cases": admission.get("cases") == list(range(1, 80)),
    "exact_source_contract": summary.get("trajectory_integrity_contract")
    == "exact_source_v1",
    "exact_source_training_qualified": summary.get(
        "upstream_valid_for_training"
    )
    is True,
    "exact_source_manifest_hash": source_manifest.is_file()
    and summary.get("exact_source_manifest_sha256")
    == hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    "exact_source_audit_hash": source_audit.is_file()
    and summary.get("exact_source_admission_sha256")
    == hashlib.sha256(source_audit.read_bytes()).hexdigest(),
    "exact_source_audit_passed": exact_source_audit.get("passed") is True
    and exact_source_audit.get("training_authorized") is True,
    "capture_exact_source_hash": admission.get("exact_source_manifest_sha256")
    == summary.get("exact_source_manifest_sha256"),
    "capture_exact_source_admission_hash": admission.get(
        "exact_source_admission_sha256"
    )
    == summary.get("exact_source_admission_sha256"),
    "dataset_case_count": dataset.get("case_count") == 79,
    "dataset_no_leakage": dataset.get("trajectory_leakage") is False,
    "dataset_finite": dataset.get("finite_values") is True,
    "dataset_zero_clip": dataset.get("action_clip_ratio") == [0.0, 0.0, 0.0],
    "dataset_reconstruction": dataset.get(
        "teacher_command_reconstruction_max_error", float("inf")
    )
    <= 2e-6,
    "dataset_no_source_actions": dataset.get("source_action_labels_used") is False,
    "dataset_no_physical_gimbal_actions": dataset.get(
        "physical_gimbal_labels_used_as_actions"
    ) is False,
    "dataset_hash": dataset.get("dataset_sha256")
    == hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
}
if not all(checks.values()):
    raise SystemExit(f"BC admission rejected: {checks}")
print(source_commit)
PY
})"
git -C "$ROOT" cat-file -e "$DATASET_COMMIT^{commit}"
git -C "$ROOT" merge-base --is-ancestor "$DATASET_COMMIT" HEAD || {
  printf 'dataset commit is not an ancestor of current HEAD: %s\n' "$DATASET_COMMIT" >&2
  exit 2
}
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make policy provenance ambiguous\n' >&2
  exit 2
}
POLICY_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
printf 'BC dataset admission passed: source commit %s\n' "$DATASET_COMMIT"

if [[ -e "$POLICY_ROOT" ]]; then
  printf 'refusing to overwrite policy output: %s\n' "$POLICY_ROOT" >&2
  exit 3
fi
mkdir -p "$POLICY_ROOT"
"$PY" -u -X utf8 "$TRAINER_WIN" \
  --dataset "$DATASET_WIN" \
  --output-dir "$POLICY_WIN" \
  --source-commit "$POLICY_COMMIT" \
  "$@" | tee "$POLICY_ROOT/train.log"

python3 - "$POLICY_ROOT/report.json" "$POLICY_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "source_commit": report.get("source_commit") == sys.argv[2],
    "offline_gate": report.get("offline_gate_passed") is True,
    "rollout_authorized": report.get("learned_rollout_authorized") is True,
    "ppo_not_started": report.get("ppo_started") is False,
    "rollout_not_started": report.get("learned_rollout_started") is False,
    "no_leakage": report.get("source_group_leakage") is False,
    "validation_only_selection": report.get("offline_gate_splits")
    == ["validation"],
    "holdout_reserved": report.get("holdout_used_for_model_selection") is False,
    "holdout_unopened": report.get("holdout_metrics_computed") is False,
    "case_balanced_training": report.get("case_balanced_training_loss") is True,
    "case_balanced_validation": report.get("case_balanced_validation_gate") is True,
    "checkpoint": isinstance(report.get("checkpoint_sha256"), str),
    "torchscript": isinstance(report.get("torchscript_sha256"), str),
}
if not all(checks.values()):
    raise SystemExit(f"offline BC gate rejected: {checks}")
print(json.dumps(checks, sort_keys=True))
PY

printf 'offline BC gate passed: %s\n' "$POLICY_ROOT"
