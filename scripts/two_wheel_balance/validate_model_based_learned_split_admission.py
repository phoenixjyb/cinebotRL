#!/usr/bin/env python3
"""Validate learned validation/holdout admission without starting Isaac."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_split_contract as contract,
)


CODE_IDENTITY_KEYS = contract.CODE_IDENTITY_KEYS
SPLIT_MODES = contract.SPLIT_MODES
admission_identity = contract.admission_identity
validate_learned_split_admission = contract.validate_learned_split_admission


ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = {
    "playback": ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py",
    "rollout_gate": ROOT / "scripts/two_wheel_balance/gate_riser_residual_rollouts.py",
    "admission_contract": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_learned_split_contract.py"
    ),
    "policy_artifact": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_policy_artifact.py"
    ),
    "preflight_validator": Path(__file__).resolve(),
    "execution_wrapper": (
        ROOT
        / "scripts/two_wheel_balance/"
        "run_model_based_learned_split_policy_gate.sh"
    ),
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate(args: argparse.Namespace) -> dict:
    admission = _load(args.admission)
    bc_report = _load(args.bc_report)
    prior_validation = (
        None
        if args.prior_validation_gate_report is None
        else _load(args.prior_validation_gate_report)
    )
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    tracked_clean = not _git("status", "--short", "--untracked-files=no")
    validate_learned_split_admission(
        admission,
        identity_root=ROOT,
        mode=args.mode,
        bc_report_path=args.bc_report.resolve(),
        bc_report=bc_report,
        policy_path=args.policy.resolve(),
        plan_manifest_path=args.plan_manifest.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        lqr_gains_path=args.lqr_gains.resolve(),
        robot_build_audit_path=args.robot_build_audit.resolve(),
        robot_usd_path=args.robot_usd.resolve(),
        drive_profile_selection_path=args.drive_profile_selection.resolve(),
        prior_validation_report_path=(
            None
            if args.prior_validation_gate_report is None
            else args.prior_validation_gate_report.resolve()
        ),
        prior_validation_report=prior_validation,
        code_paths=CODE_PATHS,
        expected_execution_commit=head,
        require_authorized=args.require_authorized,
    )
    checks = {
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "head_matches_bc_execution_commit": head
        == bc_report.get("execution_commit"),
        "code_identity_complete": set(CODE_PATHS) == CODE_IDENTITY_KEYS,
        "runtime_authorized": admission.get("split_evaluation_approved") is True
        and admission.get("learned_rollout_authorized") is True,
        "capture_closed": admission.get("residual_capture_authorized") is False,
        "bc_closed": admission.get("bc_authorized") is False,
        "ppo_closed": admission.get("ppo_authorized") is False,
        "training_closed": admission.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"learned split preflight failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_model_based_learned_split_preflight_v1",
        "mode": args.mode,
        "cases": admission["cases"],
        "execution_commit": head,
        "admission": admission_identity(args.admission),
        "bc_report": admission_identity(args.bc_report),
        "policy": admission_identity(args.policy),
        "plan_manifest": admission_identity(args.plan_manifest),
        "source_manifest": admission_identity(args.source_manifest),
        "prior_validation_gate_report": (
            None
            if args.prior_validation_gate_report is None
            else admission_identity(args.prior_validation_gate_report)
        ),
        "checks": checks,
        "runtime_started": False,
        "dataset_written": False,
        "capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(SPLIT_MODES), required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--bc-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--lqr-gains", type=Path, required=True)
    parser.add_argument("--robot-build-audit", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--drive-profile-selection", type=Path, required=True)
    parser.add_argument("--prior-validation-gate-report", type=Path)
    parser.add_argument("--require-authorized", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
