#!/usr/bin/env python3
"""Validate a learned all-79 rollout admission before Isaac starts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_model_based_learned_all79_contract as contract,
)


SCHEMA = "cinebotrl_two_wheel_riser_model_based_learned_all79_preflight_v1"


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _git(*args: str) -> str:
    wsl_root = os.environ.get("RISER_GIT_ROOT_WSL")
    command = (
        ["wsl.exe", "git", "-C", wsl_root, *args]
        if wsl_root
        else ["git", "-C", str(PROJECT_ROOT), *args]
    )
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity(path: Path) -> dict[str, str]:
    try:
        display = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": contract.sha256_file(path)}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "admission": _resolve(args.admission),
        "bc_report": _resolve(args.bc_report),
        "policy": _resolve(args.policy),
        "plan_manifest": _resolve(args.plan_manifest),
        "source_manifest": _resolve(args.source_manifest),
        "lqr_gains": _resolve(args.lqr_gains),
        "robot_build_audit": _resolve(args.robot_build_audit),
        "robot_usd": _resolve(args.robot_usd),
        "drive_profile_selection": _resolve(args.drive_profile_selection),
        "validation_gate_report": _resolve(args.validation_gate_report),
        "holdout_gate_report": _resolve(args.holdout_gate_report),
    }
    admission = _load(paths["admission"])
    bc_report = _load(paths["bc_report"])
    validation_report = _load(paths["validation_gate_report"])
    holdout_report = _load(paths["holdout_gate_report"])
    execution_commit = bc_report.get("execution_commit")
    if not isinstance(execution_commit, str):
        raise ValueError("BC report has no execution commit")
    code_paths = {
        "playback": PROJECT_ROOT
        / "scripts/two_wheel_balance/smoke_riser_reference_playback.py",
        "rollout_gate": PROJECT_ROOT
        / "scripts/two_wheel_balance/gate_riser_residual_rollouts.py",
        "completion_auditor": PROJECT_ROOT
        / "scripts/two_wheel_balance/audit_riser_goal_completion.py",
        "admission_contract": SRC_ROOT
        / "rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_learned_all79_contract.py",
        "policy_artifact": SRC_ROOT
        / "rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_policy_artifact.py",
        "preflight_validator": Path(__file__).resolve(),
        "execution_wrapper": PROJECT_ROOT
        / "scripts/two_wheel_balance/"
        "run_model_based_learned_all79_policy_gate.sh",
    }
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    tracked_clean = _git("status", "--porcelain", "--untracked-files=no") == ""
    contract_error = None
    try:
        contract.validate_learned_all79_admission(
            admission,
            identity_root=PROJECT_ROOT,
            bc_report_path=paths["bc_report"],
            bc_report=bc_report,
            policy_path=paths["policy"],
            plan_manifest_path=paths["plan_manifest"],
            source_manifest_path=paths["source_manifest"],
            lqr_gains_path=paths["lqr_gains"],
            robot_build_audit_path=paths["robot_build_audit"],
            robot_usd_path=paths["robot_usd"],
            drive_profile_selection_path=paths["drive_profile_selection"],
            validation_report_path=paths["validation_gate_report"],
            validation_report=validation_report,
            holdout_report_path=paths["holdout_gate_report"],
            holdout_report=holdout_report,
            code_paths=code_paths,
            expected_execution_commit=execution_commit,
            require_authorized=args.require_authorized,
        )
    except ValueError as error:
        contract_error = str(error)
    checks = {
        "contract": contract_error is None,
        "head_matches_upstream": head == upstream,
        "head_matches_execution_commit": head == execution_commit,
        "tracked_worktree_clean": tracked_clean,
    }
    return {
        "schema": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "error": contract_error,
        "head": head,
        "upstream": upstream,
        "execution_commit": execution_commit,
        "require_authorized": args.require_authorized,
        "learned_rollout_authorized": (
            admission.get("learned_rollout_authorized") is True
        ),
        "all79_evaluation_approved": (
            admission.get("all79_evaluation_approved") is True
        ),
        "inputs": {name: _identity(path) for name, path in paths.items()},
        "code": {name: _identity(path) for name, path in code_paths.items()},
        "runtime_started": False,
        "dataset_created": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--bc-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--lqr-gains", type=Path, required=True)
    parser.add_argument("--robot-build-audit", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--drive-profile-selection", type=Path, required=True)
    parser.add_argument("--validation-gate-report", type=Path, required=True)
    parser.add_argument("--holdout-gate-report", type=Path, required=True)
    parser.add_argument("--require-authorized", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        result = {
            "schema": SCHEMA,
            "passed": False,
            "error": str(error),
            "runtime_started": False,
            "dataset_created": False,
            "residual_capture_started": False,
            "bc_started": False,
            "ppo_started": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
