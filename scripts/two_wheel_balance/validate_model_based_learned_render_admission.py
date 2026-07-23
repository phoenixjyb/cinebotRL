#!/usr/bin/env python3
"""Validate learned-render admission without starting Isaac."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_render_contract as contract,
)


ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "scripts/two_wheel_balance/audit_riser_goal_completion.py"
CODE_PATHS = {
    "playback": ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py",
    "admission_contract": Path(contract.__file__).resolve(),
    "policy_artifact": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_policy_artifact.py"
    ),
    "preflight_validator": Path(__file__).resolve(),
    "execution_wrapper": (
        ROOT / "scripts/two_wheel_balance/run_model_based_learned_render_gate.sh"
    ),
    "media_auditor": (
        ROOT / "scripts/two_wheel_balance/audit_model_based_learned_render_media.py"
    ),
    "report_finalizer": (
        ROOT / "scripts/two_wheel_balance/finalize_model_based_learned_render.py"
    ),
    "completion_auditor": AUDITOR,
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_auditor():
    spec = importlib.util.spec_from_file_location("riser_goal_auditor", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate(args: argparse.Namespace) -> dict:
    admission = _load(args.admission)
    all79 = _load(args.all79_report)
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    clean = not _git("status", "--short", "--untracked-files=no")
    contract.validate_render_admission(
        admission,
        identity_root=ROOT,
        all79_report_path=args.all79_report.resolve(),
        all79_report=all79,
        all79_admission_path=args.all79_admission.resolve(),
        all79_preflight_path=args.all79_preflight.resolve(),
        policy_path=args.policy.resolve(),
        plan_manifest_path=args.plan_manifest.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        lqr_gains_path=args.lqr_gains.resolve(),
        robot_build_audit_path=args.robot_build_audit.resolve(),
        robot_usd_path=args.robot_usd.resolve(),
        drive_profile_selection_path=args.drive_profile_selection.resolve(),
        code_paths=CODE_PATHS,
        expected_execution_commit=head,
        require_authorized=args.require_authorized,
    )
    auditor = _load_auditor()
    auditor._validate_all79_report(
        all79,
        policy_sha256=contract.sha256_file(args.policy),
        report_directory=args.all79_report.parent,
        admission_path=args.all79_admission,
        preflight_path=args.all79_preflight,
        plan_manifest_path=args.plan_manifest,
        execution_commit=head,
    )
    checks = {
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": clean,
        "all79_execution_commit_matches_head": all79.get("execution_commit") == head,
        "render_authorized": admission.get("render_evaluation_approved") is True
        and admission.get("learned_render_authorized") is True,
        "capture_closed": admission.get("residual_capture_authorized") is False,
        "bc_closed": admission.get("bc_authorized") is False,
        "ppo_closed": admission.get("ppo_authorized") is False,
        "training_closed": admission.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"learned render preflight failed: {checks}")
    identity = contract.sha256_file
    return {
        "schema": "cinebotrl_two_wheel_riser_model_based_learned_render_preflight_v1",
        "execution_commit": head,
        "cases": contract.REPRESENTATIVE_CASES,
        "admission": {
            "path": str(args.admission.resolve()),
            "sha256": identity(args.admission),
        },
        "all79_report": {
            "path": str(args.all79_report.resolve()),
            "sha256": identity(args.all79_report),
        },
        "policy": {
            "path": str(args.policy.resolve()),
            "sha256": identity(args.policy),
        },
        "plan_manifest": {
            "path": str(args.plan_manifest.resolve()),
            "sha256": identity(args.plan_manifest),
        },
        "checks": checks,
        "runtime_started": False,
        "recording_started": False,
        "capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "admission",
        "all79_report",
        "all79_admission",
        "all79_preflight",
        "policy",
        "plan_manifest",
        "source_manifest",
        "lqr_gains",
        "robot_build_audit",
        "robot_usd",
        "drive_profile_selection",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
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
