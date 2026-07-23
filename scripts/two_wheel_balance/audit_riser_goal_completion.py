#!/usr/bin/env python3
"""Audit end-goal evidence for the arm-free two-wheel riser project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_model_based_corrective_bc_contract as bc_contract,
)
from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_model_based_corrective_training_dataset as training_dataset_contract,
)


CODE_IDENTITY_KEYS = bc_contract.CODE_IDENTITY_KEYS
validate_bc_execution_admission = bc_contract.validate_bc_execution_admission
validate_bc_execution_report = bc_contract.validate_bc_execution_report
load_training_dataset = training_dataset_contract.load_training_dataset


DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
DEFAULT_GOAL = DOC_ROOT / "riser_recursive_improvement_goal_v1.json"
DEFAULT_ASSET_AUDIT = DOC_ROOT / "evidence_20260715_riser/gate0_asset_audit.json"
DEFAULT_LQR_GATE = DOC_ROOT / "evidence_20260714_28kg/lqr_nominal_gate.json"
DEFAULT_BASELINE = (
    DOC_ROOT / "evidence_20260716_riser_gate0_gate3_online_comp/summary.json"
)
DEFAULT_EXACT_SOURCE = (
    DOC_ROOT / "evidence_20260717_riser_exact_source_gate_a_b/summary.json"
)
DEFAULT_HARDWARE = (
    DOC_ROOT / "evidence_20260723_hardware_production_candidate_v1/summary.json"
)
DEFAULT_BENCH = (
    DOC_ROOT / "evidence_20260723_riser_bench_750w_template_v1/summary.json"
)
EXPECTED_BRANCH = "codex/two-wheel-riser-rl"
EXPECTED_MOVABLE_JOINTS = {
    "joint1_gimbal_pitch",
    "joint2_gimbal_roll",
    "joint3_gimbal_yaw",
    "left_wheel_joint",
    "right_wheel_joint",
    "riser_joint",
}
REQUIRED_COMPLETION_GATES = (
    "isolated_worktree_and_branch",
    "arm_free_robot_asset",
    "frozen_lqr_balance_baseline",
    "riser_height_and_speed_baseline",
    "exact_source_all79_reference",
    "riser_motor_and_mechanism_recommendation",
    "model_based_corrective_training_corpus",
    "projection_aware_bc_policy",
    "learned_policy_all79_dynamic_gate",
    "learned_policy_render_audit",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved)}


def _windows_to_wsl_path(value: str) -> str:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match is None:
        raise ValueError(f"cannot translate Windows path to WSL: {value}")
    drive, suffix = match.groups()
    return f"/mnt/{drive.lower()}/{suffix.replace(chr(92), '/')}"


def _git_value(*args: str) -> str:
    command = ["git", "-C", str(PROJECT_ROOT), *args]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        if os.name != "nt":
            raise
        wsl_root = _windows_to_wsl_path(str(PROJECT_ROOT))
        return subprocess.run(
            ["wsl.exe", "--exec", "git", "-C", wsl_root, *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()


def _git_state() -> dict[str, Any]:
    root = _git_value("rev-parse", "--show-toplevel")
    branch = _git_value("branch", "--show-current")
    head = _git_value("rev-parse", "HEAD")
    upstream = _git_value("rev-parse", "@{upstream}")
    tracked_dirty = bool(_git_value("status", "--porcelain", "--untracked-files=no"))
    return {
        "root": root,
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "tracked_dirty": tracked_dirty,
    }


def _gate(
    *,
    required: bool,
    passed: bool,
    evidence: list[str],
    detail: str,
) -> dict[str, Any]:
    return {
        "required_for_goal": required,
        "passed": bool(passed),
        "evidence": evidence,
        "detail": detail,
    }


def _optional_learning_evidence(
    *,
    training_dataset: Path | None,
    bc_admission: Path | None,
    bc_report: Path | None,
    all79_report: Path | None,
    learned_render_report: Path | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    provided = {
        "training_dataset": training_dataset,
        "bc_admission": bc_admission,
        "bc_report": bc_report,
        "all79_report": all79_report,
        "learned_render_report": learned_render_report,
    }
    identities = {
        name: _identity(path)
        for name, path in provided.items()
        if path is not None and path.is_file()
    }
    training_metadata: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    if any(path is not None for path in (training_dataset, bc_admission, bc_report)):
        if None in (training_dataset, bc_admission, bc_report):
            raise ValueError(
                "training dataset, BC admission, and BC report must be supplied together"
            )
        assert training_dataset is not None
        assert bc_admission is not None
        assert bc_report is not None
        training_metadata, _ = load_training_dataset(training_dataset)
        admission = _load_json(bc_admission)
        report = _load_json(bc_report)
        code_paths = {
            "trainer": PROJECT_ROOT
            / "scripts/two_wheel_balance/train_riser_residual_bc.py",
            "adapter": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_corrective_bc_adapter.py",
            "loss_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/riser_model_based_bc_loss.py",
            "policy_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/riser_residual_policy.py",
            "training_dataset_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_corrective_training_dataset.py",
            "admission_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_corrective_bc_contract.py",
        }
        if set(code_paths) != CODE_IDENTITY_KEYS:
            raise RuntimeError("internal BC code identity set drifted")
        execution_commit = report.get("execution_commit")
        if not isinstance(execution_commit, str):
            raise ValueError("BC report has no execution commit")
        validate_bc_execution_admission(
            admission,
            dataset_path=training_dataset,
            dataset_metadata=training_metadata,
            code_paths=code_paths,
            expected_execution_commit=execution_commit,
            require_authorized=True,
        )
        validate_bc_execution_report(
            report,
            admission_path=bc_admission,
            admission=admission,
            report_directory=bc_report.parent,
        )

    all79: dict[str, Any] | None = None
    if all79_report is not None:
        if report is None:
            raise ValueError("learned all-79 report requires a validated BC report")
        all79 = _load_json(all79_report)

    learned_render: dict[str, Any] | None = None
    if learned_render_report is not None:
        if report is None:
            raise ValueError("learned render report requires a validated BC report")
        learned_render = _load_json(learned_render_report)

    return {
        "training_metadata": training_metadata,
        "bc_report": report,
        "all79_report": all79,
        "learned_render_report": learned_render,
    }, identities


def build_report(
    *,
    goal: Mapping[str, Any],
    asset: Mapping[str, Any],
    lqr: Mapping[str, Any],
    baseline: Mapping[str, Any],
    exact_source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    bench: Mapping[str, Any],
    git_state: Mapping[str, Any],
    learning: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    robot_contract = goal.get("robot_contract", {})
    asset_checks = asset.get("checks", {})
    static = baseline.get("static_height_gate", {})
    dynamic = baseline.get("dynamic_riser_gate", {})
    source = exact_source.get("source_package", {})
    portfolio = exact_source.get("gate_b_portfolio_v4", {})
    recommendation = hardware.get("recommendation", {})
    training_metadata = learning.get("training_metadata")
    bc_report = learning.get("bc_report")
    all79 = learning.get("all79_report")
    learned_render = learning.get("learned_render_report")

    training_cases = (
        training_metadata.get("split_cases", {})
        if isinstance(training_metadata, Mapping)
        else {}
    )
    training_ready = (
        isinstance(training_metadata, Mapping)
        and training_metadata.get("valid_for_projection_aware_bc_input") is True
        and training_metadata.get("valid_for_training") is True
        and len(training_cases.get("train", [])) >= 4
        and len(training_cases.get("validation", [])) >= 2
        and not set(training_cases.get("train", [])).intersection(
            training_cases.get("validation", [])
        )
    )
    bc_ready = (
        isinstance(bc_report, Mapping)
        and bc_report.get("offline_gate_passed") is True
        and bc_report.get("passed") is True
        and bc_report.get("valid_for_dynamic_canary") is True
        and bc_report.get("training_started") is True
        and bc_report.get("ppo_authorized") is False
    )
    policy_sha = (
        bc_report.get("torchscript", {}).get("sha256")
        if isinstance(bc_report, Mapping)
        and isinstance(bc_report.get("torchscript"), Mapping)
        else None
    )
    all79_ready = (
        isinstance(all79, Mapping)
        and all79.get("schema")
        == "cinebotrl_two_wheel_riser_residual_all79_gate_v1"
        and all79.get("cases") == list(range(1, 80))
        and all79.get("case_count") == 79
        and all79.get("policy_sha256") == policy_sha
        and all79.get("passed") is True
        and all79.get("ppo_authorized") is False
    )
    render_videos = (
        learned_render.get("videos", [])
        if isinstance(learned_render, Mapping)
        else []
    )
    render_ready = (
        isinstance(learned_render, Mapping)
        and learned_render.get("schema")
        == "cinebotrl_two_wheel_riser_learned_render_audit_v1"
        and learned_render.get("policy_sha256") == policy_sha
        and learned_render.get("passed") is True
        and isinstance(render_videos, list)
        and len(render_videos) >= 3
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("case"), int)
            and isinstance(item.get("sha256"), str)
            and len(item["sha256"]) == 64
            for item in render_videos
        )
    )

    gates = {
        "isolated_worktree_and_branch": _gate(
            required=True,
            passed=git_state.get("branch") == EXPECTED_BRANCH
            and git_state.get("head") == git_state.get("upstream")
            and git_state.get("tracked_dirty") is False,
            evidence=[
                f"git:{git_state.get('branch', '')}@{git_state.get('head', '')}"
            ],
            detail="dedicated riser branch is clean and synchronized",
        ),
        "arm_free_robot_asset": _gate(
            required=True,
            passed=asset.get("passed") is True
            and asset_checks.get("arm_joints_absent") is True
            and set(asset.get("movable_joint_names", [])) == EXPECTED_MOVABLE_JOINTS
            and asset_checks.get("wheel_track_620mm") is True
            and asset_checks.get("wheel_diameter_8in") is True
            and robot_contract.get("arm_joint_count") == 0,
            evidence=[inputs["asset"]["path"]],
            detail="two wheels, riser, physical gimbal, and no movable arm joints",
        ),
        "frozen_lqr_balance_baseline": _gate(
            required=True,
            passed=lqr.get("schema") == "recomo_two_wheel_lqr_nominal_gate_v1"
            and lqr.get("passed") is True
            and lqr.get("training_started") is False
            and lqr.get("selected", {}).get("success_rate") == 1.0,
            evidence=[inputs["lqr"]["path"]],
            detail="provisional 28 kg simulation LQR baseline passed its nominal gate",
        ),
        "riser_height_and_speed_baseline": _gate(
            required=True,
            passed=baseline.get("passed") is True
            and static.get("passed") is True
            and static.get("camera_height_targets_m") == [0.6, 0.9, 1.8]
            and dynamic.get("passed") is True
            and dynamic.get("requested_speeds_mps") == [0.1, 0.25, 0.5, 1.0]
            and float(dynamic.get("measured_speed_at_1mps_mps", 0.0)) >= 0.95
            and robot_contract.get("camera_height_m") == [0.6, 1.8]
            and robot_contract.get("riser_speed_mps") == 1.0,
            evidence=[inputs["baseline"]["path"]],
            detail="camera range is capped at 1.8 m and the simulated 1 m/s gate passed",
        ),
        "exact_source_all79_reference": _gate(
            required=True,
            passed=source.get("case_count") == 79
            and portfolio.get("exact_source_pass_count") == 79
            and portfolio.get("kinematic_pass_count", 0) >= 40
            and portfolio.get("valid_for_training") is False,
            evidence=[inputs["exact_source"]["path"]],
            detail="79/79 immutable source references exist; planning is not training",
        ),
        "riser_motor_and_mechanism_recommendation": _gate(
            required=True,
            passed=hardware.get("passed") is True
            and hardware.get("candidate_ready_for_supplier_and_bench_review") is True
            and hardware.get("checks", {}).get("camera_height_ceiling_is_1p8m")
            is True
            and hardware.get("checks", {}).get("motor_speed_covers_1mps") is True
            and hardware.get("checks", {}).get(
                "motor_is_pinned_48v_750w_brake_absolute"
            )
            is True
            and float(
                hardware.get("calculated", {}).get(
                    "motor_mechanical_power_from_rating_w", 0.0
                )
            )
            >= 740.0
            and bool(recommendation.get("production_design_review_candidate")),
            evidence=[inputs["hardware"]["path"]],
            detail="750 W servo plus guided belt/telescoping mechanism is design-review ready",
        ),
        "model_based_corrective_training_corpus": _gate(
            required=True,
            passed=training_ready,
            evidence=(
                [inputs["training_dataset"]["path"]]
                if "training_dataset" in inputs
                else []
            ),
            detail="requires at least four train and two disjoint validation cases",
        ),
        "projection_aware_bc_policy": _gate(
            required=True,
            passed=bc_ready,
            evidence=[inputs["bc_report"]["path"]] if "bc_report" in inputs else [],
            detail="requires an authorized real BC run and passing offline report",
        ),
        "learned_policy_all79_dynamic_gate": _gate(
            required=True,
            passed=all79_ready,
            evidence=(
                [inputs["all79_report"]["path"]]
                if "all79_report" in inputs
                else []
            ),
            detail="requires one hash-bound learned-policy result for every case 1-79",
        ),
        "learned_policy_render_audit": _gate(
            required=True,
            passed=render_ready,
            evidence=(
                [inputs["learned_render_report"]["path"]]
                if "learned_render_report" in inputs
                else []
            ),
            detail="requires at least three audited learned-policy rollout videos",
        ),
        "physical_riser_bench_qualification": _gate(
            required=False,
            passed=bench.get("passed") is True
            and bench.get("ready_for_production_design_review") is True,
            evidence=[inputs["bench"]["path"]],
            detail="deployment qualification remains separate from the requested recommendation",
        ),
    }
    missing = [
        name
        for name in REQUIRED_COMPLETION_GATES
        if not gates[name]["passed"]
    ]
    return {
        "schema": "cinebotrl_two_wheel_riser_goal_completion_audit_v1",
        "objective": goal.get("objective"),
        "required_completion_gates": list(REQUIRED_COMPLETION_GATES),
        "gates": gates,
        "required_gate_pass_count": len(REQUIRED_COMPLETION_GATES) - len(missing),
        "required_gate_count": len(REQUIRED_COMPLETION_GATES),
        "completion_blockers": missing,
        "goal_achieved": not missing,
        "obstacle_avoidance_in_scope": False,
        "runtime_started": False,
        "bc_started_by_audit": False,
        "ppo_started_by_audit": False,
        "git": {
            "branch": git_state.get("branch"),
            "head": git_state.get("head"),
            "upstream": git_state.get("upstream"),
            "tracked_dirty": git_state.get("tracked_dirty"),
        },
        "inputs": dict(inputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--asset-audit", type=Path, default=DEFAULT_ASSET_AUDIT)
    parser.add_argument("--lqr-gate", type=Path, default=DEFAULT_LQR_GATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--exact-source", type=Path, default=DEFAULT_EXACT_SOURCE)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--bc-admission", type=Path)
    parser.add_argument("--bc-report", type=Path)
    parser.add_argument("--all79-report", type=Path)
    parser.add_argument("--learned-render-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    fixed_paths = {
        "goal": args.goal,
        "asset": args.asset_audit,
        "lqr": args.lqr_gate,
        "baseline": args.baseline,
        "exact_source": args.exact_source,
        "hardware": args.hardware,
        "bench": args.bench,
    }
    inputs = {name: _identity(path) for name, path in fixed_paths.items()}
    learning, optional_identities = _optional_learning_evidence(
        training_dataset=args.training_dataset,
        bc_admission=args.bc_admission,
        bc_report=args.bc_report,
        all79_report=args.all79_report,
        learned_render_report=args.learned_render_report,
    )
    inputs.update(optional_identities)
    report = build_report(
        goal=_load_json(args.goal),
        asset=_load_json(args.asset_audit),
        lqr=_load_json(args.lqr_gate),
        baseline=_load_json(args.baseline),
        exact_source=_load_json(args.exact_source),
        hardware=_load_json(args.hardware),
        bench=_load_json(args.bench),
        git_state=_git_state(),
        learning=learning,
        inputs=inputs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["goal_achieved"] or args.allow_incomplete:
        return 0
    return 8


if __name__ == "__main__":
    raise SystemExit(main())
