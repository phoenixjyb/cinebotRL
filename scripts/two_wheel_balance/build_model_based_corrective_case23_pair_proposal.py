#!/usr/bin/env python3
"""Build a closed CPU-only paired-canary proposal for corrective case 23."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (  # noqa: E402
    CORRECTIVE_TARGET_ADMISSION_CONTRACT,
    load_corrective_teacher_profile,
)


SCHEMA = "cinebotrl_two_wheel_riser_model_based_corrective_case23_pair_proposal_v1"
SELECTION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_pair_tranche_selection_v1"
)
PORTFOLIO_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_export_v1"
CASE = 23
HOLDOUT_CASES = [3, 5, 13, 19, 24]
VALIDATION_CASES = [8, 16, 22, 32, 78]
RESIDUAL_SCALES = [0.05, 0.05, 0.02]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _closed(payload: Mapping[str, object]) -> bool:
    return all(
        payload.get(name) is False
        for name in (
            "runtime_authorized",
            "gpu_launch_authorized",
            "label_capture_authorized",
            "dataset_merge_authorized",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
            "valid_for_training",
        )
        if name in payload
    )


def build_proposal(
    selection: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    profile_path: Path,
    *,
    selection_path: Path,
    portfolio_path: Path,
) -> dict[str, Any]:
    selected_rows = {
        int(row["case"]): row for row in selection.get("selected_rows", [])
    }
    portfolio_items = {
        int(item["case"]): item for item in portfolio.get("items", [])
    }
    selected = selected_rows.get(CASE, {})
    item = portfolio_items.get(CASE, {})
    profile_case, config, profile_identity = load_corrective_teacher_profile(
        profile_path, expected_case=CASE
    )
    plan_path = portfolio_path.parent / str(item.get("file", ""))
    checks = {
        "selection_schema": selection.get("schema") == SELECTION_SCHEMA,
        "selection_passed_closed": selection.get("passed") is True
        and _closed(selection),
        "selection_binds_portfolio": selection.get("identities", {})
        .get("portfolio", {})
        .get("sha256")
        == sha256_file(portfolio_path),
        "case_selected_for_pair": CASE in selection.get("selected_cases", [])
        and selected.get("selection_role") == "same_seed_paired_canary_required",
        "case_not_validation_or_holdout": CASE
        not in selection.get("validation_cases", [])
        and CASE not in selection.get("holdout_cases", [])
        and selection.get("validation_cases") == VALIDATION_CASES
        and selection.get("holdout_cases") == HOLDOUT_CASES,
        "case30_profile_reuse_closed": selection.get(
            "case30_profile_reuse_authorized"
        )
        is False,
        "portfolio_schema": portfolio.get("schema") == PORTFOLIO_SCHEMA,
        "plan_admitted": item.get("case") == CASE
        and item.get("passed") is True
        and item.get("timing_transition_kinematic_gate_passed") is True,
        "plan_hash_matches_selection": item.get("plan_sha256")
        == selected.get("plan_sha256"),
        "plan_file_hash": plan_path.is_file()
        and sha256_file(plan_path) == item.get("plan_sha256"),
        "case_specific_profile": profile_case == CASE,
        "profile_margin": config.maximum_residuals == (0.045, 0.045, 0.018)
        and all(
            value < scale
            for value, scale in zip(
                config.maximum_residuals, RESIDUAL_SCALES, strict=True
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"case23 pair proposal contract failed: {checks}")

    source_duration = float(item["source_duration_s"])
    execution_duration = float(item["execution_duration_s"])
    return {
        "schema": SCHEMA,
        "case": CASE,
        "split": "train",
        "checks": checks,
        "identities": {
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
            },
            "portfolio": {
                "path": str(portfolio_path.resolve()),
                "sha256": sha256_file(portfolio_path),
            },
            "plan": {
                "path": str(plan_path.resolve()),
                "sha256": sha256_file(plan_path),
            },
            "corrective_profile": profile_identity,
        },
        "source_duration_s": source_duration,
        "execution_duration_s": execution_duration,
        "controller_arguments": {
            "configuration_seed": 20260716,
            "reset_seed": 20260739,
            "controller_wz_kp": 1.05,
            "maximum_duration_scale": 3.0,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.05,
            "policy_command_base": "model_based_planner",
            "zero_policy_action": True,
            "residual_action_scales": RESIDUAL_SCALES,
        },
        "proposed_perturbation": {
            "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
            "case": CASE,
            "start_phase_time_s": execution_duration / 2.0,
            "duration_steps": 20,
            "force_body_x_n": 20.0,
            "application_height_m": 0.5,
        },
        "paired_admission_contract": CORRECTIVE_TARGET_ADMISSION_CONTRACT,
        "paired_experiment": {
            "baseline": "complete_model_based_planner_plus_exact_zero_residual",
            "candidate": "complete_model_based_planner_plus_corrective_teacher",
            "rollout_order": ["baseline", "candidate"],
            "same_plan_seed_physics_and_perturbation_required": True,
            "candidate_requires_baseline_dynamic_pass": True,
            "minimum_position_p95_improvement_m": 0.003,
            "minimum_position_p95_relative_improvement": 0.02,
            "position_max_regression_allowance_m": 0.005,
            "attitude_max_regression_allowance_deg": 0.10,
            "pitch_regression_allowance_deg": 0.50,
            "riser_error_regression_allowance_m": 0.002,
            "saturation_regression_allowed": False,
        },
        "runtime_route_implemented": False,
        "authorization_token_issued": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_created": False,
        "dataset_merge_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--corrective-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proposal: {args.output}")
    result = build_proposal(
        load_json(args.selection),
        load_json(args.portfolio),
        args.corrective_profile,
        selection_path=args.selection,
        portfolio_path=args.portfolio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
