#!/usr/bin/env python3
"""Audit corrected physical-camera references against the riser kinematics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_reference import (  # noqa: E402
    discover_corrected_riser_stage,
    plan_corrected_riser_reference,
    plan_rate_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=62)
    parser.add_argument("--cases", default="all")
    parser.add_argument(
        "--vertical-shift-mode",
        choices=("none", "per_case_preserve_shape"),
        default="none",
    )
    parser.add_argument("--minimum-camera-height-m", type=float, default=0.6)
    parser.add_argument("--maximum-base-linear-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-base-lateral-velocity", type=float, default=0.02)
    parser.add_argument("--maximum-base-yaw-rate", type=float, default=0.4)
    parser.add_argument("--maximum-riser-rate", type=float, default=1.0)
    parser.add_argument("--maximum-gimbal-rate", type=float, default=0.5)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.02)
    parser.add_argument("--maximum-position-error-p95-m", type=float, default=0.02)
    parser.add_argument("--maximum-attitude-error-deg", type=float, default=2.0)
    parser.add_argument("--maximum-attitude-error-p95-deg", type=float, default=2.0)
    parser.add_argument("--heading-iterations", type=int, default=2)
    parser.add_argument("--orientation-scale-rad", type=float, default=0.01)
    parser.add_argument("--position-scale-m", type=float, default=0.03)
    parser.add_argument("--heading-weight", type=float, default=0.05)
    parser.add_argument(
        "--heading-mode",
        choices=(
            "initial_constant",
            "initial_or_reverse_constant",
            "bidirectional_fixed_point",
            "bounded_attitude_allocation",
            "bounded_unicycle_pose",
        ),
        default="bounded_attitude_allocation",
    )
    parser.add_argument("--save-case-arrays", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_camera_height_m < 0.0:
        raise ValueError("minimum camera height must be non-negative")
    references = discover_corrected_riser_stage(
        args.stage, expected_count=args.expected_count
    )
    if args.cases != "all":
        selected = {int(item.strip()) for item in args.cases.split(",") if item.strip()}
        missing = selected - set(references)
        if missing:
            raise ValueError(f"selected cases are absent from corrected stage: {sorted(missing)}")
        references = {case: references[case] for case in sorted(selected)}
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, reference in references.items():
        shift = 0.0
        if args.vertical_shift_mode == "per_case_preserve_shape":
            shift = max(
                0.0,
                args.minimum_camera_height_m - float(np.min(reference.positions_m[:, 2])),
            )
        plan = plan_corrected_riser_reference(
            reference,
            kinematics,
            vertical_shift_m=shift,
            heading_iterations=args.heading_iterations,
            heading_mode=args.heading_mode,
            orientation_scale_rad=args.orientation_scale_rad,
            heading_weight=args.heading_weight,
            maximum_base_yaw_rate_radps=args.maximum_base_yaw_rate,
            maximum_gimbal_rate_radps=args.maximum_gimbal_rate,
            maximum_linear_velocity_mps=args.maximum_base_linear_velocity,
            maximum_riser_rate_mps=args.maximum_riser_rate,
            position_scale_m=args.position_scale_m,
            attitude_tolerance_rad=math.radians(args.maximum_attitude_error_deg),
        )
        metrics = plan_rate_metrics(plan)
        epsilon = 1e-9
        checks = {
            "position_error_bounded": metrics["position_error_max_m"] <= args.maximum_position_error_m + epsilon,
            "position_error_p95_bounded": metrics["position_error_p95_m"] <= args.maximum_position_error_p95_m + epsilon,
            "attitude_error_bounded": metrics["attitude_error_max_deg"] <= args.maximum_attitude_error_deg + epsilon,
            "attitude_error_p95_bounded": metrics["attitude_error_p95_deg"] <= args.maximum_attitude_error_p95_deg + epsilon,
            "attitude_ik_converged": metrics["attitude_ik_converged_ratio"] == 1.0,
            "base_linear_velocity_bounded": metrics["maximum_abs_base_linear_velocity_mps"] <= args.maximum_base_linear_velocity + epsilon,
            "base_lateral_velocity_bounded": metrics["maximum_abs_base_lateral_velocity_mps"] <= args.maximum_base_lateral_velocity + epsilon,
            "base_yaw_rate_bounded": metrics["maximum_abs_base_yaw_rate_radps"] <= args.maximum_base_yaw_rate + epsilon,
            "riser_rate_bounded": metrics["maximum_abs_riser_rate_mps"] <= args.maximum_riser_rate + epsilon,
            "gimbal_rate_bounded": metrics["maximum_abs_gimbal_rate_radps"] <= args.maximum_gimbal_rate + epsilon,
            "riser_lower_bound": metrics["minimum_riser_position_m"] >= kinematics.riser_lower - 1e-9,
            "riser_upper_bound": metrics["maximum_riser_position_m"] <= kinematics.riser_upper + 1e-9,
        }
        row = {
            "case": case,
            "source_minimum_camera_height_m": float(np.min(reference.positions_m[:, 2])),
            "source_maximum_camera_height_m": float(np.max(reference.positions_m[:, 2])),
            "vertical_shift_m": shift,
            **metrics,
            **checks,
            "passed": all(checks.values()),
        }
        rows.append(row)
        if args.save_case_arrays:
            np.savez_compressed(
                args.output_dir / f"case_{case:04d}.npz",
                time_s=plan.time_s,
                target_position_world_m=plan.targets_m,
                target_attitude_world_dfr_quat_wxyz=reference.semantic_dfr_quat_wxyz,
                base_xy_yaw=plan.base_xy_yaw,
                riser_q=plan.riser_q,
                physical_gimbal_q=plan.gimbal_q,
                achieved_physical_cam_position_world_m=plan.achieved_m,
                physical_cam_attitude_error_rad=plan.attitude_error_rad,
                vertical_shift_m=np.array(shift),
                learned_action_contract=np.array("wheel_effort_2_plus_riser_target_1"),
                physical_gimbal_action_is_learned=np.array(False),
            )

    fieldnames = list(rows[0])
    with (args.output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "cinebotrl_two_wheel_riser_corrected_reference_audit_v1",
        "training_started": False,
        "ppo_authorized": False,
        "source_contract": "corrected semantic DFR targets with physical cam_link observations",
        "learned_action_contract": "wheel_effort_2_plus_riser_target_1",
        "physical_gimbal_action_is_learned": False,
        "heading_mode": args.heading_mode,
        "orientation_scale_rad": args.orientation_scale_rad,
        "position_scale_m": args.position_scale_m,
        "heading_weight": args.heading_weight,
        "vertical_shift_mode": args.vertical_shift_mode,
        "vertical_shift_changes_absolute_target_height": args.vertical_shift_mode != "none",
        "case_count": len(rows),
        "passed_case_count": sum(row["passed"] for row in rows),
        "failed_cases": [row["case"] for row in rows if not row["passed"]],
        "shifted_cases": [row["case"] for row in rows if row["vertical_shift_m"] > 0.0],
        "maximum_vertical_shift_m": max(row["vertical_shift_m"] for row in rows),
        "worst_metrics": {
            key: max(row[key] for row in rows)
            for key in (
                "position_error_max_m",
                "position_error_p95_m",
                "attitude_error_max_deg",
                "attitude_error_p95_deg",
                "maximum_abs_base_linear_velocity_mps",
                "maximum_abs_base_lateral_velocity_mps",
                "maximum_abs_base_yaw_rate_radps",
                "maximum_abs_riser_rate_mps",
                "maximum_abs_gimbal_rate_radps",
            )
        },
        "thresholds": {
            "maximum_position_error_m": args.maximum_position_error_m,
            "maximum_position_error_p95_m": args.maximum_position_error_p95_m,
            "maximum_attitude_error_deg": args.maximum_attitude_error_deg,
            "maximum_attitude_error_p95_deg": args.maximum_attitude_error_p95_deg,
            "maximum_base_linear_velocity_mps": args.maximum_base_linear_velocity,
            "maximum_base_lateral_velocity_mps": args.maximum_base_lateral_velocity,
            "maximum_base_yaw_rate_radps": args.maximum_base_yaw_rate,
            "maximum_riser_rate_mps": args.maximum_riser_rate,
            "maximum_gimbal_rate_radps": args.maximum_gimbal_rate,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed_case_count"] == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
