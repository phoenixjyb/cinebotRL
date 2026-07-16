#!/usr/bin/env python3
"""Audit full corrected poses through the riser RS4 proxy decomposition."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    parse_acquisition_time_scale_overrides,
)
from rl_platform.tasks.two_wheel_balance.riser_reference import (  # noqa: E402
    discover_corrected_riser_stage,
    plan_rate_metrics,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_reference import (  # noqa: E402
    plan_rs4_riser_reference,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_attitude import (  # noqa: E402
    proxy_joint_rates_rad_s,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=62)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--time-scale-overrides", default="")
    parser.add_argument("--acquisition-time-scale-overrides", default="")
    parser.add_argument(
        "--vertical-shift-mode",
        choices=("none", "per_case_preserve_shape"),
        default="per_case_preserve_shape",
    )
    parser.add_argument("--minimum-camera-height-m", type=float, default=0.6)
    parser.add_argument("--maximum-position-error-p95-m", type=float, default=0.15)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.25)
    parser.add_argument("--maximum-attitude-error-p95-deg", type=float, default=5.0)
    parser.add_argument("--maximum-attitude-error-deg", type=float, default=10.0)
    parser.add_argument("--maximum-base-linear-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-base-lateral-velocity", type=float, default=0.02)
    parser.add_argument("--maximum-base-yaw-rate", type=float, default=0.25)
    parser.add_argument("--maximum-riser-rate", type=float, default=1.0)
    parser.add_argument("--maximum-proxy-joint-rate", type=float, default=0.4188790205)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = discover_corrected_riser_stage(
        args.stage, expected_count=args.expected_count
    )
    if args.cases.strip().lower() != "all":
        selected = {
            int(item.strip()) for item in args.cases.split(",") if item.strip()
        }
        missing = selected - set(references)
        if not selected or missing:
            raise ValueError(f"invalid selected cases: {sorted(selected)}")
        references = {case: references[case] for case in sorted(selected)}
    time_scale_overrides = parse_acquisition_time_scale_overrides(
        args.time_scale_overrides
    )
    acquisition_scale_overrides = parse_acquisition_time_scale_overrides(
        args.acquisition_time_scale_overrides
    )
    overlap = set(time_scale_overrides) & set(acquisition_scale_overrides)
    if overlap:
        raise ValueError(f"cases cannot use both retiming modes: {sorted(overlap)}")
    unselected_overrides = (
        set(time_scale_overrides) | set(acquisition_scale_overrides)
    ) - set(references)
    if unselected_overrides:
        raise ValueError(
            f"time-scale overrides target unselected cases: {sorted(unselected_overrides)}"
        )
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, reference in references.items():
        time_scale = time_scale_overrides.get(case, 1.0)
        acquisition_scale = acquisition_scale_overrides.get(case, 1.0)
        if time_scale != 1.0:
            reference = replace(reference, time_s=reference.time_s * time_scale)
        elif acquisition_scale != 1.0:
            acquisition_end = int(reference.metadata["acquisition_end_index"])
            time_s = reference.time_s.copy()
            acquisition_duration = float(time_s[acquisition_end])
            time_s[: acquisition_end + 1] *= acquisition_scale
            time_s[acquisition_end + 1 :] += (
                acquisition_scale - 1.0
            ) * acquisition_duration
            reference = replace(reference, time_s=time_s)
        shift = 0.0
        if args.vertical_shift_mode == "per_case_preserve_shape":
            shift = max(
                0.0,
                args.minimum_camera_height_m
                - float(np.min(reference.positions_m[:, 2])),
            )
        plan = plan_rs4_riser_reference(
            reference,
            kinematics,
            vertical_shift_m=shift,
            maximum_base_yaw_rate_rad_s=args.maximum_base_yaw_rate,
            maximum_base_linear_velocity_mps=args.maximum_base_linear_velocity,
            maximum_riser_rate_mps=args.maximum_riser_rate,
        )
        metrics = plan_rate_metrics(plan)
        proxy_rate = proxy_joint_rates_rad_s(plan.gimbal_q, plan.time_s)
        raw_proxy_rate = np.diff(plan.gimbal_q, axis=0) / np.diff(plan.time_s)[:, None]
        metrics["maximum_abs_gimbal_rate_radps"] = float(
            np.max(np.abs(proxy_rate))
        )
        metrics["maximum_abs_raw_proxy_target_rate_radps"] = float(
            np.max(np.abs(raw_proxy_rate))
        )
        metrics["maximum_abs_raw_proxy_target_step_rad"] = float(
            np.max(np.abs(np.diff(plan.gimbal_q, axis=0)))
        )
        epsilon = 1e-9
        checks = {
            "position_p95_bounded": metrics["position_error_p95_m"] <= args.maximum_position_error_p95_m + epsilon,
            "position_max_bounded": metrics["position_error_max_m"] <= args.maximum_position_error_m + epsilon,
            "attitude_p95_bounded": metrics["attitude_error_p95_deg"] <= args.maximum_attitude_error_p95_deg + epsilon,
            "attitude_max_bounded": metrics["attitude_error_max_deg"] <= args.maximum_attitude_error_deg + epsilon,
            "attitude_adapter_converged": metrics["attitude_ik_converged_ratio"] == 1.0,
            "base_linear_velocity_bounded": metrics["maximum_abs_base_linear_velocity_mps"] <= args.maximum_base_linear_velocity + epsilon,
            "base_lateral_velocity_bounded": metrics["maximum_abs_base_lateral_velocity_mps"] <= args.maximum_base_lateral_velocity + epsilon,
            "base_yaw_rate_bounded": metrics["maximum_abs_base_yaw_rate_radps"] <= args.maximum_base_yaw_rate + epsilon,
            "riser_rate_bounded": metrics["maximum_abs_riser_rate_mps"] <= args.maximum_riser_rate + epsilon,
            "proxy_joint_rate_bounded": metrics["maximum_abs_gimbal_rate_radps"] <= args.maximum_proxy_joint_rate + epsilon,
            "raw_proxy_target_rate_bounded": metrics["maximum_abs_raw_proxy_target_rate_radps"] <= args.maximum_proxy_joint_rate + epsilon,
            "riser_lower_bound": metrics["minimum_riser_position_m"] >= kinematics.riser_lower - epsilon,
            "riser_upper_bound": metrics["maximum_riser_position_m"] <= kinematics.riser_upper + epsilon,
        }
        rows.append(
            {
                "case": case,
                "planning_strategy": plan.planning_strategy,
                "time_scale": time_scale,
                "acquisition_time_scale": acquisition_scale,
                "retimed_duration_s": float(reference.time_s[-1]),
                "vertical_shift_m": shift,
                **metrics,
                **checks,
                "passed": all(checks.values()),
            }
        )

    with (args.output_dir / "cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "cinebotrl_two_wheel_riser_rs4_pose_audit_v1",
        "training_started": False,
        "ppo_authorized": False,
        "case_count": len(rows),
        "time_scale_overrides": {
            str(case): scale for case, scale in time_scale_overrides.items()
        },
        "acquisition_time_scale_overrides": {
            str(case): scale for case, scale in acquisition_scale_overrides.items()
        },
        "retimed_cases": [
            row["case"]
            for row in rows
            if row["time_scale"] != 1.0 or row["acquisition_time_scale"] != 1.0
        ],
        "passed_case_count": sum(row["passed"] for row in rows),
        "failed_cases": [row["case"] for row in rows if not row["passed"]],
        "strategy_counts": {
            strategy: sum(row["planning_strategy"] == strategy for row in rows)
            for strategy in sorted({row["planning_strategy"] for row in rows})
        },
        "shifted_cases": [row["case"] for row in rows if row["vertical_shift_m"] > 0.0],
        "maximum_vertical_shift_m": max(row["vertical_shift_m"] for row in rows),
        "failure_counts": {
            key: sum(not row[key] for row in rows)
            for key in checks
        },
        "worst_metrics": {
            key: max(row[key] for row in rows)
            for key in (
                "position_error_p95_m",
                "position_error_max_m",
                "attitude_error_p95_deg",
                "attitude_error_max_deg",
                "maximum_abs_base_linear_velocity_mps",
                "maximum_abs_base_lateral_velocity_mps",
                "maximum_abs_base_yaw_rate_radps",
                "maximum_abs_riser_rate_mps",
                "maximum_abs_gimbal_rate_radps",
                "maximum_abs_raw_proxy_target_rate_radps",
                "maximum_abs_raw_proxy_target_step_rad",
            )
        },
        "vertical_shift_changes_absolute_target_height": args.vertical_shift_mode != "none",
        "status": "pure_kinematic_gate_only_pending_isaac_dynamics",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed_case_count"] == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
