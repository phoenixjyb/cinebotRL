#!/usr/bin/env python3
"""Audit corrected riser references against the RS4 attitude command surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_reference import (  # noqa: E402
    discover_corrected_riser_stage,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_attitude import (  # noqa: E402
    RS4_FILMING_RATE_LIMIT_DEG_S,
    RS4_HARD_RATE_LIMIT_DEG_S,
    bounded_path_yaw_schedule,
    fit_corpus_centered_body_basis,
    plan_rs4_attitude_commands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=62)
    parser.add_argument("--maximum-base-yaw-rate", type=float, default=0.4)
    parser.add_argument(
        "--filming-rate-limit-deg-s",
        type=float,
        default=RS4_FILMING_RATE_LIMIT_DEG_S,
    )
    parser.add_argument(
        "--hard-rate-limit-deg-s", type=float, default=RS4_HARD_RATE_LIMIT_DEG_S
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = discover_corrected_riser_stage(
        args.stage, expected_count=args.expected_count
    )
    yaw = {
        case: bounded_path_yaw_schedule(
            reference, maximum_yaw_rate_rad_s=args.maximum_base_yaw_rate
        )
        for case, reference in references.items()
    }
    body_basis = fit_corpus_centered_body_basis(references, yaw)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case, reference in references.items():
        plan = plan_rs4_attitude_commands(reference, body_basis, yaw[case])
        absolute_rate = np.abs(plan.command_rate_yaw_roll_pitch_deg_s)
        rate_p95 = np.percentile(absolute_rate, 95, axis=0)
        rate_max = np.max(absolute_rate, axis=0)
        base_yaw_rate = np.diff(plan.base_yaw_rad) / np.diff(reference.time_s)
        checks = {
            "command_envelope_feasible": bool(np.all(plan.command_feasible)),
            "attitude_reconstruction_exact": math.degrees(
                float(np.max(plan.reconstruction_error_rad))
            )
            <= 1e-5,
            "base_yaw_rate_bounded": float(np.max(np.abs(base_yaw_rate)))
            <= args.maximum_base_yaw_rate + 1e-9,
            "hard_command_rate_bounded": bool(
                np.all(rate_max <= args.hard_rate_limit_deg_s + 1e-9)
            ),
            "filming_command_rate_p95_bounded": bool(
                np.all(rate_p95 <= args.filming_rate_limit_deg_s + 1e-9)
            ),
        }
        command_deg = np.rad2deg(plan.command_yaw_roll_pitch_rad)
        row = {
            "case": case,
            "duration_s": float(reference.time_s[-1]),
            "command_feasible_ratio": float(np.mean(plan.command_feasible)),
            "attitude_reconstruction_error_max_deg": math.degrees(
                float(np.max(plan.reconstruction_error_rad))
            ),
            "base_yaw_rate_max_rad_s": float(np.max(np.abs(base_yaw_rate))),
            "yaw_command_min_deg": float(np.min(command_deg[:, 0])),
            "yaw_command_max_deg": float(np.max(command_deg[:, 0])),
            "roll_command_min_deg": float(np.min(command_deg[:, 1])),
            "roll_command_max_deg": float(np.max(command_deg[:, 1])),
            "pitch_command_min_deg": float(np.min(command_deg[:, 2])),
            "pitch_command_max_deg": float(np.max(command_deg[:, 2])),
            "yaw_rate_p95_deg_s": float(rate_p95[0]),
            "roll_rate_p95_deg_s": float(rate_p95[1]),
            "pitch_rate_p95_deg_s": float(rate_p95[2]),
            "yaw_rate_max_deg_s": float(rate_max[0]),
            "roll_rate_max_deg_s": float(rate_max[1]),
            "pitch_rate_max_deg_s": float(rate_max[2]),
            **checks,
            "passed": all(checks.values()),
        }
        rows.append(row)

    with (args.output_dir / "cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "cinebotrl_two_wheel_riser_rs4_attitude_audit_v1",
        "training_started": False,
        "ppo_authorized": False,
        "basis_fit_source": "accepted corrected 62-case semantic DFR corpus",
        "physical_motor_joint_labels_used": False,
        "command_order": ["ronin_yaw_from_rot_x", "ronin_roll_from_rot_y", "ronin_pitch_from_rot_z"],
        "body_basis_quat_xyzw": Rotation.from_matrix(body_basis).as_quat().tolist(),
        "body_basis_zyx_deg": Rotation.from_matrix(body_basis)
        .as_euler("ZYX", degrees=True)
        .tolist(),
        "case_count": len(rows),
        "passed_case_count": sum(row["passed"] for row in rows),
        "command_envelope_feasible_case_count": sum(
            row["command_envelope_feasible"] for row in rows
        ),
        "hard_rate_passed_case_count": sum(
            row["hard_command_rate_bounded"] for row in rows
        ),
        "filming_rate_p95_passed_case_count": sum(
            row["filming_command_rate_p95_bounded"] for row in rows
        ),
        "failed_cases": [row["case"] for row in rows if not row["passed"]],
        "thresholds": {
            "maximum_base_yaw_rate_rad_s": args.maximum_base_yaw_rate,
            "filming_command_rate_p95_deg_s": args.filming_rate_limit_deg_s,
            "hard_command_rate_max_deg_s": args.hard_rate_limit_deg_s,
        },
        "status": "diagnostic_only_pending_isaac_and_hardware_mapping_validation",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed_case_count"] == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
