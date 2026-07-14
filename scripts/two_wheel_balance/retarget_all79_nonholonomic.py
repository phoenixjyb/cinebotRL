#!/usr/bin/env python3
"""Generate deterministic unicycle-plus-arm candidates for full all-79 paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    discover_full_stage,
    parse_acquisition_time_scale_overrides,
    regenerate_acquisition_prefix,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (  # noqa: E402
    UrdfPositionKinematics,
    integrate_unicycle,
)


HOME_V0 = np.array([0.0, np.pi / 2.0, 3.0 * np.pi / 4.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-stage", type=Path, required=True)
    parser.add_argument("--contract-audit-summary", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--maximum-linear-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-yaw-rate", type=float, default=0.4)
    parser.add_argument("--maximum-arm-rate", type=float, default=0.5)
    parser.add_argument("--position-scale-m", type=float, default=0.01)
    parser.add_argument("--control-regularization", type=float, default=0.01)
    parser.add_argument("--maximum-mean-error-m", type=float, default=0.05)
    parser.add_argument("--maximum-p95-error-m", type=float, default=0.10)
    parser.add_argument("--maximum-error-m", type=float, default=0.20)
    parser.add_argument("--maximum-acquisition-time-scale", type=float, default=2.0)
    parser.add_argument("--acquisition-time-scale-overrides", default="")
    parser.add_argument("--save-case-arrays", action="store_true")
    return parser.parse_args()


def load_acquisition_end_indices(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "cinebotrl_two_wheel_all79_contract_audit_v1":
        raise ValueError(f"wrong contract audit schema in {path}")
    if payload.get("passed_for_position_retargeting") is not True:
        raise ValueError(f"contract audit is not approved for position retargeting: {path}")
    if payload.get("training_started") is not False:
        raise ValueError(f"contract audit unexpectedly reports training started: {path}")
    indices = {
        int(item["case"]): int(item["acquisition_end_index"])
        for item in payload.get("cases", [])
    }
    expected = set(range(1, 80))
    if set(indices) != expected:
        raise ValueError(f"contract audit cases differ: {sorted(set(indices) ^ expected)}")
    return indices


def retarget_case(
    reference,
    kinematics: UrdfPositionKinematics,
    acquisition_end_index: int,
    acquisition_time_scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if acquisition_time_scale < 1.0:
        raise ValueError("acquisition time scale must be at least 1.0")
    state = np.concatenate((np.zeros(3), HOME_V0.copy()))
    targets, semantic_start = regenerate_acquisition_prefix(
        reference, kinematics.position(state), acquisition_end_index
    )
    time_s = reference.time_s.copy()
    original_acquisition_duration = time_s[acquisition_end_index]
    time_s[: acquisition_end_index + 1] *= acquisition_time_scale
    time_s[acquisition_end_index + 1 :] += (
        acquisition_time_scale - 1.0
    ) * original_acquisition_duration

    states = np.empty((len(targets), 6), dtype=np.float64)
    achieved = np.empty_like(targets)
    controls = np.zeros((len(targets) - 1, 5), dtype=np.float64)
    states[0] = state
    achieved[0] = kinematics.position(state)
    previous_control = np.zeros(5, dtype=np.float64)

    for index in range(1, len(targets)):
        dt = float(time_s[index] - time_s[index - 1])
        arm_delta_lower = np.maximum(
            -args.maximum_arm_rate * dt,
            kinematics.arm_lower - state[3:],
        )
        arm_delta_upper = np.minimum(
            args.maximum_arm_rate * dt,
            kinematics.arm_upper - state[3:],
        )
        lower = np.concatenate(
            (
                [-args.maximum_linear_velocity, -args.maximum_yaw_rate],
                arm_delta_lower,
            )
        )
        upper = np.concatenate(
            (
                [args.maximum_linear_velocity, args.maximum_yaw_rate],
                arm_delta_upper,
            )
        )
        initial = np.clip(previous_control, lower, upper)

        def candidate(control: np.ndarray) -> np.ndarray:
            base = integrate_unicycle(state[:3], control[0], control[1], dt)
            return np.concatenate((base, state[3:] + control[2:]))

        def residual(control: np.ndarray) -> np.ndarray:
            next_state = candidate(control)
            position_error = (kinematics.position(next_state) - targets[index]) / args.position_scale_m
            regularization = args.control_regularization * np.concatenate(
                (
                    control[:1] / args.maximum_linear_velocity,
                    control[1:2] / args.maximum_yaw_rate,
                    control[2:] / max(args.maximum_arm_rate * dt, 1e-9),
                )
            )
            return np.concatenate((position_error, regularization))

        solution = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=80,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        previous_control = solution.x
        state = candidate(solution.x)
        controls[index - 1] = solution.x
        states[index] = state
        achieved[index] = kinematics.position(state)

    errors = np.linalg.norm(achieved - targets, axis=1)
    arm_rate = np.diff(states[:, 3:], axis=0) / np.diff(time_s)[:, None]
    acquisition_errors = errors[: acquisition_end_index + 1]
    semantic_errors = errors[acquisition_end_index:]
    checks = {
        "finite": bool(
            np.isfinite(states).all()
            and np.isfinite(controls).all()
            and np.isfinite(errors).all()
        ),
        "zero_lateral_base_action": True,
        "linear_velocity_bounded": bool(
            np.max(np.abs(controls[:, 0])) <= args.maximum_linear_velocity + 1e-9
        ),
        "yaw_rate_bounded": bool(
            np.max(np.abs(controls[:, 1])) <= args.maximum_yaw_rate + 1e-9
        ),
        "arm_rate_bounded": bool(
            np.max(np.abs(arm_rate)) <= args.maximum_arm_rate + 1e-8
        ),
        "arm_position_bounded": bool(
            np.all(states[:, 3:] >= kinematics.arm_lower - 1e-9)
            and np.all(states[:, 3:] <= kinematics.arm_upper + 1e-9)
        ),
        "mean_error_bounded": float(np.mean(errors)) <= args.maximum_mean_error_m,
        "p95_error_bounded": float(np.percentile(errors, 95))
        <= args.maximum_p95_error_m,
        "maximum_error_bounded": float(np.max(errors)) <= args.maximum_error_m,
        "semantic_p95_error_bounded": float(np.percentile(semantic_errors, 95))
        <= args.maximum_p95_error_m,
        "semantic_maximum_error_bounded": float(np.max(semantic_errors))
        <= args.maximum_error_m,
    }
    summary = {
        "case": reference.case,
        "source_duration_s": float(reference.time_s[-1]),
        "retargeted_duration_s": float(time_s[-1]),
        "samples": len(reference.time_s),
        "acquisition_end_index": acquisition_end_index,
        "source_acquisition_duration_s": float(original_acquisition_duration),
        "retargeted_acquisition_duration_s": float(time_s[acquisition_end_index]),
        "acquisition_time_scale": acquisition_time_scale,
        "acquisition_retimed": acquisition_time_scale > 1.0,
        "semantic_start_position_world_m": semantic_start.tolist(),
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(np.max(errors)),
        "position_error_final_m": float(errors[-1]),
        "acquisition_position_error_p95_m": float(
            np.percentile(acquisition_errors, 95)
        ),
        "acquisition_position_error_max_m": float(np.max(acquisition_errors)),
        "semantic_position_error_p95_m": float(np.percentile(semantic_errors, 95)),
        "semantic_position_error_max_m": float(np.max(semantic_errors)),
        "maximum_abs_linear_velocity_mps": float(np.max(np.abs(controls[:, 0]))),
        "maximum_abs_yaw_rate_radps": float(np.max(np.abs(controls[:, 1]))),
        "maximum_abs_arm_rate_radps": float(np.max(np.abs(arm_rate))),
        "checks": checks,
        "passed": all(checks.values()),
    }
    arrays = {
        "time_s": time_s,
        "source_time_s": reference.time_s,
        "target_position_world_m": targets,
        "achieved_position_world_m": achieved,
        "base_arm_q": states,
        "control_v_wz_darm": controls,
        "position_error_m": errors,
    }
    return summary, arrays


def main() -> int:
    args = parse_args()
    references = discover_full_stage(args.full_stage)
    acquisition_end_indices = load_acquisition_end_indices(args.contract_audit_summary)
    kinematics = UrdfPositionKinematics(args.urdf)
    cases = (
        sorted(references)
        if args.cases.strip().lower() == "all"
        else [int(value) for value in args.cases.split(",") if value.strip()]
    )
    if not cases or any(case not in references for case in cases):
        raise ValueError(f"invalid cases: {cases}")
    acquisition_scale_overrides = parse_acquisition_time_scale_overrides(
        args.acquisition_time_scale_overrides
    )
    unselected_overrides = set(acquisition_scale_overrides) - set(cases)
    if unselected_overrides:
        raise ValueError(
            f"acquisition overrides target unselected cases: {sorted(unselected_overrides)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        attempts = []
        acquisition_time_scales = (
            [acquisition_scale_overrides[case]]
            if case in acquisition_scale_overrides
            else np.arange(1.0, args.maximum_acquisition_time_scale + 0.001, 0.25)
        )
        for acquisition_time_scale in acquisition_time_scales:
            summary, arrays = retarget_case(
                references[case],
                kinematics,
                acquisition_end_indices[case],
                float(acquisition_time_scale),
                args,
            )
            attempts.append(summary)
            if summary["passed"]:
                break
        summary["attempted_acquisition_time_scales"] = [
            item["acquisition_time_scale"] for item in attempts
        ]
        if args.save_case_arrays:
            np.savez_compressed(args.output_dir / f"case_{case:04d}.npz", **arrays)
        rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    result = {
        "schema": "cinebotrl_two_wheel_nonholonomic_retarget_smoke_v2",
        "training_started": False,
        "control_contract": "unicycle_v_wz_plus_arm3_delta",
        "acquisition_contract": "regenerated_home_to_audited_semantic_start_v1",
        "orientation_tracking_included": False,
        "physical_gimbal_adapter_included": False,
        "acquisition_time_scale_overrides": {
            str(case): scale for case, scale in acquisition_scale_overrides.items()
        },
        "cases": cases,
        "passed_case_count": sum(row["passed"] for row in rows),
        "retimed_cases": [row["case"] for row in rows if row["acquisition_retimed"]],
        "source_duration_total_s": float(sum(row["source_duration_s"] for row in rows)),
        "retargeted_duration_total_s": float(
            sum(row["retargeted_duration_s"] for row in rows)
        ),
        "maximum_position_error_m": float(
            max(row["position_error_max_m"] for row in rows)
        ),
        "maximum_semantic_position_error_m": float(
            max(row["semantic_position_error_max_m"] for row in rows)
        ),
        "maximum_abs_linear_velocity_mps": float(
            max(row["maximum_abs_linear_velocity_mps"] for row in rows)
        ),
        "maximum_abs_yaw_rate_radps": float(
            max(row["maximum_abs_yaw_rate_radps"] for row in rows)
        ),
        "maximum_abs_arm_rate_radps": float(
            max(row["maximum_abs_arm_rate_radps"] for row in rows)
        ),
        "passed": all(row["passed"] for row in rows),
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    flat_rows = [
        {key: value for key, value in row.items() if key != "checks"} for row in rows
    ]
    with (args.output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(flat_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
