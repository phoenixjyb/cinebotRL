#!/usr/bin/env python3
"""Gate learned riser rollouts against teacher and null-action baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


TRACKING_PROFILE = "riser_phase_consistent_v2"
PHASE_CONTRACT = "derivatives_scaled_by_progress_v1"
LEARNED_SOURCE = "torchscript_residual_policy"
ZERO_SOURCE = "zero_policy_action_baseline"
REGRESSION_METRICS = (
    "position_error_p95_m",
    "position_error_max_m",
    "attitude_error_p95_deg",
    "attitude_error_max_deg",
    "pitch_p95_deg",
    "pitch_max_deg",
    "riser_servo_error_p95_m",
    "riser_servo_error_max_m",
    "proxy_servo_error_p95_deg",
    "proxy_servo_error_max_deg",
)


def parse_cases(value: str) -> list[int]:
    cases = [int(item) for item in value.split(",") if item]
    if not cases or len(cases) != len(set(cases)) or any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("cases must be unique positive integers")
    return cases


def load_result(path: Path, case: int, expected_source: str) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("cases") != [case]
        or payload.get("trajectory_command_source") != expected_source
        or payload.get("tracking_profile") != TRACKING_PROFILE
        or payload.get("phase_feedforward_contract") != PHASE_CONTRACT
        or len(payload.get("results", [])) != 1
        or payload["results"][0].get("case") != case
    ):
        raise ValueError(f"rollout contract mismatch: {path}")
    return payload, payload["results"][0]


def regression_checks(
    teacher: dict, learned: dict, maximum_regression_fraction: float
) -> dict[str, bool]:
    multiplier = 1.0 + maximum_regression_fraction
    return {
        metric: learned[metric] <= multiplier * teacher[metric] + 1e-9
        for metric in REGRESSION_METRICS
    }


def gate_rollouts(
    *,
    teacher_dir: Path,
    learned_dir: Path,
    cases: list[int],
    policy: Path,
    mode: str,
    maximum_regression_fraction: float,
    zero_dir: Path | None = None,
    minimum_zero_improvement_fraction: float = 0.05,
) -> dict:
    if mode not in {"holdout", "all79"}:
        raise ValueError("unknown rollout gate mode")
    if mode == "all79" and cases != list(range(1, 80)):
        raise ValueError("all79 mode requires cases 1 through 79")
    if mode == "holdout" and zero_dir is None:
        raise ValueError("holdout mode requires zero-policy-action rollouts")
    if not 0.0 <= maximum_regression_fraction < 1.0:
        raise ValueError("invalid regression fraction")
    if not 0.0 < minimum_zero_improvement_fraction < 1.0:
        raise ValueError("invalid null-action improvement fraction")

    rows = []
    for case in cases:
        name = f"case_{case:04d}.json"
        teacher_payload, teacher = load_result(
            teacher_dir / name, case, "deterministic_teacher"
        )
        learned_payload, learned = load_result(
            learned_dir / name, case, LEARNED_SOURCE
        )
        zero = None
        if zero_dir is not None:
            _, zero = load_result(zero_dir / name, case, ZERO_SOURCE)
        checks = {
            "learned_hard_gate": learned_payload.get("passed") is True
            and learned.get("passed") is True,
            "teacher_hard_gate": teacher_payload.get("passed") is True
            and teacher.get("passed") is True,
            "bounded_residual": max(learned["residual_action_abs_max"])
            <= 1.0 + 1e-6,
        }
        checks.update(
            {
                f"regression_{metric}": passed
                for metric, passed in regression_checks(
                    teacher, learned, maximum_regression_fraction
                ).items()
            }
        )
        row = {
            "case": case,
            "checks": checks,
            "teacher": {metric: teacher[metric] for metric in REGRESSION_METRICS},
            "learned": {metric: learned[metric] for metric in REGRESSION_METRICS},
            "learned_residual_action_abs_max": learned["residual_action_abs_max"],
        }
        if zero is not None:
            row["zero"] = {metric: zero[metric] for metric in REGRESSION_METRICS}
            row["learned_beats_zero_position_p95"] = (
                learned["position_error_p95_m"] < zero["position_error_p95_m"]
            )
        rows.append(row)

    teacher_position_mean = statistics.fmean(
        row["teacher"]["position_error_p95_m"] for row in rows
    )
    learned_position_mean = statistics.fmean(
        row["learned"]["position_error_p95_m"] for row in rows
    )
    aggregate_checks = {
        "all_case_checks": all(all(row["checks"].values()) for row in rows),
        "learned_position_mean_within_teacher_budget": learned_position_mean
        <= (1.0 + maximum_regression_fraction) * teacher_position_mean,
    }
    means = {
        "teacher_position_p95_m": teacher_position_mean,
        "learned_position_p95_m": learned_position_mean,
    }
    if zero_dir is not None:
        zero_position_mean = statistics.fmean(
            row["zero"]["position_error_p95_m"] for row in rows
        )
        improved_case_count = sum(row["learned_beats_zero_position_p95"] for row in rows)
        means["zero_position_p95_m"] = zero_position_mean
        aggregate_checks.update(
            {
                "learned_beats_zero_by_required_mean": learned_position_mean
                <= (1.0 - minimum_zero_improvement_fraction) * zero_position_mean,
                "learned_beats_zero_on_majority_of_cases": improved_case_count
                > len(rows) / 2,
            }
        )
    return {
        "schema": f"cinebotrl_two_wheel_riser_residual_{mode}_gate_v1",
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "cases": cases,
        "case_count": len(cases),
        "maximum_regression_fraction": maximum_regression_fraction,
        "minimum_zero_improvement_fraction": (
            minimum_zero_improvement_fraction if zero_dir is not None else None
        ),
        "means": means,
        "aggregate_checks": aggregate_checks,
        "rows": rows,
        "passed": all(aggregate_checks.values()),
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("holdout", "all79"), required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--learned-dir", type=Path, required=True)
    parser.add_argument("--zero-dir", type=Path)
    parser.add_argument("--cases", type=parse_cases, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-regression-fraction", type=float, default=0.05)
    parser.add_argument(
        "--minimum-zero-improvement-fraction", type=float, default=0.05
    )
    args = parser.parse_args()
    summary = gate_rollouts(
        teacher_dir=args.teacher_dir,
        learned_dir=args.learned_dir,
        zero_dir=args.zero_dir,
        cases=args.cases,
        policy=args.policy,
        mode=args.mode,
        maximum_regression_fraction=args.maximum_regression_fraction,
        minimum_zero_improvement_fraction=args.minimum_zero_improvement_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
