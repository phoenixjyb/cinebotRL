#!/usr/bin/env python3
"""Gate learned riser rollouts against teacher and null-action baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TRACKING_PROFILE = "riser_phase_consistent_v2"
PHASE_CONTRACT = "derivatives_scaled_by_progress_v1"
LEARNED_SOURCE = "torchscript_residual_policy"
ZERO_SOURCE = "zero_policy_action_baseline"
LEGACY_POLICY_COMMAND_CONTRACT = "legacy_phase_feedforward_residual_v1"
MODEL_BASED_POLICY_COMMAND_CONTRACT = (
    "model_based_planner_plus_bounded_policy_residual_v1"
)
MODEL_BASED_GATE_SCHEMA_VERSION = "v2"
BALANCE_SAFETY_CONTRACT = "balance_first_rollout_safety_v1"
DEFAULT_MAXIMUM_PITCH_DEG = 12.0
DEFAULT_MAXIMUM_SATURATION_RATIO = 0.20
DEFAULT_MAXIMUM_RISER_THERMAL_LOAD = 1.0
DEFAULT_MAXIMUM_RISER_PEAK_FORCE_VIOLATIONS = 0
THERMAL_CHECKS = (
    "initialization_riser_thermal_force_observed",
    "initialization_riser_thermal_load_bounded",
    "initialization_riser_peak_force_bounded",
    "riser_thermal_force_observed",
    "riser_thermal_load_bounded",
    "riser_peak_force_bounded",
)
POLICY_COMMAND_CONTRACTS = {
    LEGACY_POLICY_COMMAND_CONTRACT: {
        "teacher_source": "deterministic_teacher",
        "learned_source": LEARNED_SOURCE,
        "zero_source": ZERO_SOURCE,
        "policy_command_base": "phase_feedforward",
        "residual_action_scales": [0.3, 0.4, 0.1],
    },
    MODEL_BASED_POLICY_COMMAND_CONTRACT: {
        "teacher_source": "model_based_planner_plus_zero_policy_residual",
        "learned_source": "model_based_planner_plus_torchscript_residual",
        "zero_source": "model_based_planner_plus_zero_policy_residual",
        "policy_command_base": "model_based_planner",
        "residual_action_scales": [0.05, 0.05, 0.02],
    },
}
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


def artifact_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def parse_cases(value: str) -> list[int]:
    cases = [int(item) for item in value.split(",") if item]
    if not cases or len(cases) != len(set(cases)) or any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("cases must be unique positive integers")
    return cases


def load_result(
    path: Path,
    case: int,
    expected_source: str,
    expected_tracking_profile: str = TRACKING_PROFILE,
    expected_policy_command_base: str = "phase_feedforward",
    expected_residual_action_scales: list[float] | None = None,
) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("cases") != [case]
        or payload.get("trajectory_command_source") != expected_source
        or payload.get("tracking_profile") != expected_tracking_profile
        or payload.get("phase_feedforward_contract") != PHASE_CONTRACT
        or payload.get("policy_command_base") != expected_policy_command_base
        or payload.get("residual_action_scales")
        != (expected_residual_action_scales or [0.3, 0.4, 0.1])
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


def balance_safety_snapshot(
    payload: dict,
    result: dict,
    *,
    maximum_pitch_deg: float,
    maximum_saturation_ratio: float,
) -> dict:
    checks = result.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("rollout is missing dynamic safety checks")
    numeric_fields = (
        "pitch_max_deg",
        "action_saturation_ratio",
        "riser_saturation_ratio",
        "proxy_saturation_ratio",
        "riser_thermal_load_max",
    )
    if any(
        not isinstance(result.get(name), (int, float))
        or isinstance(result.get(name), bool)
        or not math.isfinite(float(result[name]))
        or float(result[name]) < 0.0
        for name in numeric_fields
    ):
        raise ValueError("rollout is missing numeric balance safety evidence")
    if (
        not isinstance(result.get("riser_peak_force_violation_count"), int)
        or isinstance(result.get("riser_peak_force_violation_count"), bool)
        or result["riser_peak_force_violation_count"] < 0
    ):
        raise ValueError("rollout has invalid peak-force violation evidence")
    saturation_values = [
        float(result["action_saturation_ratio"]),
        float(result["riser_saturation_ratio"]),
        float(result["proxy_saturation_ratio"]),
    ]
    snapshot = {
        "payload_dynamic_quality_passed": (
            payload.get("dynamic_quality_passed") is True
        ),
        "payload_thermal_admission_passed": (
            payload.get("thermal_admission_passed") is True
        ),
        "result_dynamic_quality_passed": (
            result.get("dynamic_quality_passed") is True
        ),
        "result_thermal_admission_passed": (
            result.get("thermal_admission_passed") is True
        ),
        "controller_evidence_passed": (
            result.get("controller_evidence_passed") is True
        ),
        "completed_reference": checks.get("completed_reference") is True,
        "termination_absent": result.get("termination") is None,
        "no_termination_check": checks.get("no_termination") is True,
        "pitch_bounded_check": checks.get("pitch_bounded") is True,
        "saturation_checks_passed": all(
            checks.get(name) is True
            for name in (
                "initialization_action_saturation_bounded",
                "action_saturation_bounded",
                "riser_saturation_bounded",
                "proxy_saturation_bounded",
            )
        ),
        "thermal_checks_passed": all(
            checks.get(name) is True for name in THERMAL_CHECKS
        ),
        "pitch_max_deg": float(result["pitch_max_deg"]),
        "action_saturation_ratio": saturation_values[0],
        "riser_saturation_ratio": saturation_values[1],
        "proxy_saturation_ratio": saturation_values[2],
        "riser_thermal_load_max": float(result["riser_thermal_load_max"]),
        "riser_peak_force_violation_count": int(
            result["riser_peak_force_violation_count"]
        ),
    }
    snapshot["passed"] = (
        all(
            snapshot[name] is True
            for name in (
                "payload_dynamic_quality_passed",
                "payload_thermal_admission_passed",
                "result_dynamic_quality_passed",
                "result_thermal_admission_passed",
                "controller_evidence_passed",
                "completed_reference",
                "termination_absent",
                "no_termination_check",
                "pitch_bounded_check",
                "saturation_checks_passed",
                "thermal_checks_passed",
            )
        )
        and snapshot["pitch_max_deg"] <= maximum_pitch_deg
        and max(saturation_values) <= maximum_saturation_ratio
        and snapshot["riser_thermal_load_max"]
        <= DEFAULT_MAXIMUM_RISER_THERMAL_LOAD
        and snapshot["riser_peak_force_violation_count"]
        <= DEFAULT_MAXIMUM_RISER_PEAK_FORCE_VIOLATIONS
    )
    return snapshot


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
    expected_tracking_profile: str = TRACKING_PROFILE,
    policy_command_contract: str = LEGACY_POLICY_COMMAND_CONTRACT,
    rollout_admission: Path | None = None,
    preflight_receipt: Path | None = None,
    plan_manifest: Path | None = None,
    execution_commit: str | None = None,
    maximum_pitch_deg: float = DEFAULT_MAXIMUM_PITCH_DEG,
    maximum_saturation_ratio: float = DEFAULT_MAXIMUM_SATURATION_RATIO,
) -> dict:
    if mode not in {"validation_canary", "holdout", "all79"}:
        raise ValueError("unknown rollout gate mode")
    if mode == "all79" and cases != list(range(1, 80)):
        raise ValueError("all79 mode requires cases 1 through 79")
    if mode in {"validation_canary", "holdout"} and zero_dir is None:
        raise ValueError(f"{mode} mode requires zero-policy-action rollouts")
    if not 0.0 <= maximum_regression_fraction < 1.0:
        raise ValueError("invalid regression fraction")
    if not 0.0 < minimum_zero_improvement_fraction < 1.0:
        raise ValueError("invalid null-action improvement fraction")
    if not math.isfinite(maximum_pitch_deg) or maximum_pitch_deg <= 0.0:
        raise ValueError("maximum pitch must be positive")
    if (
        not math.isfinite(maximum_saturation_ratio)
        or not 0.0 <= maximum_saturation_ratio <= 1.0
    ):
        raise ValueError("invalid maximum saturation ratio")
    if policy_command_contract not in POLICY_COMMAND_CONTRACTS:
        raise ValueError("unknown policy command contract")
    model_based_gate = (
        policy_command_contract == MODEL_BASED_POLICY_COMMAND_CONTRACT
    )
    provenance = (
        rollout_admission,
        preflight_receipt,
        plan_manifest,
        execution_commit,
    )
    if (
        model_based_gate
        and (
            any(value is None for value in provenance)
            or re.fullmatch(r"[0-9a-f]{40}", str(execution_commit)) is None
        )
    ):
        raise ValueError("model-based rollout mode requires bound runtime provenance")
    command_contract = POLICY_COMMAND_CONTRACTS[policy_command_contract]

    rows = []
    for case in cases:
        name = f"case_{case:04d}.json"
        teacher_payload, teacher = load_result(
            teacher_dir / name,
            case,
            command_contract["teacher_source"],
            expected_tracking_profile,
            command_contract["policy_command_base"],
            command_contract["residual_action_scales"],
        )
        learned_payload, learned = load_result(
            learned_dir / name,
            case,
            command_contract["learned_source"],
            expected_tracking_profile,
            command_contract["policy_command_base"],
            command_contract["residual_action_scales"],
        )
        zero = None
        profile_checks = {
            "teacher_tracking_profile": teacher_payload.get("tracking_profile")
            == expected_tracking_profile,
            "learned_tracking_profile": learned_payload.get("tracking_profile")
            == expected_tracking_profile,
        }
        if zero_dir is not None:
            zero_payload, zero = load_result(
                zero_dir / name,
                case,
                command_contract["zero_source"],
                expected_tracking_profile,
                command_contract["policy_command_base"],
                command_contract["residual_action_scales"],
            )
            profile_checks["zero_tracking_profile"] = (
                zero_payload.get("tracking_profile") == expected_tracking_profile
            )
        teacher_safety = None
        learned_safety = None
        zero_safety = None
        if model_based_gate:
            teacher_safety = balance_safety_snapshot(
                teacher_payload,
                teacher,
                maximum_pitch_deg=maximum_pitch_deg,
                maximum_saturation_ratio=maximum_saturation_ratio,
            )
            learned_safety = balance_safety_snapshot(
                learned_payload,
                learned,
                maximum_pitch_deg=maximum_pitch_deg,
                maximum_saturation_ratio=maximum_saturation_ratio,
            )
            if zero is not None:
                zero_safety = balance_safety_snapshot(
                    zero_payload,
                    zero,
                    maximum_pitch_deg=maximum_pitch_deg,
                    maximum_saturation_ratio=maximum_saturation_ratio,
                )
        checks = {
            "learned_hard_gate": learned_payload.get("passed") is True
            and learned.get("passed") is True,
            "teacher_hard_gate": teacher_payload.get("passed") is True
            and teacher.get("passed") is True,
            "bounded_residual": max(learned["residual_action_abs_max"])
            <= 1.0 + 1e-6,
        }
        if model_based_gate:
            checks["learned_balance_safety"] = learned_safety["passed"] is True
            checks["teacher_balance_safety"] = teacher_safety["passed"] is True
            if zero_safety is not None:
                checks["zero_balance_safety"] = zero_safety["passed"] is True
        checks.update(profile_checks)
        checks.update(
            {
                f"regression_{metric}": passed
                for metric, passed in regression_checks(
                    teacher, learned, maximum_regression_fraction
                ).items()
            }
        )
        teacher_path = teacher_dir / name
        learned_path = learned_dir / name
        row = {
            "case": case,
            "checks": checks,
            "teacher": {metric: teacher[metric] for metric in REGRESSION_METRICS},
            "learned": {metric: learned[metric] for metric in REGRESSION_METRICS},
            "learned_residual_action_abs_max": learned["residual_action_abs_max"],
            "teacher_rollout": artifact_identity(teacher_path),
            "learned_rollout": artifact_identity(learned_path),
        }
        if model_based_gate:
            row["teacher_safety"] = teacher_safety
            row["learned_safety"] = learned_safety
        if zero is not None:
            row["zero"] = {metric: zero[metric] for metric in REGRESSION_METRICS}
            if model_based_gate:
                row["zero_safety"] = zero_safety
            row["learned_beats_zero_position_p95"] = (
                learned["position_error_p95_m"] < zero["position_error_p95_m"]
            )
            row["zero_rollout"] = artifact_identity(zero_dir / name)
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
        "schema": (
            f"cinebotrl_two_wheel_riser_residual_{mode}_gate_"
            f"{MODEL_BASED_GATE_SCHEMA_VERSION}"
            if model_based_gate
            else f"cinebotrl_two_wheel_riser_residual_{mode}_gate_v1"
        ),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "cases": cases,
        "case_count": len(cases),
        "maximum_regression_fraction": maximum_regression_fraction,
        "minimum_zero_improvement_fraction": (
            minimum_zero_improvement_fraction if zero_dir is not None else None
        ),
        "expected_tracking_profile": expected_tracking_profile,
        "policy_command_contract": policy_command_contract,
        "residual_action_scales": command_contract["residual_action_scales"],
        **(
            {
                "balance_safety_contract": BALANCE_SAFETY_CONTRACT,
                "maximum_pitch_deg": maximum_pitch_deg,
                "maximum_saturation_ratio": maximum_saturation_ratio,
                "maximum_riser_thermal_load": (
                    DEFAULT_MAXIMUM_RISER_THERMAL_LOAD
                ),
                "maximum_riser_peak_force_violations": (
                    DEFAULT_MAXIMUM_RISER_PEAK_FORCE_VIOLATIONS
                ),
            }
            if model_based_gate
            else {}
        ),
        "rollout_admission": (
            artifact_identity(rollout_admission)
            if rollout_admission is not None
            else None
        ),
        "preflight_receipt": (
            artifact_identity(preflight_receipt)
            if preflight_receipt is not None
            else None
        ),
        "plan_manifest": (
            artifact_identity(plan_manifest) if plan_manifest is not None else None
        ),
        "execution_commit": execution_commit,
        "means": means,
        "aggregate_checks": aggregate_checks,
        "rows": rows,
        "passed": all(aggregate_checks.values()),
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validation_canary", "holdout", "all79"), required=True
    )
    parser.add_argument("--rollout-admission", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--plan-manifest", type=Path)
    parser.add_argument("--execution-commit")
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
    parser.add_argument(
        "--maximum-pitch-deg", type=float, default=DEFAULT_MAXIMUM_PITCH_DEG
    )
    parser.add_argument(
        "--maximum-saturation-ratio",
        type=float,
        default=DEFAULT_MAXIMUM_SATURATION_RATIO,
    )
    parser.add_argument("--expected-tracking-profile", default=TRACKING_PROFILE)
    parser.add_argument(
        "--policy-command-contract",
        choices=tuple(POLICY_COMMAND_CONTRACTS),
        default=LEGACY_POLICY_COMMAND_CONTRACT,
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
        expected_tracking_profile=args.expected_tracking_profile,
        policy_command_contract=args.policy_command_contract,
        rollout_admission=args.rollout_admission,
        preflight_receipt=args.preflight_receipt,
        plan_manifest=args.plan_manifest,
        execution_commit=args.execution_commit,
        maximum_pitch_deg=args.maximum_pitch_deg,
        maximum_saturation_ratio=args.maximum_saturation_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
