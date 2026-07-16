#!/usr/bin/env python3
"""Separate planned reverse motion from post-fault recovery in a Gate C trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(value: float, deadband: float) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def first_index(rows: list[dict[str, object]], predicate) -> int | None:
    return next((index for index, row in enumerate(rows) if predicate(row)), None)


def count_sign_changes(values: list[float], deadband: float) -> int:
    signs = [sign(value, deadband) for value in values]
    nonzero = [value for value in signs if value]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def trace_event(row: dict[str, object]) -> dict[str, object]:
    return {
        "step": int(row["step"]),
        "elapsed_s": float(row["elapsed_s"]),
        "phase_time_s": float(row["phase_time_s"]),
        "feedforward_v_mps": float(row["phase_feedforward_v_mps"]),
        "vx_reference_mps": float(row["vx_reference_mps"]),
        "position_error_m": float(row["position_error_m"]),
        "base_xy_error_m": float(row["base_xy_error_m"]),
        "pitch_deg": float(row["pitch_deg"]),
        "proxy_yaw_error_deg": float(row["proxy_signed_error_deg"][2]),
    }


def audit(
    case_json: Path,
    plan_path: Path,
    *,
    expected_case_sha256: str,
    expected_plan_sha256: str,
    maximum_linear_velocity_mps: float,
    velocity_deadband_mps: float,
    yaw_branch_fault_deg: float,
    base_error_gate_m: float,
) -> dict[str, object]:
    case_hash = sha256_file(case_json)
    plan_hash = sha256_file(plan_path)
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if len(results) != 1:
        raise ValueError("expected exactly one Gate C result")
    result = results[0]
    trace = list(result.get("trace", []))
    if len(trace) < 2:
        raise ValueError("Gate C trace must contain at least two samples")
    if any(
        int(right["step"]) <= int(left["step"])
        for left, right in zip(trace, trace[1:])
    ):
        raise ValueError("Gate C trace steps must be strictly increasing")

    with np.load(plan_path, allow_pickle=False) as arrays:
        execution_time_s = np.asarray(arrays["execution_time_s"], dtype=np.float64)
        feedforward_v = np.asarray(arrays["feedforward_v_wz"], dtype=np.float64)[:, 0]
    if execution_time_s.ndim != 1 or feedforward_v.shape != (
        execution_time_s.size - 1,
    ):
        raise ValueError("plan timing/feed-forward arrays are inconsistent")
    if not np.isfinite(execution_time_s).all() or not np.isfinite(feedforward_v).all():
        raise ValueError("plan arrays contain non-finite values")

    yaw_fault_index = first_index(
        trace,
        lambda row: abs(float(row["proxy_signed_error_deg"][2])) > yaw_branch_fault_deg,
    )
    base_failure_index = first_index(
        trace, lambda row: float(row["base_xy_error_m"]) > base_error_gate_m
    )
    if yaw_fault_index is None:
        raise ValueError("trace does not contain the expected proxy-yaw branch fault")
    pre_fault = trace[:yaw_fault_index]
    post_fault = trace[yaw_fault_index:]
    if not pre_fault:
        raise ValueError("proxy-yaw fault occurs before any healthy trace sample")

    def is_reverse(row: dict[str, object]) -> bool:
        return float(row["phase_feedforward_v_mps"]) < -velocity_deadband_mps

    def is_direction_override(row: dict[str, object]) -> bool:
        feedforward = float(row["phase_feedforward_v_mps"])
        command = float(row["vx_reference_mps"])
        return sign(feedforward, velocity_deadband_mps) * sign(
            command, velocity_deadband_mps
        ) == -1

    def is_velocity_saturated(row: dict[str, object]) -> bool:
        return math.isclose(
            abs(float(row["vx_reference_mps"])),
            maximum_linear_velocity_mps,
            abs_tol=1e-9,
        )

    pre_reverse = [row for row in pre_fault if is_reverse(row)]
    post_saturation = [row for row in post_fault if is_velocity_saturated(row)]
    post_override = [row for row in post_fault if is_direction_override(row)]
    first_post_saturation = post_saturation[0] if post_saturation else None
    trace_commands = [float(row["vx_reference_mps"]) for row in trace]
    pre_commands = [float(row["vx_reference_mps"]) for row in pre_fault]
    post_commands = [float(row["vx_reference_mps"]) for row in post_fault]

    plan_signs = np.sign(
        np.where(np.abs(feedforward_v) > velocity_deadband_mps, feedforward_v, 0.0)
    )
    plan_nonzero_signs = plan_signs[plan_signs != 0.0]
    plan_reversals = int(np.count_nonzero(np.diff(plan_nonzero_signs) != 0.0))

    checks = {
        "case_json_hash_matches": case_hash == expected_case_sha256,
        "plan_hash_matches": plan_hash == expected_plan_sha256,
        "case_is_dynamic_reject": not bool(result["dynamic_quality_passed"]),
        "no_dataset_created": result["executed_residual_dataset"] is None,
        "reverse_motion_exists_in_plan": bool(np.any(plan_signs < 0.0)),
        "reverse_motion_observed_before_yaw_fault": bool(pre_reverse),
        "yaw_fault_precedes_base_gate_failure": (
            base_failure_index is not None and yaw_fault_index < base_failure_index
        ),
        "velocity_saturation_follows_yaw_fault": bool(post_saturation),
        "post_fault_recovery_changes_direction": (
            count_sign_changes(post_commands, velocity_deadband_mps) > 0
        ),
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_case_recovery_timeline_audit_v1",
        "case": int(result["case"]),
        "case_json": str(case_json.resolve()),
        "case_json_sha256": case_hash,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_hash,
        "trace_sample_count": len(trace),
        "trace_sampling_boundary": (
            "sampled at 1 Hz; unsuitable for gain identification"
        ),
        "plan": {
            "execution_duration_s": float(execution_time_s[-1]),
            "transition_count": int(feedforward_v.size),
            "forward_transition_count": int(np.count_nonzero(plan_signs > 0.0)),
            "reverse_transition_count": int(np.count_nonzero(plan_signs < 0.0)),
            "deadband_transition_count": int(np.count_nonzero(plan_signs == 0.0)),
            "direction_reversal_count": plan_reversals,
        },
        "pre_fault": {
            "sample_count": len(pre_fault),
            "reverse_sample_count": len(pre_reverse),
            "direction_override_sample_count": sum(
                is_direction_override(row) for row in pre_fault
            ),
            "velocity_saturation_sample_count": sum(
                is_velocity_saturated(row) for row in pre_fault
            ),
            "vx_reference_abs_max_mps": max(abs(value) for value in pre_commands),
            "vx_reference_sign_change_count": count_sign_changes(
                pre_commands, velocity_deadband_mps
            ),
            "position_error_max_m": max(
                float(row["position_error_m"]) for row in pre_fault
            ),
            "base_xy_error_max_m": max(
                float(row["base_xy_error_m"]) for row in pre_fault
            ),
        },
        "fault": trace_event(trace[yaw_fault_index]),
        "first_base_gate_failure": (
            None
            if base_failure_index is None
            else trace_event(trace[base_failure_index])
        ),
        "first_post_fault_velocity_saturation": (
            None
            if first_post_saturation is None
            else trace_event(first_post_saturation)
        ),
        "post_fault": {
            "sample_count": len(post_fault),
            "direction_override_sample_count": len(post_override),
            "velocity_saturation_sample_count": len(post_saturation),
            "vx_reference_abs_max_mps": max(abs(value) for value in post_commands),
            "vx_reference_sign_change_count": count_sign_changes(
                post_commands, velocity_deadband_mps
            ),
            "position_error_max_m": max(
                float(row["position_error_m"]) for row in post_fault
            ),
            "base_xy_error_max_m": max(
                float(row["base_xy_error_m"]) for row in post_fault
            ),
        },
        "whole_trace_vx_reference_sign_change_count": count_sign_changes(
            trace_commands, velocity_deadband_mps
        ),
        "diagnosis": (
            "proxy_yaw_branch_fault_is_primary_observed_precursor; "
            "velocity-saturated bidirectional recovery is downstream"
        ),
        "reverse_controller_change_authorized": False,
        "reason_no_reverse_controller_change": (
            "sealed 1 Hz evidence shows bounded reverse tracking before the yaw fault; "
            "the corrected-yaw canary is required before attributing remaining error "
            "to recovery"
        ),
        "corrected_dynamic_rerun_required_for_causality": True,
        "runtime_controller_changed": False,
        "gpu_work_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--expected-case-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--maximum-linear-velocity-mps", type=float, default=0.4)
    parser.add_argument("--velocity-deadband-mps", type=float, default=1e-4)
    parser.add_argument("--yaw-branch-fault-deg", type=float, default=180.0)
    parser.add_argument("--base-error-gate-m", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.maximum_linear_velocity_mps,
        args.velocity_deadband_mps,
        args.yaw_branch_fault_deg,
        args.base_error_gate_m,
    ) <= 0.0:
        raise ValueError("audit thresholds must be positive")

    summary = audit(
        args.case_json,
        args.plan,
        expected_case_sha256=args.expected_case_sha256,
        expected_plan_sha256=args.expected_plan_sha256,
        maximum_linear_velocity_mps=args.maximum_linear_velocity_mps,
        velocity_deadband_mps=args.velocity_deadband_mps,
        yaw_branch_fault_deg=args.yaw_branch_fault_deg,
        base_error_gate_m=args.base_error_gate_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "case": summary["case"],
                "fault": summary["fault"],
                "pre_fault": summary["pre_fault"],
                "post_fault": summary["post_fault"],
                "diagnosis": summary["diagnosis"],
                "passed": summary["passed"],
            },
            indent=2,
        )
    )
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
