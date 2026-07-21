#!/usr/bin/env python3
"""Audit a bounded camera-recovery governor candidate from sealed case-78 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CANONICAL_GATE_SHA256 = (
    "46ab1f27d2ed16271853e068e21497d66f6cacfb8599f98dde0c72df6d31c97a"
)
MAXIMUM_STEPS = 115381
ERROR_START_M = 0.13
ERROR_FULL_M = 0.155
MINIMUM_RECOVERY_SCALE = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_scale(error_m: float, saturated: bool) -> float:
    if not saturated:
        return 1.0
    severity = min(
        1.0,
        max(0.0, (error_m - ERROR_START_M) / (ERROR_FULL_M - ERROR_START_M)),
    )
    return 1.0 - severity * (1.0 - MINIMUM_RECOVERY_SCALE)


def audit(gate_path: Path) -> dict[str, object]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    results = gate.get("results")
    result = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    checks = result.get("checks", {})
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    trace = result.get("trace", [])
    current_scales = [float(item["progress_scale"]) for item in trace]
    candidate_scales = [
        min(
            float(item["progress_scale"]),
            recovery_scale(
                float(item["position_error_m"]),
                bool(item["camera_lever_arm_correction_saturated"]),
            ),
        )
        for item in trace
    ]
    current_mean = (
        sum(current_scales) / len(current_scales) if current_scales else 0.0
    )
    candidate_mean = (
        sum(candidate_scales) / len(candidate_scales) if candidate_scales else 0.0
    )
    step_multiplier = (
        current_mean / candidate_mean if candidate_mean > 0.0 else math.inf
    )
    completed_steps = int(result.get("completed_steps", 0))
    projected_steps = (
        math.ceil(completed_steps * step_multiplier)
        if math.isfinite(step_multiplier)
        else None
    )
    high_error_trace_ratio = (
        sum(float(item["position_error_m"]) > 0.15 for item in trace) / len(trace)
        if trace
        else 0.0
    )
    saturated_trace_ratio = (
        sum(bool(item["camera_lever_arm_correction_saturated"]) for item in trace)
        / len(trace)
        if trace
        else 0.0
    )
    audit_checks = {
        "canonical_gate_hash": sha256_file(gate_path) == CANONICAL_GATE_SHA256,
        "case78_complete": result.get("case") == 78
        and result.get("completed_phase_time_s")
        == result.get("execution_duration_s")
        and result.get("termination") is None,
        "only_position_p95_failed": failed_checks == ["position_p95_bounded"],
        "position_miss_is_near_gate": 0.15
        < float(result.get("position_error_p95_m", math.inf))
        <= 0.175,
        "position_max_still_bounded": float(
            result.get("position_error_max_m", math.inf)
        )
        <= 0.25,
        "lever_correction_mostly_saturated": saturated_trace_ratio >= 0.90,
        "candidate_changes_trace_progress": candidate_mean < current_mean,
        "candidate_stays_inside_step_horizon": projected_steps is not None
        and projected_steps <= MAXIMUM_STEPS,
        "candidate_does_not_change_thresholds": True,
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_camera_recovery_audit_v1",
        "case": 78,
        "gate": {
            "path": str(gate_path.resolve()),
            "sha256": sha256_file(gate_path),
        },
        "failed_checks": failed_checks,
        "position_error_p95_m": result.get("position_error_p95_m"),
        "position_error_max_m": result.get("position_error_max_m"),
        "trace_sample_count": len(trace),
        "high_error_trace_ratio": high_error_trace_ratio,
        "lever_correction_saturated_trace_ratio": saturated_trace_ratio,
        "current_progress_scale_mean": current_mean,
        "candidate_progress_scale_mean": candidate_mean,
        "estimated_step_multiplier": step_multiplier,
        "completed_steps": completed_steps,
        "projected_candidate_steps": projected_steps,
        "maximum_steps": MAXIMUM_STEPS,
        "candidate": {
            "enable_camera_error_recovery_governor": True,
            "camera_recovery_error_start_m": ERROR_START_M,
            "camera_recovery_error_full_m": ERROR_FULL_M,
            "minimum_camera_recovery_scale": MINIMUM_RECOVERY_SCALE,
            "changes_source_plan": False,
            "changes_inner_lqr": False,
            "changes_dynamic_gate_thresholds": False,
        },
        "checks": audit_checks,
        "candidate_supported_for_bounded_canary": all(audit_checks.values()),
        "offline_trace_estimate_is_physical_proof": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "split_change_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["candidate_supported_for_bounded_canary"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

