#!/usr/bin/env python3
"""Select a bounded case-78 camera-correction cap candidate from sealed traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

if __package__:
    from .audit_case78_recovery_outcome import (
        BASELINE_SHA256,
        RECOVERY_SHA256,
        load_result,
        quantile,
        sha256_file,
    )
else:
    from audit_case78_recovery_outcome import (
        BASELINE_SHA256,
        RECOVERY_SHA256,
        load_result,
        quantile,
        sha256_file,
    )


CURRENT_CAP_M = 0.05
CANDIDATE_CAP_M = 0.10
P95_GATE_M = 0.15
MAX_GATE_M = 0.25
P95_PROJECTION_MARGIN_M = 0.015
MAX_PROJECTION_MARGIN_M = 0.015


def clipped_correction(
    lever_error_xy_m: list[float], cap_m: float
) -> list[float]:
    raw = [-float(value) for value in lever_error_xy_m]
    norm = math.hypot(*raw)
    scale = min(1.0, cap_m / norm) if norm > 0.0 else 1.0
    return [value * scale for value in raw]


def project_trace(
    trace: list[dict[str, object]], candidate_cap_m: float
) -> dict[str, float | int]:
    if not math.isfinite(candidate_cap_m) or candidate_cap_m <= CURRENT_CAP_M:
        raise ValueError("candidate cap must be finite and exceed the current cap")
    projected_errors: list[float] = []
    current_errors: list[float] = []
    current_correction_mismatch_max = 0.0
    position_norm_mismatch_max = 0.0
    for item in trace:
        lever_error = [float(value) for value in item["camera_lever_arm_error_xy_m"]]
        recorded_correction = [
            float(value) for value in item["camera_lever_arm_correction_xy_m"]
        ]
        position_error_xyz = [
            float(value) for value in item["camera_position_error_xyz_m"]
        ]
        if (
            len(lever_error) != 2
            or len(recorded_correction) != 2
            or len(position_error_xyz) != 3
        ):
            raise ValueError("trace vectors have invalid dimensions")
        if not all(
            math.isfinite(value)
            for value in lever_error + recorded_correction + position_error_xyz
        ):
            raise ValueError("trace vectors contain non-finite values")

        reconstructed = clipped_correction(lever_error, CURRENT_CAP_M)
        correction_mismatch = math.hypot(
            recorded_correction[0] - reconstructed[0],
            recorded_correction[1] - reconstructed[1],
        )
        current_correction_mismatch_max = max(
            current_correction_mismatch_max, correction_mismatch
        )
        current_error = math.sqrt(sum(value * value for value in position_error_xyz))
        recorded_error = float(item["position_error_m"])
        position_norm_mismatch_max = max(
            position_norm_mismatch_max, abs(current_error - recorded_error)
        )

        candidate_correction = clipped_correction(lever_error, candidate_cap_m)
        projected_xyz = [
            position_error_xyz[0]
            + candidate_correction[0]
            - recorded_correction[0],
            position_error_xyz[1]
            + candidate_correction[1]
            - recorded_correction[1],
            position_error_xyz[2],
        ]
        current_errors.append(current_error)
        projected_errors.append(
            math.sqrt(sum(value * value for value in projected_xyz))
        )

    if not projected_errors:
        raise ValueError("trace is empty")
    return {
        "sample_count": len(projected_errors),
        "current_position_p95_m": quantile(current_errors, 0.95),
        "current_position_max_m": max(current_errors),
        "projected_position_p95_m": quantile(projected_errors, 0.95),
        "projected_position_max_m": max(projected_errors),
        "projected_over_p95_gate_ratio": sum(
            value > P95_GATE_M for value in projected_errors
        )
        / len(projected_errors),
        "current_correction_reconstruction_mismatch_max_m": (
            current_correction_mismatch_max
        ),
        "position_norm_mismatch_max_m": position_norm_mismatch_max,
    }


def failed_checks(result: dict[str, object]) -> list[str]:
    return sorted(
        name for name, passed in result.get("checks", {}).items() if not passed
    )


def audit(baseline_path: Path, recovery_path: Path) -> dict[str, object]:
    _, baseline = load_result(baseline_path)
    _, recovery = load_result(recovery_path)
    baseline_projection = project_trace(baseline.get("trace", []), CANDIDATE_CAP_M)
    recovery_projection = project_trace(recovery.get("trace", []), CANDIDATE_CAP_M)
    evidence_checks = {
        "canonical_baseline_hash": sha256_file(baseline_path) == BASELINE_SHA256,
        "canonical_recovery_hash": sha256_file(recovery_path) == RECOVERY_SHA256,
        "same_case_and_clocks": baseline.get("case")
        == recovery.get("case")
        == 78
        and baseline.get("source_duration_s") == recovery.get("source_duration_s")
        and baseline.get("execution_duration_s")
        == recovery.get("execution_duration_s"),
        "both_completed_without_termination": baseline.get("termination") is None
        and recovery.get("termination") is None
        and baseline.get("completed_phase_time_s")
        == baseline.get("execution_duration_s")
        and recovery.get("completed_phase_time_s")
        == recovery.get("execution_duration_s"),
        "both_failed_only_position_p95": failed_checks(baseline)
        == failed_checks(recovery)
        == ["position_p95_bounded"],
        "current_cap_and_gain_are_frozen": baseline.get(
            "maximum_camera_lever_arm_correction_m"
        )
        == recovery.get("maximum_camera_lever_arm_correction_m")
        == CURRENT_CAP_M
        and baseline.get("camera_lever_arm_compensation_gain")
        == recovery.get("camera_lever_arm_compensation_gain")
        == 1.0,
        "trace_geometry_reconstructs": all(
            projection["current_correction_reconstruction_mismatch_max_m"]
            <= 1e-9
            and projection["position_norm_mismatch_max_m"] <= 1e-9
            for projection in (baseline_projection, recovery_projection)
        ),
        "learning_outputs_absent": all(
            result.get("executed_residual_dataset") is False
            and result.get("executed_policy_trace") is False
            and result.get("executed_raw_teacher_capture") is False
            for result in (baseline, recovery)
        ),
    }
    candidate_checks = {
        "baseline_projected_p95_has_margin": baseline_projection[
            "projected_position_p95_m"
        ]
        <= P95_GATE_M - P95_PROJECTION_MARGIN_M,
        "recovery_projected_p95_has_margin": recovery_projection[
            "projected_position_p95_m"
        ]
        <= P95_GATE_M - P95_PROJECTION_MARGIN_M,
        "baseline_projected_max_has_margin": baseline_projection[
            "projected_position_max_m"
        ]
        <= MAX_GATE_M - MAX_PROJECTION_MARGIN_M,
        "recovery_projected_max_has_margin": recovery_projection[
            "projected_position_max_m"
        ]
        <= MAX_GATE_M - MAX_PROJECTION_MARGIN_M,
        "candidate_is_single_bounded_cap_change": CANDIDATE_CAP_M
        == 2.0 * CURRENT_CAP_M,
    }
    cpu_candidate_supported = all(evidence_checks.values()) and all(
        candidate_checks.values()
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_camera_cap_candidate_v1",
        "case": 78,
        "method": (
            "ideal unit-transfer replay of the additional bounded base-target "
            "correction; this does not model closed-loop dynamics"
        ),
        "baseline_gate": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
        },
        "recovery_gate": {
            "path": str(recovery_path.resolve()),
            "sha256": sha256_file(recovery_path),
        },
        "current_cap_m": CURRENT_CAP_M,
        "candidate_cap_m": CANDIDATE_CAP_M,
        "baseline_projection": baseline_projection,
        "recovery_projection": recovery_projection,
        "evidence_checks": evidence_checks,
        "candidate_checks": candidate_checks,
        "audit_passed": all(evidence_checks.values()),
        "cpu_candidate_supported": cpu_candidate_supported,
        "proposed_runtime_delta": {
            "enable_camera_error_recovery": False,
            "maximum_camera_lever_arm_correction_m": CANDIDATE_CAP_M,
            "all_other_controller_plan_gate_and_physics_values_unchanged": True,
        },
        "dynamic_proof_obtained": False,
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
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--recovery-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.baseline_gate, args.recovery_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["audit_passed"] and result["cpu_candidate_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
