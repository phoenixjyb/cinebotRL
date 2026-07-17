#!/usr/bin/env python3
"""Audit reverse-recovery steering direction in a sealed Gate C trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    riser_tracking_config,
    wrap_to_pi,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def reconstruct_row(row: dict[str, object]) -> dict[str, float | bool]:
    config = riser_tracking_config()
    target = [float(value) for value in row["target_base_xy_yaw"]]
    actual = [float(value) for value in row["actual_base_xy_yaw"]]
    delta_x = target[0] - actual[0]
    delta_y = target[1] - actual[1]
    cosine = math.cos(actual[2])
    sine = math.sin(actual[2])
    along_error = cosine * delta_x + sine * delta_y
    cross_error = -sine * delta_x + cosine * delta_y
    yaw_error = wrap_to_pi(target[2] - actual[2])
    feedforward_v = float(row["phase_feedforward_v_mps"])
    feedforward_wz = float(row["phase_feedforward_wz_rad_s"])
    raw_velocity = feedforward_v + config.along_track_kp * along_error
    velocity = clamp(raw_velocity, config.maximum_linear_velocity_mps)
    legacy_direction = 1.0 if feedforward_v >= 0.0 else -1.0
    motion_direction = clamp(
        velocity / config.direction_blend_speed_mps,
        1.0,
    )
    legacy_yaw_rate = clamp(
        feedforward_wz
        + config.yaw_kp * yaw_error
        + config.cross_track_kp * legacy_direction * cross_error,
        config.maximum_yaw_rate_radps,
    )
    candidate_yaw_rate = clamp(
        feedforward_wz
        + config.yaw_kp * yaw_error
        + config.cross_track_kp * motion_direction * cross_error,
        config.maximum_yaw_rate_radps,
    )
    return {
        "along_error_m": along_error,
        "cross_error_m": cross_error,
        "yaw_error_rad": yaw_error,
        "reconstructed_velocity_mps": velocity,
        "legacy_yaw_rate_rad_s": legacy_yaw_rate,
        "candidate_yaw_rate_rad_s": candidate_yaw_rate,
        "legacy_direction": legacy_direction,
        "motion_direction": motion_direction,
        "direction_conflict": feedforward_v * velocity < 0.0,
    }


def audit(
    case_json: Path,
    *,
    expected_case_sha256: str,
    expected_case: int,
    position_gate_m: float,
    reconstruction_tolerance: float,
    motion_deadband_mps: float,
) -> dict[str, object]:
    case_hash = sha256_file(case_json)
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if len(results) != 1:
        raise ValueError("expected exactly one Gate C result")
    result = results[0]
    trace = list(result.get("trace", []))
    if len(trace) < 2:
        raise ValueError("Gate C trace must contain at least two samples")

    reconstructed = [reconstruct_row(row) for row in trace]
    velocity_reconstruction_error = max(
        abs(float(row["vx_reference_mps"]) - float(item["reconstructed_velocity_mps"]))
        for row, item in zip(trace, reconstructed)
    )
    yaw_reconstruction_error = max(
        abs(float(row["wz_reference_rad_s"]) - float(item["legacy_yaw_rate_rad_s"]))
        for row, item in zip(trace, reconstructed)
    )
    conflict_indices = [
        index
        for index, (row, item) in enumerate(zip(trace, reconstructed))
        if abs(float(row["phase_feedforward_v_mps"])) > motion_deadband_mps
        and abs(float(item["reconstructed_velocity_mps"])) > motion_deadband_mps
        and item["direction_conflict"]
    ]
    bad_position_indices = [
        index
        for index, row in enumerate(trace)
        if float(row["position_error_m"]) > position_gate_m
    ]
    bad_conflict_indices = sorted(set(conflict_indices).intersection(bad_position_indices))
    peak_index = max(
        range(len(trace)), key=lambda index: float(trace[index]["position_error_m"])
    )
    peak_row = trace[peak_index]
    peak_reconstruction = reconstructed[peak_index]
    candidate_saturation_count = sum(
        math.isclose(
            abs(float(item["candidate_yaw_rate_rad_s"])),
            riser_tracking_config().maximum_yaw_rate_radps,
            abs_tol=1e-9,
        )
        for item in reconstructed
    )
    legacy_saturation_count = sum(
        math.isclose(
            abs(float(item["legacy_yaw_rate_rad_s"])),
            riser_tracking_config().maximum_yaw_rate_radps,
            abs_tol=1e-9,
        )
        for item in reconstructed
    )
    checks = {
        "case_hash_matches": case_hash == expected_case_sha256,
        "case_matches": int(result.get("case", -1)) == expected_case,
        "dynamic_quality_rejected": result.get("dynamic_quality_passed") is False,
        "reference_incomplete": result["checks"]["completed_reference"] is False,
        "position_gate_failed": (
            result["checks"]["position_p95_bounded"] is False
            and result["checks"]["position_max_bounded"] is False
        ),
        "yaw_repair_physically_bounded": (
            result["checks"]["proxy_servo_error_bounded"] is True
            and result["checks"]["proxy_saturation_bounded"] is True
            and result["checks"]["no_termination"] is True
        ),
        "no_dataset_created": result.get("executed_residual_dataset") is None,
        "legacy_reference_reconstruction_matches": max(
            velocity_reconstruction_error, yaw_reconstruction_error
        )
        <= reconstruction_tolerance,
        "direction_conflict_exists": bool(conflict_indices),
        "direction_conflict_dominates_bad_position_samples": (
            len(bad_conflict_indices) > len(bad_position_indices) / 2.0
        ),
        "peak_position_error_has_direction_conflict": bool(
            peak_reconstruction["direction_conflict"]
        ),
        "motion_direction_candidate_changes_peak_steering": abs(
            float(peak_reconstruction["candidate_yaw_rate_rad_s"])
            - float(peak_row["wz_reference_rad_s"])
        )
        >= 0.2,
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_motion_direction_recovery_audit_v1",
        "case": expected_case,
        "case_json": str(case_json.resolve()),
        "case_json_sha256": case_hash,
        "trace_sample_count": len(trace),
        "position_gate_m": position_gate_m,
        "motion_deadband_mps": motion_deadband_mps,
        "velocity_reconstruction_error_max_mps": velocity_reconstruction_error,
        "yaw_reconstruction_error_max_rad_s": yaw_reconstruction_error,
        "direction_conflict_sample_count": len(conflict_indices),
        "bad_position_sample_count": len(bad_position_indices),
        "bad_position_direction_conflict_sample_count": len(bad_conflict_indices),
        "legacy_yaw_saturation_sample_count": legacy_saturation_count,
        "candidate_yaw_saturation_sample_count": candidate_saturation_count,
        "peak_position_error": {
            "step": int(peak_row["step"]),
            "elapsed_s": float(peak_row["elapsed_s"]),
            "phase_time_s": float(peak_row["phase_time_s"]),
            "position_error_m": float(peak_row["position_error_m"]),
            "base_xy_error_m": float(peak_row["base_xy_error_m"]),
            "feedforward_v_mps": float(peak_row["phase_feedforward_v_mps"]),
            "velocity_reference_mps": float(peak_row["vx_reference_mps"]),
            "legacy_yaw_rate_rad_s": float(peak_row["wz_reference_rad_s"]),
            "candidate_yaw_rate_rad_s": float(
                peak_reconstruction["candidate_yaw_rate_rad_s"]
            ),
            "cross_track_error_m": float(peak_reconstruction["cross_error_m"]),
            "yaw_error_rad": float(peak_reconstruction["yaw_error_rad"]),
            "direction_conflict": bool(peak_reconstruction["direction_conflict"]),
        },
        "diagnosis": (
            "feedforward_sign_cross_track_cancellation_during_feedback_reverse"
        ),
        "candidate": "motion_command_direction_with_zero_speed_blend",
        "candidate_dynamically_validated": False,
        "thresholds_relaxed": False,
        "runtime_result_modified": False,
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
    parser.add_argument("--expected-case", type=int, default=74)
    parser.add_argument("--position-gate-m", type=float, default=0.25)
    parser.add_argument("--reconstruction-tolerance", type=float, default=0.01)
    parser.add_argument("--motion-deadband-mps", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.position_gate_m,
        args.reconstruction_tolerance,
        args.motion_deadband_mps,
    ) <= 0.0:
        raise ValueError("audit thresholds must be positive")
    summary = audit(
        args.case_json,
        expected_case_sha256=args.expected_case_sha256,
        expected_case=args.expected_case,
        position_gate_m=args.position_gate_m,
        reconstruction_tolerance=args.reconstruction_tolerance,
        motion_deadband_mps=args.motion_deadband_mps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
