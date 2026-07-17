#!/usr/bin/env python3
"""Prove a recovery steering candidate preserves sealed healthy Gate C traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    bounded_base_references,
    riser_tracking_config,
    wrap_to_pi,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_reference(row: dict[str, object]) -> tuple[float, float]:
    config = riser_tracking_config()
    target = np.asarray(row["target_base_xy_yaw"], dtype=np.float64)
    actual = np.asarray(row["actual_base_xy_yaw"], dtype=np.float64)
    delta = target[:2] - actual[:2]
    cosine = math.cos(float(actual[2]))
    sine = math.sin(float(actual[2]))
    along_error = cosine * delta[0] + sine * delta[1]
    cross_error = -sine * delta[0] + cosine * delta[1]
    yaw_error = wrap_to_pi(float(target[2] - actual[2]))
    feedforward_v = float(row["phase_feedforward_v_mps"])
    direction = 1.0 if feedforward_v >= 0.0 else -1.0
    velocity = float(
        np.clip(
            feedforward_v + config.along_track_kp * along_error,
            -config.maximum_linear_velocity_mps,
            config.maximum_linear_velocity_mps,
        )
    )
    yaw_rate = float(
        np.clip(
            float(row["phase_feedforward_wz_rad_s"])
            + config.yaw_kp * yaw_error
            + config.cross_track_kp * direction * cross_error,
            -config.maximum_yaw_rate_radps,
            config.maximum_yaw_rate_radps,
        )
    )
    return velocity, yaw_rate


def audit_case(
    case_json: Path,
    expected_sha256: str,
    reconstruction_tolerance: float,
) -> dict[str, object]:
    case_hash = sha256_file(case_json)
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if len(results) != 1:
        raise ValueError("expected one Gate C result per healthy case JSON")
    result = results[0]
    trace = list(result.get("trace", []))
    if len(trace) < 2:
        raise ValueError("healthy Gate C trace must contain at least two samples")

    config = riser_tracking_config()
    legacy_rows = [legacy_reference(row) for row in trace]
    candidate_rows = [
        bounded_base_references(
            np.asarray(row["target_base_xy_yaw"], dtype=np.float64),
            np.asarray(row["actual_base_xy_yaw"], dtype=np.float64),
            float(row["phase_feedforward_v_mps"]),
            float(row["phase_feedforward_wz_rad_s"]),
            config,
        )
        for row in trace
    ]
    reconstruction_error = max(
        max(
            abs(float(row["vx_reference_mps"]) - legacy[0]),
            abs(float(row["wz_reference_rad_s"]) - legacy[1]),
        )
        for row, legacy in zip(trace, legacy_rows)
    )
    candidate_delta = [
        max(abs(candidate[0] - legacy[0]), abs(candidate[1] - legacy[1]))
        for candidate, legacy in zip(candidate_rows, legacy_rows)
    ]
    active_count = sum(
        candidate[2]["direction_recovery_blend"] > 0.0
        for candidate in candidate_rows
    )
    checks = {
        "case_hash_matches": case_hash == expected_sha256,
        "sealed_result_passed": payload.get("passed") is True
        and result.get("passed") is True,
        "all_original_checks_passed": all(result.get("checks", {}).values()),
        "no_dataset_created": result.get("executed_residual_dataset") is None,
        "legacy_reference_reconstruction_matches": (
            reconstruction_error <= reconstruction_tolerance
        ),
        "recovery_gate_inactive": active_count == 0,
        "candidate_preserves_legacy_commands": max(candidate_delta) <= 1e-12,
    }
    return {
        "case": int(result["case"]),
        "case_json": str(case_json.resolve()),
        "case_json_sha256": case_hash,
        "trace_sample_count": len(trace),
        "peak_base_xy_error_m": float(result["peak_base_xy_error_m"]),
        "legacy_reference_reconstruction_error_max": reconstruction_error,
        "recovery_active_sample_count": active_count,
        "candidate_command_delta_max": max(candidate_delta),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", action="append", type=Path, required=True)
    parser.add_argument("--expected-sha256", action="append", required=True)
    parser.add_argument("--reconstruction-tolerance", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case_json) != len(args.expected_sha256):
        raise ValueError("case JSON and expected hash counts differ")
    if args.reconstruction_tolerance <= 0.0:
        raise ValueError("reconstruction tolerance must be positive")
    rows = [
        audit_case(path, expected, args.reconstruction_tolerance)
        for path, expected in zip(args.case_json, args.expected_sha256)
    ]
    cases = [row["case"] for row in rows]
    checks = {
        "cases_are_unique": len(cases) == len(set(cases)),
        "all_healthy_case_audits_pass": all(row["passed"] for row in rows),
    }
    summary = {
        "schema": "cinebotrl_two_wheel_riser_motion_direction_compatibility_v1",
        "candidate": "recovery_gated_motion_command_direction",
        "recovery_error_range_m": [
            riser_tracking_config().direction_recovery_error_start_m,
            riser_tracking_config().direction_recovery_error_full_m,
        ],
        "healthy_cases": cases,
        "healthy_case_count": len(rows),
        "rows": rows,
        "candidate_dynamically_validated": False,
        "gpu_work_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
