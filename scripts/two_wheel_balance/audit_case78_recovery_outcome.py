#!/usr/bin/env python3
"""Compare sealed baseline and camera-recovery case-78 outcomes by phase."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path


BASELINE_SHA256 = (
    "46ab1f27d2ed16271853e068e21497d66f6cacfb8599f98dde0c72df6d31c97a"
)
RECOVERY_SHA256 = (
    "ced834a3f0787ca11e33bc23c9134e041bd6d4ee4159249c05c0f7ef6e32eb50"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def phase_series(trace: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    phases: list[float] = []
    errors: list[float] = []
    for item in trace:
        phase = float(item["phase_time_s"])
        error = float(item["position_error_m"])
        if not math.isfinite(phase) or not math.isfinite(error):
            raise ValueError("phase trace contains a non-finite value")
        if phases and phase < phases[-1]:
            raise ValueError("phase trace is not ordered")
        if phases and phase == phases[-1]:
            errors[-1] = max(errors[-1], error)
        else:
            phases.append(phase)
            errors.append(error)
    if not phases:
        raise ValueError("phase trace is empty")
    return phases, errors


def interpolate(phases: list[float], values: list[float], target: float) -> float:
    index = bisect_right(phases, target)
    if index <= 0:
        return values[0]
    if index >= len(phases):
        return values[-1]
    left_phase = phases[index - 1]
    right_phase = phases[index]
    if right_phase <= left_phase:
        return values[index]
    fraction = (target - left_phase) / (right_phase - left_phase)
    return values[index - 1] * (1.0 - fraction) + values[index] * fraction


def load_result(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    results = gate.get("results")
    result = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    return gate, result


def audit(
    baseline_path: Path,
    recovery_path: Path,
    *,
    phase_step_s: float = 0.25,
) -> dict[str, object]:
    if not math.isfinite(phase_step_s) or phase_step_s <= 0.0:
        raise ValueError("phase step must be finite and positive")
    baseline_gate, baseline = load_result(baseline_path)
    recovery_gate, recovery = load_result(recovery_path)
    duration_s = float(baseline.get("execution_duration_s", 0.0))
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("baseline execution duration must be finite and positive")
    sample_count = math.floor(duration_s / phase_step_s) + 1
    grid = [index * phase_step_s for index in range(sample_count)]
    if not grid or grid[-1] < duration_s:
        grid.append(duration_s)
    baseline_phases, baseline_errors = phase_series(baseline.get("trace", []))
    recovery_phases, recovery_errors = phase_series(recovery.get("trace", []))
    baseline_grid = [
        interpolate(baseline_phases, baseline_errors, phase) for phase in grid
    ]
    recovery_grid = [
        interpolate(recovery_phases, recovery_errors, phase) for phase in grid
    ]
    paired_delta = [
        recovery_error - baseline_error
        for baseline_error, recovery_error in zip(baseline_grid, recovery_grid)
    ]
    baseline_failed = sorted(
        name for name, passed in baseline.get("checks", {}).items() if not passed
    )
    recovery_failed = sorted(
        name for name, passed in recovery.get("checks", {}).items() if not passed
    )
    official_p95_delta = float(recovery["position_error_p95_m"]) - float(
        baseline["position_error_p95_m"]
    )
    phase_p95_delta = quantile(recovery_grid, 0.95) - quantile(
        baseline_grid, 0.95
    )
    evidence_checks = {
        "canonical_baseline_hash": sha256_file(baseline_path) == BASELINE_SHA256,
        "canonical_recovery_hash": sha256_file(recovery_path) == RECOVERY_SHA256,
        "same_case_and_clocks": baseline.get("case")
        == recovery.get("case")
        == 78
        and baseline.get("source_duration_s") == recovery.get("source_duration_s")
        and baseline.get("execution_duration_s")
        == recovery.get("execution_duration_s"),
        "both_complete_without_termination": baseline.get(
            "completed_phase_time_s"
        )
        == duration_s
        and recovery.get("completed_phase_time_s") == duration_s
        and baseline.get("termination") is None
        and recovery.get("termination") is None,
        "same_only_failed_gate": baseline_failed
        == recovery_failed
        == ["position_p95_bounded"],
        "recovery_governor_was_active": recovery.get(
            "camera_recovery_activation_ratio", 0.0
        )
        > 0.0,
    }
    findings = {
        "official_p95_worsened": official_p95_delta > 0.0,
        "phase_aligned_p95_improved": phase_p95_delta < 0.0,
        "recovery_dynamic_gate_passed": bool(
            recovery.get("dynamic_quality_passed")
        ),
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_recovery_outcome_audit_v1",
        "case": 78,
        "baseline_gate": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
        },
        "recovery_gate": {
            "path": str(recovery_path.resolve()),
            "sha256": sha256_file(recovery_path),
        },
        "official_time_weighted": {
            "baseline_position_p95_m": baseline.get("position_error_p95_m"),
            "recovery_position_p95_m": recovery.get("position_error_p95_m"),
            "position_p95_delta_m": official_p95_delta,
            "baseline_position_max_m": baseline.get("position_error_max_m"),
            "recovery_position_max_m": recovery.get("position_error_max_m"),
            "completed_step_delta": int(recovery.get("completed_steps", 0))
            - int(baseline.get("completed_steps", 0)),
        },
        "phase_aligned": {
            "phase_step_s": phase_step_s,
            "sample_count": len(grid),
            "baseline_position_p95_m": quantile(baseline_grid, 0.95),
            "recovery_position_p95_m": quantile(recovery_grid, 0.95),
            "position_p95_delta_m": phase_p95_delta,
            "mean_paired_delta_m": sum(paired_delta) / len(paired_delta),
            "recovery_improved_sample_ratio": sum(
                delta < 0.0 for delta in paired_delta
            )
            / len(paired_delta),
            "paired_delta_max_m": max(paired_delta),
            "paired_delta_min_m": min(paired_delta),
        },
        "recovery_activation_ratio": recovery.get(
            "camera_recovery_activation_ratio"
        ),
        "evidence_checks": evidence_checks,
        "findings": findings,
        "audit_passed": all(evidence_checks.values()),
        "camera_recovery_candidate_admitted": False,
        "camera_recovery_candidate_rejected": not findings[
            "recovery_dynamic_gate_passed"
        ],
        "direct_tracking_correction_required": float(
            recovery["position_error_p95_m"]
        )
        > 0.15,
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
    parser.add_argument("--phase-step-s", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase_step_s <= 0.0:
        parser.error("phase step must be positive")
    result = audit(
        args.baseline_gate,
        args.recovery_gate,
        phase_step_s=args.phase_step_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
