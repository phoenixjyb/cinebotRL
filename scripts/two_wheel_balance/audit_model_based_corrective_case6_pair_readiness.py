#!/usr/bin/env python3
"""Audit case 6 before designing a case-specific paired canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
SCHEMA = "cinebotrl_two_wheel_riser_case6_pair_readiness_cpu_v1"
SELECTION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_pair_tranche_selection_v1"
)
PLAN_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_v1"
GATE_SCHEMA = "recomo_two_wheel_riser_reference_playback_v1"
CASE = 6
SELECTED_CASES = [30, 23, 6, 2, 7]
ELIGIBLE_VALIDATION_CASES = [8, 16, 22, 32, 78]
HOLDOUT_CASES = [3, 5, 13, 19, 24]
LIMITS = {
    "base_linear_velocity_mps": 0.4,
    "base_yaw_rate_radps": 0.4,
    "riser_rate_mps": 1.0,
    "proxy_rate_radps": 0.41887902047863906,
}
PAIR_THRESHOLDS = {
    "position_error_p95_m": 0.15,
    "position_error_max_m": 0.25,
    "pitch_max_deg": 12.0,
    "attitude_error_max_deg": 10.0,
    "riser_servo_error_max_m": 0.03,
    "action_saturation_ratio": 0.2,
}
LOW_MOTION_LIMITS = {
    "base_linear_velocity_mps": 0.3,
    "base_yaw_rate_radps": 0.3,
    "riser_rate_mps": 0.25,
    "proxy_rate_radps": 0.3,
}
MINIMUM_PROFILE_WINDOW_S = 0.1
MAXIMUM_NORMALIZED_LABEL_ABS = 0.95

DEFAULT_SELECTION = (
    DOC_ROOT / "evidence_20260723_model_based_pair_tranche_v1/selection.json"
)
DEFAULT_PLAN = (
    DOC_ROOT
    / "evidence_20260724_case6_pair_readiness_cpu_v1/source/"
    "case_0006_smoothed_riser_plan_v1.npz"
)
DEFAULT_GATE = (
    DOC_ROOT
    / "evidence_20260724_case6_pair_readiness_cpu_v1/source/"
    "case_0006_dynamic_gate.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": _display(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _closed(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("bc_authorized") is False
        and payload.get("ppo_authorized") is False
        and payload.get("training_started") is False
        and payload.get("valid_for_training") is False
    )


def _case6_selection_row(selection: Mapping[str, object]) -> dict[str, object]:
    rows = selection.get("selected_rows")
    if not isinstance(rows, list):
        raise ValueError("pair selection rows are missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("case") == CASE
    ]
    if len(matches) != 1:
        raise ValueError("pair selection must contain exactly one case-6 row")
    return matches[0]


def _load_plan(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "time_s",
                "execution_time_s",
                "feedforward_v_wz",
                "feedforward_riser_velocity",
                "feedforward_proxy_velocity",
            )
        }
    if not isinstance(metadata, dict):
        raise ValueError("case-6 plan metadata must be an object")
    return metadata, arrays


def _single_result(gate: Mapping[str, object]) -> dict[str, object]:
    results = gate.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("case-6 gate must contain exactly one result")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("case-6 gate result must be an object")
    return result


def _low_motion_windows(
    time_s: np.ndarray,
    v_wz: np.ndarray,
    riser: np.ndarray,
    proxy: np.ndarray,
) -> list[dict[str, object]]:
    mask = (
        (np.abs(v_wz[:, 0]) <= LOW_MOTION_LIMITS["base_linear_velocity_mps"])
        & (np.abs(v_wz[:, 1]) <= LOW_MOTION_LIMITS["base_yaw_rate_radps"])
        & (np.abs(riser) <= LOW_MOTION_LIMITS["riser_rate_mps"])
        & (
            np.max(np.abs(proxy), axis=1)
            <= LOW_MOTION_LIMITS["proxy_rate_radps"]
        )
    )
    windows: list[dict[str, object]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        end_of_run = start is not None and (
            not active or index == len(mask) - 1
        )
        if not end_of_run:
            continue
        end = index if active else index - 1
        duration = float(time_s[end] - time_s[start])
        if duration >= MINIMUM_PROFILE_WINDOW_S - 1e-9:
            windows.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "start_time_s": float(time_s[start]),
                    "end_time_s": float(time_s[end]),
                    "duration_s": duration,
                }
            )
        start = None
    return windows


def audit_readiness(
    selection_path: Path,
    plan_path: Path,
    gate_path: Path,
) -> dict[str, object]:
    selection = _load_object(selection_path)
    selection_row = _case6_selection_row(selection)
    plan_sha = _sha256(plan_path)
    gate_sha = _sha256(gate_path)
    selection_checks = {
        "schema": selection.get("schema") == SELECTION_SCHEMA,
        "passed": selection.get("passed") is True,
        "selected_cases": selection.get("selected_cases") == SELECTED_CASES,
        "validation_cases": selection.get("validation_cases")
        == ELIGIBLE_VALIDATION_CASES,
        "holdout_cases": selection.get("holdout_cases") == HOLDOUT_CASES,
        "case6_role": selection_row.get("selection_role")
        == "same_seed_paired_canary_required",
        "case6_checks": isinstance(selection_row.get("checks"), Mapping)
        and bool(selection_row["checks"])
        and all(value is True for value in selection_row["checks"].values()),
        "plan_hash": selection_row.get("plan_sha256") == plan_sha,
        "gate_hash": selection_row.get("dynamic_gate_sha256") == gate_sha,
        "runtime_closed": selection.get("runtime_authorized") is False
        and selection.get("gpu_launch_authorized") is False
        and selection.get("label_capture_authorized") is False,
        "learning_closed": selection.get("dataset_merge_authorized") is False
        and _closed(selection),
    }
    if not all(selection_checks.values()):
        raise ValueError(
            f"case-6 pair selection checks failed: {selection_checks}"
        )

    plan, arrays = _load_plan(plan_path)
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    execution_time_s = np.asarray(
        arrays["execution_time_s"], dtype=np.float64
    )
    v_wz = np.asarray(arrays["feedforward_v_wz"], dtype=np.float64)
    riser = np.asarray(
        arrays["feedforward_riser_velocity"], dtype=np.float64
    )
    proxy = np.asarray(
        arrays["feedforward_proxy_velocity"], dtype=np.float64
    )
    count = len(time_s)
    plan_checks = {
        "schema_case": plan.get("schema") == PLAN_SCHEMA
        and plan.get("case") == CASE,
        "integrity": plan.get("trajectory_integrity_passed") is True,
        "kinematic_gate": plan.get(
            "timing_transition_kinematic_gate_passed"
        )
        is True,
        "all_plan_checks": isinstance(plan.get("checks"), Mapping)
        and bool(plan["checks"])
        and all(value is True for value in plan["checks"].values()),
        "all_kinematic_checks": isinstance(
            plan.get("kinematic_checks"), Mapping
        )
        and bool(plan["kinematic_checks"])
        and all(value is True for value in plan["kinematic_checks"].values()),
        "training_closed": plan.get("residual_capture_started") is False
        and plan.get("bc_started") is False
        and plan.get("ppo_started") is False
        and plan.get("valid_for_training") is False,
        "array_shapes": count > 1
        and execution_time_s.shape == (count,)
        and v_wz.shape == (count - 1, 2)
        and riser.shape == (count - 1,)
        and proxy.shape == (count - 1, 3),
        "arrays_finite": bool(
            all(
                np.isfinite(value).all()
                for value in (time_s, execution_time_s, v_wz, riser, proxy)
            )
        ),
        "clocks": bool(
            abs(float(time_s[0])) <= 1e-9
            and np.all(np.diff(time_s) > 0.0)
            and np.all(np.diff(execution_time_s) > 0.0)
        ),
    }
    if not all(plan_checks.values()):
        raise ValueError(f"case-6 plan checks failed: {plan_checks}")

    gate = _load_object(gate_path)
    result = _single_result(gate)
    normalized_labels = np.asarray(
        result.get("normalized_residual_label_abs_max"), dtype=np.float64
    )
    residual_actions = np.asarray(
        result.get("residual_action_abs_max"), dtype=np.float64
    )
    gate_checks = {
        "schema_case": gate.get("schema") == GATE_SCHEMA
        and gate.get("cases") == [CASE]
        and result.get("case") == CASE,
        "passed": gate.get("passed") is True
        and gate.get("dynamic_quality_passed") is True
        and gate.get("thermal_admission_passed") is True
        and gate.get("controller_evidence_passed") is True,
        "result_passed": result.get("dynamic_quality_passed") is True
        and result.get("thermal_admission_passed") is True
        and result.get("controller_evidence_passed") is True,
        "no_termination": result.get("termination") is None,
        "zero_residual_action": bool(
            residual_actions.shape == (3,)
            and np.allclose(residual_actions, 0.0, rtol=0.0, atol=1e-12)
        ),
        "raw_label_not_applied": result.get(
            "raw_residual_label_applied_to_commands"
        )
        is False,
        "label_margin": bool(
            normalized_labels.shape == (3,)
            and np.isfinite(normalized_labels).all()
            and float(np.max(normalized_labels))
            < MAXIMUM_NORMALIZED_LABEL_ABS
        ),
        "non_training_playback": bool(
            gate.get("ppo_authorized") is False
            and gate.get("training_started") is False
            and residual_actions.shape == (3,)
            and np.allclose(residual_actions, 0.0, rtol=0.0, atol=1e-12)
            and result.get("raw_residual_label_applied_to_commands") is False
        ),
    }
    if not all(gate_checks.values()):
        raise ValueError(f"case-6 dynamic gate checks failed: {gate_checks}")

    kinematics = plan["kinematic_metrics"]
    maximums = {
        "base_linear_velocity_mps": float(np.max(np.abs(v_wz[:, 0]))),
        "base_yaw_rate_radps": float(np.max(np.abs(v_wz[:, 1]))),
        "riser_rate_mps": float(np.max(np.abs(riser))),
        "proxy_rate_radps": float(np.max(np.abs(proxy))),
    }
    headroom = {
        name: max(0.0, LIMITS[name] - value)
        for name, value in maximums.items()
    }
    metric_checks = {
        "base_linear": bool(
            np.isclose(
                maximums["base_linear_velocity_mps"],
                float(kinematics["maximum_abs_base_linear_velocity_mps"]),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "base_yaw": bool(
            np.isclose(
                maximums["base_yaw_rate_radps"],
                float(kinematics["maximum_abs_base_yaw_rate_radps"]),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "riser": bool(
            np.isclose(
                maximums["riser_rate_mps"],
                float(kinematics["maximum_abs_riser_rate_mps"]),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "proxy": bool(
            np.isclose(
                maximums["proxy_rate_radps"],
                float(
                    kinematics["maximum_abs_raw_proxy_target_rate_radps"]
                ),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "camera_height": float(kinematics["minimum_target_camera_height_m"])
        >= 0.6 - 1e-9
        and float(kinematics["maximum_target_camera_height_m"]) <= 1.8 + 1e-9,
    }
    if not all(metric_checks.values()):
        raise ValueError(f"case-6 metric checks failed: {metric_checks}")

    dynamic_margins = {
        name: PAIR_THRESHOLDS[name] - float(result[name])
        for name in PAIR_THRESHOLDS
    }
    label_headroom = (
        MAXIMUM_NORMALIZED_LABEL_ABS - normalized_labels
    ).tolist()
    lever_saturation = float(
        result["camera_lever_arm_correction_saturation_ratio"]
    )
    windows = _low_motion_windows(time_s[:-1], v_wz, riser, proxy)
    case_specific_profile_required = (
        headroom["base_linear_velocity_mps"] <= 1e-6
        or headroom["base_yaw_rate_radps"] <= 1e-6
        or headroom["proxy_rate_radps"] <= 1e-6
        or lever_saturation > 0.9
    )
    return {
        "schema": SCHEMA,
        "case": CASE,
        "selection_checks": selection_checks,
        "plan_checks": plan_checks,
        "gate_checks": gate_checks,
        "metric_checks": metric_checks,
        "inputs": {
            "selection": _identity(selection_path),
            "plan": _identity(plan_path),
            "dynamic_gate": _identity(gate_path),
        },
        "plan": {
            "sample_count": count,
            "transition_count": count - 1,
            "source_duration_s": float(
                plan["source_provenance"]["source_duration_s"]
            ),
            "execution_duration_s": float(time_s[-1]),
            "minimum_target_camera_height_m": float(
                kinematics["minimum_target_camera_height_m"]
            ),
            "maximum_target_camera_height_m": float(
                kinematics["maximum_target_camera_height_m"]
            ),
            "maximums": maximums,
            "headroom": headroom,
        },
        "zero_residual_dynamic_gate": {
            "position_error_p95_m": float(result["position_error_p95_m"]),
            "position_error_max_m": float(result["position_error_max_m"]),
            "pitch_max_deg": float(result["pitch_max_deg"]),
            "camera_lever_arm_correction_saturation_ratio": lever_saturation,
            "normalized_residual_label_abs_max": normalized_labels.tolist(),
            "normalized_residual_label_headroom": label_headroom,
            "dynamic_margins": dynamic_margins,
        },
        "profile_window_contract": {
            "limits": LOW_MOTION_LIMITS,
            "minimum_duration_s": MINIMUM_PROFILE_WINDOW_S,
            "windows": windows,
            "bounded_window_found": bool(windows),
        },
        "case_specific_profile_required": case_specific_profile_required,
        "case23_profile_reuse_authorized": False,
        "pair_profile_cpu_ready": False,
        "next_bounded_action": (
            "design_case6_specific_corrective_and_perturbation_profiles_cpu_only"
        ),
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dynamic-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_readiness(args.selection, args.plan, args.dynamic_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
