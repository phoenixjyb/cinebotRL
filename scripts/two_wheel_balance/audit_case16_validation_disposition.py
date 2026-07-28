#!/usr/bin/env python3
"""Classify case 16 and select an untuned corrective-validation replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
EVIDENCE_ROOT = (
    DOC_ROOT / "evidence_20260728_case16_validation_disposition_cpu_v1"
)
SCHEMA = "cinebotrl_two_wheel_riser_case16_validation_disposition_cpu_v1"
DYNAMIC_POSITION_P95_LIMIT_M = 0.15
STRONG_BASELINE_POSITION_P95_M = 0.10
MINIMUM_ABSOLUTE_IMPROVEMENT_M = 0.003
MINIMUM_RELATIVE_IMPROVEMENT = 0.02

DEFAULT_CASE16_PLAN = (
    DOC_ROOT
    / "evidence_20260724_case16_validation_pair_readiness_cpu_v1/source/"
    "case_0016_smoothed_riser_plan_v1.npz"
)
DEFAULT_CASE16_PAIR_SUMMARY = (
    DOC_ROOT
    / "evidence_20260728_case16_validation_pair_v2_rejected/summary.json"
)
DEFAULT_CASE16_PAIR_FINAL = (
    DOC_ROOT
    / "evidence_20260728_case16_validation_pair_v2_rejected/final_status.json"
)
DEFAULT_CASE22_PLAN = EVIDENCE_ROOT / "source/case_0022_smoothed_riser_plan_v1.npz"
DEFAULT_CASE22_GATE = (
    EVIDENCE_ROOT / "source/case_0022_historical_dynamic_gate.json"
)
DEFAULT_CASE32_PLAN = EVIDENCE_ROOT / "source/case_0032_smoothed_riser_plan_v1.npz"
DEFAULT_CASE32_GATE = (
    EVIDENCE_ROOT / "source/case_0032_historical_dynamic_gate.json"
)
DEFAULT_SOURCE_MANIFEST = EVIDENCE_ROOT / "source/source_manifest.json"

EXPECTED_SHA256 = {
    "case16_plan": "742d1f705d3559916c3e1d7d35caffd5ea9e7200b6e321d1f9f70c8e5a7dad16",
    "case16_pair_summary": "1e7fc0f630465b7de54949c2f687d3c77f04ef3063d125e303eca63965e6493b",
    "case16_pair_final": "89b7060366c836fd41b460b97f7cd9a58bcdfb2ae920d06ccc2ba79320b53c9d",
    "case22_plan": "8f1638cd771cfac32ca251906e2c095bd7091edb2561974f12ae09b0a65d4a79",
    "case22_gate": "115623a6f1239b9e4fc78a7a60087a176b340f275f817f123c90f593e943892a",
    "case32_plan": "71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f",
    "case32_gate": "d2a7477254d6a80426370217d8f08db8fe2bdf65e5f4b892a33247f90cf1ce75",
    "source_manifest": "11b25f51fc44c0838cf78599aa48fbed6d1d96f8b6cfbe1743c8d340e5a0c45e",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _require_hash(name: str, path: Path) -> dict[str, object]:
    digest = _sha256(path)
    if digest != EXPECTED_SHA256[name]:
        raise ValueError(
            f"{name} sha256 mismatch: expected {EXPECTED_SHA256[name]}, got {digest}"
        )
    return {
        "path": _display(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_plan(path: Path, expected_case: int) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "metadata_json",
            "time_s",
            "execution_time_s",
            "source_time_s",
            "source_anchor_execution_index",
            "target_position_world_m",
            "source_target_position_world_m",
            "feedforward_v_wz",
            "feedforward_riser_velocity",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"plan missing arrays: {sorted(missing)}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        execution_time_s = np.asarray(
            archive["execution_time_s"], dtype=np.float64
        )
        source_time_s = np.asarray(archive["source_time_s"], dtype=np.float64)
        source_anchor_map = np.asarray(
            archive["source_anchor_execution_index"], dtype=np.int64
        )
        target_position = np.asarray(
            archive["target_position_world_m"], dtype=np.float64
        )
        source_position = np.asarray(
            archive["source_target_position_world_m"], dtype=np.float64
        )
        feedforward_v_wz = np.asarray(
            archive["feedforward_v_wz"], dtype=np.float64
        )
        feedforward_riser = np.asarray(
            archive["feedforward_riser_velocity"], dtype=np.float64
        )

    source = metadata["source_provenance"]
    smoothed = metadata["smoothed_target"]
    kinematic = metadata["kinematic_metrics"]
    path_metrics = metadata["path_metrics"]
    checks = {
        "schema": metadata.get("schema")
        == "cinebotrl_two_wheel_riser_smoothed_plan_v1",
        "case": metadata.get("case") == expected_case,
        "exact_source_contract": source.get("trajectory_integrity_contract")
        == "exact_source_v1",
        "raw_source_preserved": source.get("raw_arrays_preserved_verbatim")
        is True,
        "trajectory_integrity": metadata.get("trajectory_integrity_passed")
        is True,
        "timing_transition_kinematic_gate": metadata.get(
            "timing_transition_kinematic_gate_passed"
        )
        is True,
        "source_anchor_identity": smoothed.get("source_anchor_map_identity")
        is True,
        "same_state_count": len(time_s)
        == len(source_time_s)
        == int(source["source_pose_count"]),
        "transition_count": len(feedforward_v_wz) == len(time_s) - 1,
        "riser_transition_count": len(feedforward_riser) == len(time_s) - 1,
        "execution_clock_alias": bool(
            np.array_equal(time_s, execution_time_s)
        ),
        "execution_clock_strict": bool(np.all(np.diff(time_s) > 0.0)),
        "source_clock_strict": bool(np.all(np.diff(source_time_s) > 0.0)),
        "execution_duration": bool(
            np.isclose(time_s[-1], smoothed["execution_duration_s"], atol=1e-9)
        ),
        "source_duration": bool(
            np.isclose(source_time_s[-1], source["source_duration_s"], atol=1e-9)
        ),
        "anchor_map_identity_array": bool(
            np.array_equal(
                source_anchor_map,
                np.arange(len(source_time_s), dtype=np.int64),
            )
        ),
        "target_shape": target_position.shape == (len(time_s), 3),
        "source_target_shape": source_position.shape
        == (len(source_time_s), 3),
        "path_geometry_preserved": bool(
            np.isclose(path_metrics["path_length_relative_drift"], 0.0, atol=1e-12)
        ),
        "learning_closed": metadata.get("bc_started") is False
        and metadata.get("ppo_started") is False
        and metadata.get("residual_capture_started") is False
        and metadata.get("valid_for_training") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"case {expected_case} plan checks failed: {failed}")

    return {
        "case": expected_case,
        "checks": checks,
        "state_count": len(time_s),
        "transition_count": len(time_s) - 1,
        "source_duration_s": float(source_time_s[-1]),
        "execution_duration_s": float(time_s[-1]),
        "source_path_length_m": float(source["source_path_length_m"]),
        "planning_position_p95_m": float(kinematic["position_error_p95_m"]),
        "planning_position_max_m": float(kinematic["position_error_max_m"]),
        "maximum_abs_base_linear_velocity_mps": float(
            np.max(np.abs(feedforward_v_wz[:, 0]))
        ),
        "maximum_abs_base_yaw_rate_radps": float(
            np.max(np.abs(feedforward_v_wz[:, 1]))
        ),
        "maximum_abs_riser_rate_mps": float(
            np.max(np.abs(feedforward_riser))
        ),
        "camera_height_span_m": float(
            kinematic["maximum_target_camera_height_m"]
            - kinematic["minimum_target_camera_height_m"]
        ),
    }


def _load_historical_gate(path: Path, expected_case: int) -> dict[str, object]:
    gate = _load_json(path)
    results = gate.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"case {expected_case} gate must contain one result")
    result = results[0]
    checks = {
        "schema": gate.get("schema")
        == "recomo_two_wheel_riser_reference_playback_v1",
        "case": gate.get("cases") == [expected_case]
        and result.get("case") == expected_case,
        "top_level_passed": gate.get("passed") is True,
        "dynamic_quality": gate.get("dynamic_quality_passed") is True
        and result.get("dynamic_quality_passed") is True,
        "thermal_admission": result.get("thermal_admission_passed") is True,
        "controller_evidence": result.get("controller_evidence_passed") is True,
        "completed_without_termination": result.get("termination") is None
        and result.get("passed") is True,
        "no_normalized_dataset": gate.get("normalized_dataset_capture_started")
        is False
        and result.get("executed_residual_dataset") is None,
        "training_closed": gate.get("training_started") is False
        and gate.get("ppo_authorized") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"case {expected_case} historical gate failed: {failed}")
    return {
        "case": expected_case,
        "checks": checks,
        "historical_raw_capture_present": gate.get("raw_teacher_capture_started")
        is True,
        "completed_steps": int(result["completed_steps"]),
        "position_error_p95_m": float(result["position_error_p95_m"]),
        "position_error_max_m": float(result["position_error_max_m"]),
        "attitude_error_max_deg": float(result["attitude_error_max_deg"]),
        "pitch_max_deg": float(result["pitch_max_deg"]),
        "riser_error_max_m": float(result["riser_servo_error_max_m"]),
        "action_saturation_ratio": float(result["action_saturation_ratio"]),
        "dynamic_quality_passed": True,
        "dataset_created": False,
        "training_started": False,
    }


def audit_disposition(
    case16_plan_path: Path = DEFAULT_CASE16_PLAN,
    case16_pair_summary_path: Path = DEFAULT_CASE16_PAIR_SUMMARY,
    case16_pair_final_path: Path = DEFAULT_CASE16_PAIR_FINAL,
    case22_plan_path: Path = DEFAULT_CASE22_PLAN,
    case22_gate_path: Path = DEFAULT_CASE22_GATE,
    case32_plan_path: Path = DEFAULT_CASE32_PLAN,
    case32_gate_path: Path = DEFAULT_CASE32_GATE,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, object]:
    paths = {
        "case16_plan": case16_plan_path,
        "case16_pair_summary": case16_pair_summary_path,
        "case16_pair_final": case16_pair_final_path,
        "case22_plan": case22_plan_path,
        "case22_gate": case22_gate_path,
        "case32_plan": case32_plan_path,
        "case32_gate": case32_gate_path,
        "source_manifest": source_manifest_path,
    }
    inputs = {name: _require_hash(name, path) for name, path in paths.items()}
    source_manifest = _load_json(source_manifest_path)
    if (
        source_manifest.get("selection_only") is not True
        or source_manifest.get("fresh_runtime_authorized") is not False
        or source_manifest.get("capture_authorized") is not False
        or source_manifest.get("conversion_authorized") is not False
        or source_manifest.get("training_authorized") is not False
    ):
        raise ValueError("source manifest authorization boundary drift")
    case16_plan = _load_plan(case16_plan_path, 16)
    pair_summary = _load_json(case16_pair_summary_path)
    pair_final = _load_json(case16_pair_final_path)

    pair_checks = {
        "summary_schema": pair_summary.get("schema")
        == "cinebotrl_two_wheel_riser_case16_validation_pair_v2_rejected_evidence",
        "final_schema": pair_final.get("schema")
        == (
            "cinebotrl_two_wheel_riser_corrective_teacher_case16_"
            "validation_natural_error_pair_final_v1"
        ),
        "validation_identity": pair_summary.get("case") == 16
        and pair_summary.get("split") == "validation"
        and pair_final.get("case") == 16
        and pair_final.get("split") == "validation",
        "dynamic_pair_healthy": pair_summary.get("dynamic_pair_completed") is True
        and pair_summary.get("baseline_dynamic_quality_passed") is True
        and pair_summary.get("candidate_dynamic_quality_passed") is True,
        "frozen_pair_rejected": pair_summary.get("validation_pair_passed") is False
        and pair_final["paired_admission"]["validation_pair_passed"] is False,
        "expected_failed_checks": pair_summary.get("failed_paired_checks")
        == ["minimum_position_p95_improvement", "saturation_not_regressed"],
        "absolute_gate_unchanged": bool(
            np.isclose(
                pair_summary["minimum_position_p95_improvement_m"],
                MINIMUM_ABSOLUTE_IMPROVEMENT_M,
            )
        ),
        "learning_closed": pair_summary.get("label_capture_authorized") is False
        and pair_summary.get("dataset_created") is False
        and pair_summary.get("bc_authorized") is False
        and pair_summary.get("ppo_authorized") is False
        and pair_summary.get("training_started") is False
        and pair_summary.get("valid_for_training") is False,
    }
    failed = [name for name, passed in pair_checks.items() if not passed]
    if failed:
        raise ValueError(f"case 16 pair checks failed: {failed}")

    baseline_p95 = float(pair_summary["baseline_position_error_p95_m"])
    candidate_p95 = float(pair_summary["candidate_position_error_p95_m"])
    absolute_improvement = float(
        pair_summary["position_p95_absolute_improvement_m"]
    )
    relative_improvement = float(
        pair_summary["position_p95_relative_improvement"]
    )
    projection_samples = int(
        pair_final["candidate_projection_evidence"]["sample_count"]
    )
    saturation_ratio = float(pair_summary["candidate_action_saturation_ratio"])
    estimated_saturated_samples = round(saturation_ratio * projection_samples)
    strong_baseline = baseline_p95 <= STRONG_BASELINE_POSITION_P95_M
    dynamic_margin_m = DYNAMIC_POSITION_P95_LIMIT_M - baseline_p95
    relative_gate_passed = relative_improvement >= MINIMUM_RELATIVE_IMPROVEMENT
    absolute_shortfall_m = (
        MINIMUM_ABSOLUTE_IMPROVEMENT_M - absolute_improvement
    )
    ceiling_limited = (
        strong_baseline
        and relative_gate_passed
        and 0.0 < absolute_shortfall_m < 0.001
        and pair_summary["baseline_dynamic_quality_passed"] is True
        and pair_summary["candidate_dynamic_quality_passed"] is True
    )
    if not ceiling_limited:
        raise ValueError("case 16 does not satisfy the bounded ceiling classification")

    candidates: dict[str, dict[str, object]] = {}
    for case, plan_path, gate_path in (
        (22, case22_plan_path, case22_gate_path),
        (32, case32_plan_path, case32_gate_path),
    ):
        plan = _load_plan(plan_path, case)
        gate = _load_historical_gate(gate_path, case)
        candidates[str(case)] = {
            "plan": plan,
            "historical_dynamic_gate": gate,
            "selection_checks": {
                "exact_source_plan_passed": all(plan["checks"].values()),
                "historical_dynamic_quality_passed": gate[
                    "dynamic_quality_passed"
                ],
                "position_p95_below_dynamic_limit": gate[
                    "position_error_p95_m"
                ]
                < DYNAMIC_POSITION_P95_LIMIT_M,
                "zero_action_saturation": gate["action_saturation_ratio"] == 0.0,
                "meaningful_corrective_headroom": gate[
                    "position_error_p95_m"
                ]
                > STRONG_BASELINE_POSITION_P95_M,
                "normalized_dataset_absent": gate["dataset_created"] is False,
                "training_absent": gate["training_started"] is False,
            },
        }

    case32_checks = candidates["32"]["selection_checks"]
    if not all(case32_checks.values()):
        raise ValueError("case 32 does not satisfy replacement selection checks")
    if candidates["22"]["selection_checks"]["zero_action_saturation"] is not False:
        raise ValueError("case 22 comparison no longer exercises saturation")

    return {
        "schema": SCHEMA,
        "auditor": {
            "path": _display(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
            "size_bytes": Path(__file__).stat().st_size,
        },
        "inputs": inputs,
        "thresholds": {
            "dynamic_position_p95_limit_m": DYNAMIC_POSITION_P95_LIMIT_M,
            "strong_baseline_position_p95_m": STRONG_BASELINE_POSITION_P95_M,
            "minimum_absolute_improvement_m": MINIMUM_ABSOLUTE_IMPROVEMENT_M,
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
        },
        "case16": {
            "plan": case16_plan,
            "pair_checks": pair_checks,
            "baseline_position_p95_m": baseline_p95,
            "candidate_position_p95_m": candidate_p95,
            "dynamic_position_p95_margin_m": dynamic_margin_m,
            "absolute_improvement_m": absolute_improvement,
            "relative_improvement": relative_improvement,
            "absolute_improvement_shortfall_m": absolute_shortfall_m,
            "candidate_projection_sample_count": projection_samples,
            "candidate_action_saturation_ratio": saturation_ratio,
            "estimated_saturated_action_samples": estimated_saturated_samples,
            "baseline_is_strong": strong_baseline,
            "relative_gate_passed": relative_gate_passed,
            "ceiling_limited": ceiling_limited,
            "intrinsically_hard_in_realized_dynamics": False,
            "validation_profile_selection_already_observed": True,
            "further_case_specific_tuning_recommended": False,
            "teacher_capture_recommended": False,
            "disposition": "calibration_diagnostic_only_pair_rejection_preserved",
        },
        "replacement_candidates": candidates,
        "selected_replacement_case": 32,
        "selection_reason": (
            "exact_source_historical_dynamic_pass_with_zero_action_saturation_"
            "and_meaningful_corrective_headroom"
        ),
        "selected_candidate_is_currently_admitted": False,
        "fresh_readiness_and_provenance_review_required": True,
        "next_bounded_action": (
            "cpu_only_prepare_case32_validation_pair_readiness_and_profile"
        ),
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "cpu_conversion_authorized": False,
        "dataset_merge_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case16-plan", type=Path, default=DEFAULT_CASE16_PLAN)
    parser.add_argument(
        "--case16-pair-summary",
        type=Path,
        default=DEFAULT_CASE16_PAIR_SUMMARY,
    )
    parser.add_argument(
        "--case16-pair-final", type=Path, default=DEFAULT_CASE16_PAIR_FINAL
    )
    parser.add_argument("--case22-plan", type=Path, default=DEFAULT_CASE22_PLAN)
    parser.add_argument("--case22-gate", type=Path, default=DEFAULT_CASE22_GATE)
    parser.add_argument("--case32-plan", type=Path, default=DEFAULT_CASE32_PLAN)
    parser.add_argument("--case32-gate", type=Path, default=DEFAULT_CASE32_GATE)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_disposition(
        args.case16_plan,
        args.case16_pair_summary,
        args.case16_pair_final,
        args.case22_plan,
        args.case22_gate,
        args.case32_plan,
        args.case32_gate,
        args.source_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
