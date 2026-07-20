#!/usr/bin/env python3
"""Audit case-42 plan lineage before reverting a rejected preview allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_ARRAYS = (
    "source_time_s",
    "source_target_position_world_m",
    "source_target_semantic_dfr_quat_xyzw",
    "source_anchor_execution_index",
    "target_position_world_m",
    "target_semantic_dfr_quat_wxyz",
    "smoothed_target_position_source_frame_m",
)
EXECUTION_GEOMETRY = (
    *SOURCE_ARRAYS,
    "base_xy_yaw",
    "riser_q",
    "proxy_gimbal_q",
    "initialization_time_s",
    "initialization_state",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path, expected_sha256: str) -> dict[str, object]:
    require(path.is_file(), f"missing JSON: {path}")
    require(sha256_file(path) == expected_sha256, f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def load_plan(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing plan: {path}")
    require(sha256_file(path) == expected_sha256, f"hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        plan = {name: archive[name] for name in archive.files}
    required = {
        *EXECUTION_GEOMETRY,
        "time_s",
        "execution_time_s",
        "feedforward_v_wz",
        "feedforward_riser_velocity",
        "feedforward_proxy_velocity",
    }
    require(required.issubset(plan), f"plan arrays are incomplete: {path}")
    require(
        np.array_equal(plan["time_s"], plan["execution_time_s"]),
        f"execution clock aliases differ: {path}",
    )
    return plan


def result_from_gate(payload: dict[str, object], expected_case: int) -> dict[str, object]:
    results = payload.get("results")
    require(isinstance(results, list) and len(results) == 1, "expected one result")
    result = results[0]
    require(isinstance(result, dict), "gate result is not an object")
    require(result.get("case") == expected_case, "gate case mismatch")
    require(payload.get("training_started") is False, "gate started training")
    require(payload.get("ppo_authorized") is False, "gate authorized PPO")
    require(result.get("executed_residual_dataset") is None, "gate wrote a dataset")
    require(
        result.get("raw_residual_label_applied_to_commands") is False,
        "gate applied a prospective residual label",
    )
    return result


def manifest_item(payload: dict[str, object], expected_case: int) -> dict[str, object]:
    items = payload.get("items")
    require(isinstance(items, list), "baseline manifest has no items")
    rows = [item for item in items if item.get("case") == expected_case]
    require(len(rows) == 1 and isinstance(rows[0], dict), "baseline item mismatch")
    return rows[0]


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    require(left.shape == right.shape, "array shape mismatch")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def audit(args: argparse.Namespace) -> dict[str, object]:
    baseline = load_plan(args.baseline_plan, args.expected_baseline_plan_sha256)
    preview = load_plan(args.preview_plan, args.expected_preview_plan_sha256)
    retimed = load_plan(args.retimed_plan, args.expected_retimed_plan_sha256)
    manifest = load_json(
        args.baseline_manifest, args.expected_baseline_manifest_sha256
    )
    preview_gate = load_json(args.preview_gate, args.expected_preview_gate_sha256)
    retimed_gate = load_json(args.retimed_gate, args.expected_retimed_gate_sha256)
    baseline_item = manifest_item(manifest, args.expected_case)
    preview_result = result_from_gate(preview_gate, args.expected_case)
    retimed_result = result_from_gate(retimed_gate, args.expected_case)

    source_equal = {
        name: (
            np.array_equal(baseline[name], preview[name])
            and np.array_equal(preview[name], retimed[name])
        )
        for name in SOURCE_ARRAYS
    }
    preview_retime_geometry_equal = {
        name: np.array_equal(preview[name], retimed[name])
        for name in EXECUTION_GEOMETRY
    }
    baseline_preview_allocation_delta = {
        name: max_abs_delta(baseline[name], preview[name])
        for name in ("base_xy_yaw", "riser_q", "proxy_gimbal_q")
    }
    checks = {
        "authoritative_source_arrays_equal": all(source_equal.values()),
        "preview_retime_execution_geometry_equal": all(
            preview_retime_geometry_equal.values()
        ),
        "preview_retime_clock_changed": not np.array_equal(
            preview["time_s"], retimed["time_s"]
        ),
        "preview_retime_feedforward_changed": not np.array_equal(
            preview["feedforward_v_wz"], retimed["feedforward_v_wz"]
        ),
        "baseline_allocation_differs_from_preview": max(
            baseline_preview_allocation_delta.values()
        )
        > 0.0,
        "baseline_plan_hash_matches_manifest": (
            baseline_item.get("plan_sha256")
            == args.expected_baseline_plan_sha256
        ),
        "baseline_static_admission_passed": (
            baseline_item.get("passed") is True
            and baseline_item.get("timing_transition_kinematic_gate_passed") is True
            and all(baseline_item.get("checks", {}).values())
            and all(baseline_item.get("kinematic_checks", {}).values())
        ),
        "baseline_training_closed": baseline_item.get("valid_for_training") is False,
        "preview_completed_but_dynamic_rejected": (
            preview_result.get("checks", {}).get("completed_reference") is True
            and preview_result.get("dynamic_quality_passed") is False
        ),
        "retime_incomplete_and_dynamic_rejected": (
            retimed_result.get("checks", {}).get("completed_reference") is False
            and retimed_result.get("dynamic_quality_passed") is False
        ),
        "retime_worsened_position_p95": float(
            retimed_result["position_error_p95_m"]
        )
        > float(preview_result["position_error_p95_m"]),
        "retime_worsened_position_max": float(
            retimed_result["position_error_max_m"]
        )
        > float(preview_result["position_error_max_m"]),
        "no_gate_dataset_or_training": True,
    }
    require(all(checks.values()), "case-42 lineage audit failed")
    return {
        "schema": "cinebotrl_two_wheel_riser_case42_plan_lineage_audit_v1",
        "case": args.expected_case,
        "checks": checks,
        "source_array_equality": source_equal,
        "preview_retime_execution_geometry_equality": (
            preview_retime_geometry_equal
        ),
        "baseline_preview_allocation_abs_max": (
            baseline_preview_allocation_delta
        ),
        "baseline_candidate": {
            "plan": str(args.baseline_plan.resolve()),
            "plan_sha256": args.expected_baseline_plan_sha256,
            "source_duration_s": float(baseline["source_time_s"][-1]),
            "execution_duration_s": float(baseline["time_s"][-1]),
            "static_position_p95_m": float(
                baseline_item["kinematic_metrics"]["position_error_p95_m"]
            ),
            "static_position_max_m": float(
                baseline_item["kinematic_metrics"]["position_error_max_m"]
            ),
            "dynamically_validated": False,
        },
        "preview_reject": {
            "gate_sha256": args.expected_preview_gate_sha256,
            "completed_phase_time_s": float(
                preview_result["completed_phase_time_s"]
            ),
            "execution_duration_s": float(preview_result["execution_duration_s"]),
            "position_error_p95_m": float(preview_result["position_error_p95_m"]),
            "position_error_max_m": float(preview_result["position_error_max_m"]),
        },
        "retime_reject": {
            "gate_sha256": args.expected_retimed_gate_sha256,
            "completed_phase_time_s": float(
                retimed_result["completed_phase_time_s"]
            ),
            "execution_duration_s": float(retimed_result["execution_duration_s"]),
            "maximum_runtime_s": float(retimed_result["maximum_runtime_s"]),
            "position_error_p95_m": float(retimed_result["position_error_p95_m"]),
            "position_error_max_m": float(retimed_result["position_error_max_m"]),
        },
        "diagnosis": "preview_allocation_rejected_and_local_retime_amplified_first_reversal_lag",
        "recommended_next_candidate": "preexisting_static_admitted_baseline_plan",
        "controller_changed": False,
        "thresholds_changed": False,
        "source_geometry_changed": False,
        "runtime_authorized": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--expected-baseline-plan-sha256", required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--expected-baseline-manifest-sha256", required=True)
    parser.add_argument("--preview-plan", type=Path, required=True)
    parser.add_argument("--expected-preview-plan-sha256", required=True)
    parser.add_argument("--preview-gate", type=Path, required=True)
    parser.add_argument("--expected-preview-gate-sha256", required=True)
    parser.add_argument("--retimed-plan", type=Path, required=True)
    parser.add_argument("--expected-retimed-plan-sha256", required=True)
    parser.add_argument("--retimed-gate", type=Path, required=True)
    parser.add_argument("--expected-retimed-gate-sha256", required=True)
    parser.add_argument("--expected-case", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh output path already exists")
    payload = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
