#!/usr/bin/env python3
"""Audit continuous proxy-yaw branch exposure in a sealed riser portfolio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    nearest_equivalent_angle,
)


BRANCH_REFERENCE_TURNS = (-2, -1, 0, 1, 2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_nearest_branch(
    semantic_yaw: np.ndarray,
    initial_reference: float,
) -> np.ndarray:
    """Map an unwrapped semantic sequence onto one continuous physics branch."""

    mapped = np.empty_like(semantic_yaw)
    reference = float(initial_reference)
    for index, target in enumerate(semantic_yaw):
        mapped[index] = nearest_equivalent_angle(float(target), reference)
        reference = float(mapped[index])
    return mapped


def audit_case(root: Path, item: dict[str, object]) -> dict[str, object]:
    case = int(item["case"])
    name = str(item["file"])
    if Path(name).name != name:
        raise ValueError(f"case {case} plan file must be a basename")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    plan_hash = sha256_file(path)
    expected_hash = str(item["plan_sha256"])
    with np.load(path, allow_pickle=False) as arrays:
        proxy = np.asarray(arrays["proxy_gimbal_q"], dtype=np.float64)
    if proxy.ndim != 2 or proxy.shape[1] != 3 or proxy.shape[0] < 2:
        raise ValueError(f"case {case} has invalid proxy_gimbal_q shape {proxy.shape}")

    yaw = proxy[:, 2]
    finite = bool(np.isfinite(yaw).all())
    canonical = np.arctan2(np.sin(yaw), np.cos(yaw))
    branch_delta = yaw - canonical
    branch_crossings = np.flatnonzero(np.abs(np.diff(canonical)) > math.pi)
    outside_principal = np.abs(yaw) > math.pi + 1e-12
    initial_turns = round(float((yaw[0] - canonical[0]) / (2.0 * math.pi)))
    reconstructed = np.unwrap(canonical) + initial_turns * 2.0 * math.pi
    reconstruction_error = float(np.max(np.abs(reconstructed - yaw)))
    nearest = np.array(
        [
            nearest_equivalent_angle(float(target), float(reference))
            for target, reference in zip(yaw, canonical)
        ],
        dtype=np.float64,
    )
    nearest_error = float(np.max(np.abs(nearest - canonical)))
    orientation_error = float(
        max(
            np.max(np.abs(np.sin(yaw) - np.sin(canonical))),
            np.max(np.abs(np.cos(yaw) - np.cos(canonical))),
        )
    )
    stateful_trials = [
        replay_nearest_branch(
            yaw,
            canonical[0] + turns * 2.0 * math.pi,
        )
        for turns in BRANCH_REFERENCE_TURNS
    ]
    maximum_stateful_step = max(
        float(np.max(np.abs(np.diff(mapped)))) for mapped in stateful_trials
    )
    stateful_delta_error = max(
        float(np.max(np.abs(np.diff(mapped) - np.diff(yaw))))
        for mapped in stateful_trials
    )
    stateful_orientation_error = max(
        float(
            max(
                np.max(np.abs(np.sin(mapped) - np.sin(yaw))),
                np.max(np.abs(np.cos(mapped) - np.cos(yaw))),
            )
        )
        for mapped in stateful_trials
    )
    branch_margin = math.pi - maximum_stateful_step
    checks = {
        "plan_hash_matches": plan_hash == expected_hash,
        "yaw_is_finite": finite,
        "unwrapped_semantic_continuity_reconstructs": reconstruction_error <= 1e-9,
        "nearest_physics_branch_is_equivalent": nearest_error <= 1e-12,
        "orientation_is_preserved": orientation_error <= 1e-12,
        "stateful_mapping_preserves_semantic_deltas": stateful_delta_error <= 1e-9,
        "stateful_mapping_preserves_orientation": (
            stateful_orientation_error <= 1e-12
        ),
        "stateful_mapping_stays_below_half_turn": branch_margin > 0.0,
    }
    return {
        "case": case,
        "file": name,
        "plan_sha256": plan_hash,
        "semantic_yaw_min_deg": float(np.rad2deg(np.min(yaw))),
        "semantic_yaw_max_deg": float(np.rad2deg(np.max(yaw))),
        "maximum_abs_semantic_yaw_deg": float(np.max(np.abs(np.rad2deg(yaw)))),
        "outside_principal_branch_sample_count": int(np.count_nonzero(outside_principal)),
        "canonical_branch_crossing_count": int(branch_crossings.size),
        "maximum_naive_branch_delta_deg": float(np.max(np.abs(np.rad2deg(branch_delta)))),
        "maximum_unwrapped_step_deg": float(np.max(np.abs(np.diff(np.rad2deg(yaw))))),
        "semantic_reconstruction_error_rad": reconstruction_error,
        "nearest_branch_error_rad": nearest_error,
        "orientation_equivalence_error": orientation_error,
        "stateful_branch_reference_turn_trials": list(BRANCH_REFERENCE_TURNS),
        "maximum_stateful_mapped_step_deg": math.degrees(maximum_stateful_step),
        "minimum_stateful_branch_margin_deg": math.degrees(branch_margin),
        "stateful_semantic_delta_error_rad": stateful_delta_error,
        "stateful_orientation_equivalence_error": stateful_orientation_error,
        "nearest_physics_branch_required": bool(np.any(outside_principal)),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-accepted-count", type=int, default=71)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(args.manifest)
    accepted_cases = [int(case) for case in manifest["kinematic_accepted_cases"]]
    items = {int(item["case"]): item for item in manifest["items"]}
    rows = [audit_case(args.manifest.parent, items[case]) for case in accepted_cases]
    affected = [row["case"] for row in rows if row["nearest_physics_branch_required"]]
    crossing = [row["case"] for row in rows if row["canonical_branch_crossing_count"]]
    maximum_row = max(rows, key=lambda row: row["maximum_naive_branch_delta_deg"])
    maximum_step = max(row["maximum_unwrapped_step_deg"] for row in rows)
    maximum_step_cases = [
        row["case"]
        for row in rows
        if math.isclose(row["maximum_unwrapped_step_deg"], maximum_step, abs_tol=1e-12)
    ]
    minimum_stateful_margin = min(
        row["minimum_stateful_branch_margin_deg"] for row in rows
    )
    minimum_stateful_margin_cases = [
        row["case"]
        for row in rows
        if math.isclose(
            row["minimum_stateful_branch_margin_deg"],
            minimum_stateful_margin,
            abs_tol=1e-12,
        )
    ]
    checks = {
        "manifest_hash_matches": manifest_hash == args.expected_manifest_sha256,
        "accepted_count_matches": len(accepted_cases) == args.expected_accepted_count,
        "accepted_count_declaration_matches": (
            manifest.get("kinematic_accepted_count") == len(accepted_cases)
        ),
        "accepted_cases_are_unique": len(set(accepted_cases)) == len(accepted_cases),
        "all_case_audits_pass": all(row["passed"] for row in rows),
    }
    summary = {
        "schema": "cinebotrl_two_wheel_riser_continuous_yaw_scope_audit_v2",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "accepted_case_count": len(accepted_cases),
        "affected_case_count": len(affected),
        "affected_cases": affected,
        "canonical_crossing_case_count": len(crossing),
        "canonical_crossing_cases": crossing,
        "maximum_naive_branch_delta_deg": maximum_row["maximum_naive_branch_delta_deg"],
        "maximum_naive_branch_delta_case": maximum_row["case"],
        "maximum_unwrapped_step_deg": maximum_step,
        "maximum_unwrapped_step_cases": maximum_step_cases,
        "branch_reference_turn_trials": list(BRANCH_REFERENCE_TURNS),
        "minimum_stateful_branch_margin_deg": minimum_stateful_margin,
        "minimum_stateful_branch_margin_cases": minimum_stateful_margin_cases,
        "semantic_unwrapped_yaw_is_authoritative": True,
        "nearest_equivalent_physics_branch_required": True,
        "multi_turn_semantic_plans_rejected": False,
        "dynamic_quality_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "checks": checks,
        "rows": rows,
        "passed": all(checks.values()),
        "valid_for_training": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_fields = [
        "case",
        "semantic_yaw_min_deg",
        "semantic_yaw_max_deg",
        "maximum_abs_semantic_yaw_deg",
        "outside_principal_branch_sample_count",
        "canonical_branch_crossing_count",
        "maximum_naive_branch_delta_deg",
        "maximum_unwrapped_step_deg",
        "semantic_reconstruction_error_rad",
        "nearest_branch_error_rad",
        "orientation_equivalence_error",
        "maximum_stateful_mapped_step_deg",
        "minimum_stateful_branch_margin_deg",
        "stateful_semantic_delta_error_rad",
        "stateful_orientation_equivalence_error",
        "nearest_physics_branch_required",
        "passed",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})
    print(json.dumps({key: summary[key] for key in (
        "accepted_case_count",
        "affected_case_count",
        "canonical_crossing_case_count",
        "maximum_naive_branch_delta_deg",
        "maximum_naive_branch_delta_case",
        "maximum_unwrapped_step_deg",
        "maximum_unwrapped_step_cases",
        "branch_reference_turn_trials",
        "minimum_stateful_branch_margin_deg",
        "minimum_stateful_branch_margin_cases",
        "passed",
    )}, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
