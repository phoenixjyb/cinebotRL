#!/usr/bin/env python3
"""Validate an exact-source trajectory package before riser retargeting/training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CONTRACT = "exact_source_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict, expected_count: int) -> dict:
    items = manifest.get("items")
    if not isinstance(items, list):
        items = manifest.get("cases")
    if not isinstance(items, list):
        items = []

    rows = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        case = item.get("case", item.get("episode_index"))
        case = int(case) if isinstance(case, (int, float)) else -1
        source_pose_count = item.get("source_pose_count")
        source_timestamp_count = item.get("source_timestamp_count")
        waypoint_count = item.get("retargeted_waypoint_state_count")
        transition_count = item.get("transition_count")
        duration = item.get("source_duration_s", item.get("duration_s"))
        checks = {
            "case_in_range": 1 <= case <= expected_count,
            "case_unique": case not in seen,
            "positive_source_pose_count": isinstance(source_pose_count, int)
            and source_pose_count >= 2,
            "timestamp_count_preserved": source_timestamp_count
            == source_pose_count,
            "waypoint_state_count_preserved": waypoint_count == source_pose_count,
            "transition_count_exact": isinstance(source_pose_count, int)
            and transition_count == source_pose_count - 1,
            "ordered_target_geometry_preserved": item.get(
                "ordered_target_geometry_preserved"
            )
            is True,
            "source_timestamps_preserved": item.get(
                "source_timestamps_preserved"
            )
            is True,
            "initialization_separated": item.get("initialization_separated")
            is True,
            "trajectory_integrity_passed": item.get(
                "trajectory_integrity_passed"
            )
            is True,
            "quality_gate_passed": item.get("quality_gate_passed") is True,
            "valid_for_training": item.get("valid_for_training") is True,
            "duration_finite_positive": isinstance(duration, (int, float))
            and math.isfinite(float(duration))
            and float(duration) > 0.0,
        }
        if case > 0:
            seen.add(case)
        rows.append({"case": case, "checks": checks, "passed": all(checks.values())})

    expected_cases = list(range(1, expected_count + 1))
    top_checks = {
        "trajectory_integrity_contract": manifest.get(
            "trajectory_integrity_contract"
        )
        == CONTRACT,
        "package_valid_for_training": manifest.get("valid_for_training") is True,
        "package_quality_gate_passed": manifest.get("quality_gate_passed") is True,
        "declared_case_count": manifest.get("case_count") == expected_count,
        "item_count": len(rows) == expected_count,
        "contiguous_cases": sorted(seen) == expected_cases,
        "all_case_checks": len(rows) == expected_count
        and all(row["passed"] for row in rows),
    }
    return {
        "schema": "cinebotrl_riser_exact_source_admission_audit_v1",
        "trajectory_integrity_contract": CONTRACT,
        "expected_case_count": expected_count,
        "top_checks": top_checks,
        "rows": rows,
        "passed": all(top_checks.values()),
        "training_authorized": all(top_checks.values()),
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_count <= 0:
        raise ValueError("expected count must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_manifest(manifest, args.expected_count) | {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
