#!/usr/bin/env python3
"""Validate exact-source reference ingest or downstream training admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONTRACT = "exact_source_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _items(manifest: dict) -> list[dict]:
    items = manifest.get("items", manifest.get("cases", []))
    return items if isinstance(items, list) else []


def _case(item: dict) -> int:
    value = item.get("case", item.get("episode_index"))
    return int(value) if isinstance(value, (int, float)) else -1


def _load_source_poses(
    path: Path,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload.get("poses")
    if not isinstance(poses, list) or len(poses) < 2:
        raise ValueError(f"invalid source poses: {path}")
    position = [pose.get("position") for pose in poses]
    orientation = [pose.get("orientation") for pose in poses]
    time_s = [pose.get("time") for pose in poses]
    return position, orientation, time_s


def _finite_rows(rows: list[list[float]], width: int) -> bool:
    return all(
        isinstance(row, list)
        and len(row) == width
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in row)
        for row in rows
    )


def validate_reference_ingest(
    manifest: dict, manifest_path: Path, expected_count: int
) -> dict:
    rows = []
    seen: set[int] = set()
    for item in _items(manifest):
        case = _case(item)
        source = manifest_path.parent / str(item.get("bundled_source_json", ""))
        checks = {
            "case_in_range": 1 <= case <= expected_count,
            "case_unique": case not in seen,
            "item_contract": item.get("trajectory_integrity_contract") == CONTRACT,
            "item_integrity_passed": item.get("integrity_passed") is True,
            "item_quality_not_claimed": item.get("quality_qualified_teacher")
            is False,
            "item_not_training_qualified": item.get("valid_for_training") is False,
            "source_file_exists": source.is_file(),
            "source_hash_declared": isinstance(item.get("source_json_sha256"), str),
        }
        if source.is_file():
            checks["source_hash_matches"] = sha256(source) == item.get(
                "source_json_sha256"
            )
            try:
                positions, quaternions_xyzw, source_time_s = _load_source_poses(source)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                positions = []
                quaternions_xyzw = []
                source_time_s = []
            count = len(source_time_s)
            positions_valid = _finite_rows(positions, 3)
            quaternions_valid = _finite_rows(quaternions_xyzw, 4)
            times_valid = all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in source_time_s
            )
            checks.update(
                {
                    "source_pose_count": count == item.get("source_pose_count"),
                    "source_position_shape": len(positions) == count
                    and positions_valid,
                    "source_quaternion_shape": len(quaternions_xyzw) == count
                    and quaternions_valid,
                    "source_values_finite": bool(
                        count >= 2
                        and positions_valid
                        and quaternions_valid
                        and times_valid
                    ),
                    "source_time_preserved": count >= 2
                    and abs(float(source_time_s[0])) <= 1e-12
                    and all(
                        current > previous
                        for previous, current in zip(
                            source_time_s, source_time_s[1:]
                        )
                    )
                    and abs(
                        float(source_time_s[-1])
                        - float(item.get("source_duration_s", -1.0))
                    )
                    <= 1e-9,
                    "source_quaternions_normalized": count >= 2
                    and quaternions_valid
                    and all(
                        abs(math.sqrt(sum(value * value for value in row)) - 1.0)
                        <= 1e-10
                        for row in quaternions_xyzw
                    ),
                }
            )
        if case > 0:
            seen.add(case)
        rows.append(
            {
                "case": case,
                "source_json": str(source),
                "source_json_sha256": item.get("source_json_sha256"),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    top_checks = {
        "trajectory_integrity_contract": manifest.get(
            "trajectory_integrity_contract"
        )
        == CONTRACT,
        "package_integrity_passed": manifest.get("integrity_passed") is True,
        "package_quality_not_claimed": manifest.get("quality_qualified_teacher")
        is False,
        "package_not_training_qualified": manifest.get("valid_for_training")
        is False,
        "declared_case_count": manifest.get(
            "episode_count", manifest.get("case_count")
        )
        == expected_count,
        "item_count": len(rows) == expected_count,
        "contiguous_cases": sorted(seen) == list(range(1, expected_count + 1)),
        "all_source_checks": len(rows) == expected_count
        and all(row["passed"] for row in rows),
    }
    passed = all(top_checks.values())
    return {
        "schema": "cinebotrl_riser_exact_source_reference_ingest_audit_v1",
        "mode": "reference_ingest",
        "trajectory_integrity_contract": CONTRACT,
        "expected_case_count": expected_count,
        "top_checks": top_checks,
        "rows": rows,
        "passed": passed,
        "reference_ingest_authorized": passed,
        "training_authorized": False,
        "ppo_authorized": False,
    }


def validate_training_manifest(manifest: dict, expected_count: int) -> dict:
    rows = []
    seen: set[int] = set()
    for item in _items(manifest):
        case = _case(item)
        source_count = item.get("source_pose_count")
        duration = item.get("source_duration_s", item.get("duration_s"))
        checks = {
            "case_in_range": 1 <= case <= expected_count,
            "case_unique": case not in seen,
            "positive_source_pose_count": isinstance(source_count, int)
            and source_count >= 2,
            "timestamp_count_preserved": item.get("source_timestamp_count")
            == source_count,
            "waypoint_state_count_preserved": item.get(
                "retargeted_waypoint_state_count"
            )
            == source_count,
            "transition_count_exact": isinstance(source_count, int)
            and item.get("transition_count") == source_count - 1,
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
            "source_json_hash_bound": isinstance(item.get("source_json_sha256"), str),
            "plan_hash_bound": isinstance(item.get("plan_sha256"), str),
        }
        if case > 0:
            seen.add(case)
        rows.append({"case": case, "checks": checks, "passed": all(checks.values())})

    top_checks = {
        "trajectory_integrity_contract": manifest.get(
            "trajectory_integrity_contract"
        )
        == CONTRACT,
        "package_valid_for_training": manifest.get("valid_for_training") is True,
        "package_quality_gate_passed": manifest.get("quality_gate_passed") is True,
        "declared_case_count": manifest.get("case_count") == expected_count,
        "item_count": len(rows) == expected_count,
        "contiguous_cases": sorted(seen) == list(range(1, expected_count + 1)),
        "all_case_checks": len(rows) == expected_count
        and all(row["passed"] for row in rows),
        "quarantined_lineage_absent": manifest.get(
            "quarantined_lineage_absent"
        )
        is True,
    }
    passed = all(top_checks.values())
    return {
        "schema": "cinebotrl_riser_exact_source_training_admission_audit_v1",
        "mode": "training",
        "trajectory_integrity_contract": CONTRACT,
        "expected_case_count": expected_count,
        "top_checks": top_checks,
        "rows": rows,
        "passed": passed,
        "reference_ingest_authorized": False,
        "training_authorized": passed,
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("reference_ingest", "training"), required=True
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_count <= 0:
        raise ValueError("expected count must be positive")
    actual_hash = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.mode == "reference_ingest":
        result = validate_reference_ingest(manifest, args.manifest, args.expected_count)
    else:
        result = validate_training_manifest(manifest, args.expected_count)
    result |= {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": actual_hash,
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "manifest_hash_matches_expected": args.expected_manifest_sha256 is None
        or actual_hash == args.expected_manifest_sha256,
    }
    result["passed"] = bool(
        result["passed"] and result["manifest_hash_matches_expected"]
    )
    result["reference_ingest_authorized"] = bool(
        result["reference_ingest_authorized"] and result["passed"]
    )
    result["training_authorized"] = bool(
        result["training_authorized"] and result["passed"]
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
