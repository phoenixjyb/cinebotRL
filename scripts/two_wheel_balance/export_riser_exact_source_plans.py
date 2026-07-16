#!/usr/bin/env python3
"""Export integrity-preserving riser plans from the exact-source package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_exact_source import (  # noqa: E402
    EXACT_SOURCE_CONTRACT,
    audit_exact_source_playback_plan,
    execution_schedule_for_source,
    load_exact_source_package,
    save_exact_source_playback_plan,
    sha256_file,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_playback import (  # noqa: E402
    PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S,
    playback_plan_from_kinematic_plan,
    riser_playback_kinematic_gate,
    riser_playback_kinematic_metrics,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_reference import (  # noqa: E402
    plan_rs4_riser_reference,
)


def parse_cases(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    cases = [int(item) for item in value.split(",") if item.strip()]
    if not cases or len(cases) != len(set(cases)) or any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("cases must be 'all' or unique positive integers")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("1,4,7"))
    args = parser.parse_args()

    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    cases = sorted(references) if args.cases is None else args.cases
    missing = sorted(set(cases) - set(references))
    if missing:
        raise ValueError(f"cases absent from exact-source package: {missing}")
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        source = references[case]
        execution_time = execution_schedule_for_source(source)
        planning_reference = source.planning_reference(execution_time)
        vertical_shift_m = max(
            0.0,
            0.6 - float(planning_reference.positions_m[:, 2].min()),
        )
        kinematic_plan = plan_rs4_riser_reference(
            planning_reference,
            kinematics,
            vertical_shift_m=vertical_shift_m,
            maximum_base_yaw_rate_rad_s=PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S,
        )
        plan = playback_plan_from_kinematic_plan(
            planning_reference, kinematic_plan
        )
        metrics = riser_playback_kinematic_metrics(plan, kinematics)
        kinematic_checks = riser_playback_kinematic_gate(metrics, kinematics)
        output = args.output_dir / f"case_{case:04d}_exact_source_riser_playback_v1.npz"
        save_exact_source_playback_plan(output, plan, source)
        integrity = audit_exact_source_playback_plan(output, source)
        if not integrity["passed"]:
            raise RuntimeError(f"case {case} failed exact-source integrity: {integrity['checks']}")
        rows.append(
            {
                **{key: value for key, value in integrity.items() if key != "checks"},
                "kinematic_metrics": metrics,
                "kinematic_checks": kinematic_checks,
                "kinematic_gate_passed": all(kinematic_checks.values()),
            }
        )
        print(
            json.dumps(
                {
                    "case": case,
                    "source_poses": source.source_pose_count,
                    "source_duration_s": float(source.source_time_s[-1]),
                    "execution_duration_s": float(execution_time[-1]),
                    "integrity_passed": integrity["passed"],
                    "kinematic_gate_passed": all(kinematic_checks.values()),
                }
            ),
            flush=True,
        )

    exact_pass_count = sum(bool(row["trajectory_integrity_passed"]) for row in rows)
    manifest = {
        "schema": "cinebotrl_two_wheel_riser_exact_source_plan_export_v1",
        "trajectory_integrity_contract": EXACT_SOURCE_CONTRACT,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "source_package_case_count": len(references),
        "case_count": len(rows),
        "exact_source_pass_count": exact_pass_count,
        "all_anchor_maps_complete": exact_pass_count == len(rows),
        "trajectory_integrity_passed": exact_pass_count == len(rows),
        "quality_gate_passed": False,
        "valid_for_training": False,
        "training_started": False,
        "quarantined_lineage_absent": True,
        "items": rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        **manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "source_pose_count_total": sum(int(row["source_pose_count"]) for row in rows),
        "source_path_length_total_m": sum(float(row["source_path_length_m"]) for row in rows),
        "source_duration_total_s": sum(float(row["source_duration_s"]) for row in rows),
        "execution_duration_total_s": sum(float(row["execution_duration_s"]) for row in rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if manifest["trajectory_integrity_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
