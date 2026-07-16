#!/usr/bin/env python3
"""Compose a hash-audited all-case portfolio from base and recovery plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_exact_source import (  # noqa: E402
    EXACT_SOURCE_CONTRACT,
    audit_exact_source_playback_plan,
    load_exact_source_package,
    sha256_file,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_playback import (  # noqa: E402
    load_riser_playback_plan,
    riser_playback_kinematic_gate,
    riser_playback_kinematic_metrics,
)


def _load_plan_manifest(directory: Path, expected_sha256: str) -> tuple[dict, dict[int, dict]]:
    path = directory / "manifest.json"
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"plan manifest hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("trajectory_integrity_contract") != EXACT_SOURCE_CONTRACT:
        raise ValueError(f"wrong plan contract: {path}")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"missing plan items: {path}")
    by_case = {int(item["case"]): item for item in items}
    if len(by_case) != len(items):
        raise ValueError(f"duplicate plan case: {path}")
    return payload, by_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--base-plan-dir", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--recovery-plan-dir", type=Path, required=True)
    parser.add_argument("--expected-recovery-manifest-sha256", required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--minimum-gate-c-candidates", type=int, default=70)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output namespace: {args.output_dir}")
    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    base_manifest, base_items = _load_plan_manifest(
        args.base_plan_dir, args.expected_base_manifest_sha256
    )
    recovery_manifest, recovery_items = _load_plan_manifest(
        args.recovery_plan_dir, args.expected_recovery_manifest_sha256
    )
    if sorted(base_items) != list(range(1, args.expected_count + 1)):
        raise ValueError("base plan must cover every expected case")
    if not recovery_items:
        raise ValueError("recovery plan must contain at least one case")
    if any(
        item.get("kinematic_gate_passed") is not True
        for item in recovery_items.values()
    ):
        raise ValueError("recovery manifest contains a non-passing case")

    kinematics = UrdfRiserCameraKinematics(args.urdf)
    args.output_dir.mkdir(parents=True)
    rows = []
    for case in range(1, args.expected_count + 1):
        recovered = case in recovery_items
        item = recovery_items.get(case, base_items[case])
        source_dir = args.recovery_plan_dir if recovered else args.base_plan_dir
        source_plan = source_dir / str(item["file"])
        if sha256_file(source_plan) != item.get("plan_sha256"):
            raise ValueError(f"plan hash mismatch for case {case}: {source_plan}")
        destination = args.output_dir / source_plan.name
        shutil.copy2(source_plan, destination)
        integrity = audit_exact_source_playback_plan(destination, references[case])
        if not integrity["passed"]:
            raise ValueError(f"integrity audit failed for case {case}")
        plan = load_riser_playback_plan(destination)
        metrics = riser_playback_kinematic_metrics(plan, kinematics)
        checks = riser_playback_kinematic_gate(metrics, kinematics)
        passed = all(checks.values())
        if passed != bool(item.get("kinematic_gate_passed")):
            raise ValueError(f"quality result changed for case {case}")
        rows.append(
            {
                **{key: value for key, value in integrity.items() if key != "checks"},
                "source_plan_namespace": source_dir.name,
                "recovery_selected": recovered,
                "kinematic_metrics": metrics,
                "kinematic_checks": checks,
                "kinematic_gate_passed": passed,
                "quality_gate_passed": False,
                "valid_for_training": False,
            }
        )

    accepted = [row["case"] for row in rows if row["kinematic_gate_passed"]]
    rejected = [row["case"] for row in rows if not row["kinematic_gate_passed"]]
    gate_c_ready = len(accepted) >= args.minimum_gate_c_candidates
    manifest = {
        "schema": "cinebotrl_two_wheel_riser_exact_source_plan_portfolio_v1",
        "trajectory_integrity_contract": EXACT_SOURCE_CONTRACT,
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "base_plan_manifest_sha256": args.expected_base_manifest_sha256,
        "recovery_plan_manifest_sha256": args.expected_recovery_manifest_sha256,
        "case_count": len(rows),
        "exact_source_pass_count": sum(
            bool(row["trajectory_integrity_passed"]) for row in rows
        ),
        "kinematic_accepted_count": len(accepted),
        "kinematic_accepted_cases": accepted,
        "kinematic_rejected_count": len(rejected),
        "kinematic_rejected_cases": rejected,
        "minimum_gate_c_candidates": args.minimum_gate_c_candidates,
        "gate_c_candidate_count_sufficient": gate_c_ready,
        "gate_c_dynamic_quality_started": False,
        "quality_gate_passed": False,
        "valid_for_training": False,
        "training_started": False,
        "ppo_started": False,
        "quarantined_lineage_absent": True,
        "base_manifest_training_started": base_manifest.get("training_started"),
        "recovery_manifest_training_started": recovery_manifest.get("training_started"),
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
    return 0 if gate_c_ready else 7


if __name__ == "__main__":
    raise SystemExit(main())
