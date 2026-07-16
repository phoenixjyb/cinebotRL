#!/usr/bin/env python3
"""Export self-contained corrected riser plans for deterministic Isaac replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_playback import (  # noqa: E402
    build_riser_playback_plan,
    riser_playback_kinematic_gate,
    riser_playback_kinematic_metrics,
    save_riser_playback_plan,
)
from rl_platform.tasks.two_wheel_balance.riser_reference import (  # noqa: E402
    discover_corrected_riser_stage,
)


def parse_cases(value: str) -> list[int]:
    cases = [int(item) for item in value.split(",") if item.strip()]
    if not cases or len(cases) != len(set(cases)) or any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("cases must be unique positive integers")
    return cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("1,31,73"))
    parser.add_argument("--expected-count", type=int, default=62)
    args = parser.parse_args()

    references = discover_corrected_riser_stage(
        args.stage, expected_count=args.expected_count
    )
    missing = sorted(set(args.cases) - set(references))
    if missing:
        raise ValueError(f"cases absent from corrected stage: {missing}")
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        plan = build_riser_playback_plan(references[case], kinematics)
        metrics = riser_playback_kinematic_metrics(plan, kinematics)
        checks = riser_playback_kinematic_gate(metrics, kinematics)
        if not all(checks.values()):
            raise RuntimeError(f"case {case} regressed after export: {checks}")
        output = args.output_dir / f"case_{case:04d}_riser_playback_v1.npz"
        save_riser_playback_plan(output, plan)
        rows.append(
            {
                "case": case,
                "file": output.name,
                "sha256": sha256(output),
                "sample_count": len(plan.time_s),
                "duration_s": float(plan.time_s[-1]),
                "planning_strategy": plan.planning_strategy,
                "vertical_shift_m": plan.vertical_shift_m,
                "kinematic_metrics": metrics,
                "kinematic_checks": checks,
                "passed": all(checks.values()),
            }
        )
    manifest = {
        "schema": "cinebotrl_two_wheel_riser_playback_export_v1",
        "training_started": False,
        "source_contracts": sorted(
            {str(references[case].metadata["source"]) for case in args.cases}
        ),
        "case_count": len(rows),
        "cases": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
