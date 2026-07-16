#!/usr/bin/env python3
"""Audit and plot exact-source canary retarget artifacts independently."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.exact_source_reference import (  # noqa: E402
    validate_exact_source_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="1,4,7")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_length(positions: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def audit_case(path: Path, output_dir: Path, case: int) -> dict[str, object]:
    arrays = validate_exact_source_candidate(
        path,
        require_offline_quality=False,
        require_dynamic_approval=False,
    )
    source_position = arrays["source_position_world_m"]
    source_attitude = arrays["source_attitude_world_dfr_quat_xyzw"][:, [3, 0, 1, 2]]
    mapping = arrays["source_anchor_execution_index"]
    execution_position = arrays["target_position_world_m"]
    execution_attitude = arrays["target_attitude_world_dfr_quat_wxyz"]
    mapped_position = execution_position[mapping]
    mapped_attitude = execution_attitude[mapping]
    position_error = np.linalg.norm(mapped_position - source_position, axis=1)
    attitude_error = 2.0 * np.arccos(
        np.clip(np.abs(np.sum(mapped_attitude * source_attitude, axis=1)), -1.0, 1.0)
    )
    semantic_execution = execution_position[int(mapping[0]) :]
    result = {
        "case": case,
        "candidate": str(path.resolve()),
        "candidate_sha256": sha256(path),
        "source_manifest_sha256": str(arrays["source_manifest_sha256"].item()),
        "source_json_sha256": str(arrays["source_json_sha256"].item()),
        "source_pose_count": len(source_position),
        "source_anchor_count": len(mapping),
        "execution_sample_count": len(execution_position),
        "execution_transition_count": len(arrays["control_v_wz_darm"]),
        "source_anchor_position_max_error_m": float(np.max(position_error)),
        "source_anchor_attitude_max_error_rad": float(np.max(attitude_error)),
        "source_path_length_m": path_length(source_position),
        "mapped_anchor_path_length_m": path_length(mapped_position),
        "semantic_execution_target_path_length_m": path_length(semantic_execution),
        "anchor_mapping_strict": bool(np.all(np.diff(mapping) > 0)),
        "initialization_separate": bool(
            int(mapping[0]) == int(arrays["initialization_sample_count"])
        ),
        "offline_executable_quality_passed": bool(
            arrays["offline_executable_quality_passed"].item()
        ),
        "valid_for_dynamic_evaluation": bool(
            arrays["valid_for_dynamic_evaluation"].item()
        ),
        "valid_for_training": False,
        "trajectory_integrity_passed": True,
    }

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        source_position[:, 0],
        source_position[:, 1],
        source_position[:, 2],
        color="#d1495b",
        linewidth=2.2,
        label="authoritative source anchors",
    )
    axis.plot(
        semantic_execution[:, 0],
        semantic_execution[:, 1],
        semantic_execution[:, 2],
        color="#00798c",
        linewidth=1.0,
        alpha=0.75,
        label="retarget desired execution path",
    )
    axis.scatter(
        mapped_position[:, 0],
        mapped_position[:, 1],
        mapped_position[:, 2],
        color="#edae49",
        s=5,
        label="mapped source anchors",
    )
    axis.set_title(f"Exact-source retarget integrity: episode {case}")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_zlabel("world z (m)")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / f"case_{case:04d}_source_vs_execution.png", dpi=160)
    plt.close(figure)
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [int(value) for value in args.cases.split(",") if value.strip()]
    results = [
        audit_case(
            args.candidate_dir / f"case_{case:04d}.npz",
            args.output_dir,
            case,
        )
        for case in cases
    ]
    summary = {
        "schema": "cinebotrl_two_wheel_exact_source_canary_audit_v1",
        "trajectory_integrity_contract": "exact_source_v1",
        "cases": cases,
        "integrity_passed_cases": [
            row["case"] for row in results if row["trajectory_integrity_passed"]
        ],
        "offline_quality_passed_cases": [
            row["case"] for row in results if row["offline_executable_quality_passed"]
        ],
        "valid_for_training": False,
        "results": results,
        "trajectory_integrity_passed": all(
            row["trajectory_integrity_passed"] for row in results
        ),
        "offline_executable_quality_passed": all(
            row["offline_executable_quality_passed"] for row in results
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    with (args.output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, indent=2))
    return 0 if summary["trajectory_integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
