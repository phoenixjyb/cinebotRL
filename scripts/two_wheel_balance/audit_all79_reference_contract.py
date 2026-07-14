#!/usr/bin/env python3
"""Bind the full all-79 stage to the corrected sparse v3 teacher contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    discover_full_stage,
    discover_v3_package,
    monotonic_pose_match,
    normalize_quaternions_wxyz,
    source_body_velocities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-stage", type=Path, required=True)
    parser.add_argument("--v3-package", type=Path, required=True)
    parser.add_argument("--v3-source-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.02)
    parser.add_argument("--maximum-attitude-error-deg", type=float, default=3.0)
    parser.add_argument("--lateral-action-threshold-mps", type=float, default=0.02)
    return parser.parse_args()


def as_pose_samples(values: np.ndarray) -> np.ndarray:
    """Restore MATLAB homogeneous transforms as [samples, 4, 4]."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected pose tensor, got {values.shape}")
    if values.shape[1:] == (4, 4):
        return np.swapaxes(values, 1, 2)
    if values.shape[:2] == (4, 4):
        return np.moveaxis(values, 2, 0)
    raise ValueError(f"cannot orient pose tensor {values.shape}")


def load_mat_evidence(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        semantic_poses = as_pose_samples(handle["semanticPoses"])
        physical_targets = as_pose_samples(handle["physicalTargets"])
        semantic_quat = normalize_quaternions_wxyz(np.asarray(handle["semanticQuat"]).T)
        q_path = np.asarray(handle["qPath"], dtype=np.float64)
        time_s = np.asarray(handle["time"], dtype=np.float64).reshape(-1)
    if q_path.shape[0] != len(time_s) or q_path.shape[1] < 6:
        raise ValueError(f"bad q/time shape in {path}: {q_path.shape}, {time_s.shape}")
    if semantic_poses.shape[0] != len(time_s) or len(semantic_quat) != len(time_s):
        raise ValueError(f"bad semantic pose length in {path}")
    return {
        "semantic_poses": semantic_poses,
        "physical_targets": physical_targets,
        "semantic_quat": semantic_quat,
        "q_path": q_path,
        "time_s": time_s,
    }


def split_target_error(semantic: np.ndarray, physical: np.ndarray) -> tuple[float, float]:
    rotation_error = float(
        np.max(np.abs(semantic[:, :3, :3] - physical[:, :3, :3]))
    )
    translation_error = float(
        np.max(np.linalg.norm(semantic[:, :3, 3] - physical[:, :3, 3], axis=1))
    )
    return rotation_error, translation_error


def main() -> int:
    args = parse_args()
    full = discover_full_stage(args.full_stage)
    teachers = discover_v3_package(args.v3_package)
    rows = []
    source_contract_failures = []
    split_rotation_max = 0.0
    split_translation_max = 0.0

    for case in range(1, 80):
        mat_path = args.v3_source_batch / f"episode_{case:04d}" / "teacher_smoke.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        evidence = load_mat_evidence(mat_path)
        teacher = teachers[case]
        q_error = float(np.max(np.abs(evidence["q_path"][:, :6] - teacher.base_arm_q)))
        time_error = float(np.max(np.abs(evidence["time_s"] - teacher.time_s)))
        dots = np.abs(
            np.sum(evidence["semantic_quat"][1:] * teacher.dfr_attitudes_wxyz, axis=1)
        )
        attitude_export_error = float(
            np.max(np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))))
        )
        if q_error > 2e-6 or time_error > 2e-8 or attitude_export_error > 1e-4:
            source_contract_failures.append(case)

        split_rotation, split_translation = split_target_error(
            evidence["semantic_poses"], evidence["physical_targets"]
        )
        split_rotation_max = max(split_rotation_max, split_rotation)
        split_translation_max = max(split_translation_max, split_translation)

        indices, position_error, attitude_error = monotonic_pose_match(
            full[case],
            evidence["semantic_poses"][:, :3, 3],
            evidence["semantic_quat"],
        )
        body_velocity = source_body_velocities(teacher)
        max_abs_vy = float(np.max(np.abs(body_velocity[:, 1])))
        rows.append(
            {
                "case": case,
                "full_duration_s": float(full[case].time_s[-1]),
                "full_samples": len(full[case].time_s),
                "v3_samples": len(evidence["time_s"]),
                "acquisition_end_index": int(indices[0]),
                "acquisition_duration_s": float(full[case].time_s[indices[0]]),
                "position_error_mean_m": float(np.mean(position_error)),
                "position_error_max_m": float(np.max(position_error)),
                "attitude_error_mean_deg": float(np.mean(attitude_error)),
                "attitude_error_max_deg": float(np.max(attitude_error)),
                "source_max_abs_vx_mps": float(np.max(np.abs(body_velocity[:, 0]))),
                "source_max_abs_vy_mps": max_abs_vy,
                "source_max_abs_wz_radps": float(np.max(np.abs(body_velocity[:, 2]))),
                "requires_nonholonomic_retarget": max_abs_vy
                > args.lateral_action_threshold_mps,
                "position_contract_pass": float(np.max(position_error))
                <= args.maximum_position_error_m,
                "attitude_contract_pass": float(np.max(attitude_error))
                <= args.maximum_attitude_error_deg,
                "source_q_max_error": q_error,
                "source_time_max_error_s": time_error,
                "source_attitude_max_error_deg": attitude_export_error,
            }
        )

    position_failures = [row["case"] for row in rows if not row["position_contract_pass"]]
    attitude_quarantine = [row["case"] for row in rows if not row["attitude_contract_pass"]]
    retarget_cases = [row["case"] for row in rows if row["requires_nonholonomic_retarget"]]
    checks = {
        "full_stage_79_cases": len(full) == 79,
        "v3_package_79_cases": len(teachers) == 79,
        "source_export_contract_all79": not source_contract_failures,
        "v3_split_target_rotation_unchanged": split_rotation_max <= 1e-12,
        "v3_split_target_translation_unchanged": split_translation_max <= 1e-12,
        "position_contract_all79": not position_failures,
        "direct_holonomic_actions_blocked": bool(retarget_cases),
        "training_not_started": True,
    }
    result = {
        "schema": "cinebotrl_two_wheel_all79_contract_audit_v1",
        "training_started": False,
        "full_stage": args.full_stage.name,
        "v3_package": args.v3_package.name,
        "v3_source_batch": args.v3_source_batch.name,
        "case_count": len(rows),
        "full_duration_total_s": float(sum(row["full_duration_s"] for row in rows)),
        "full_duration_min_s": float(min(row["full_duration_s"] for row in rows)),
        "full_duration_max_s": float(max(row["full_duration_s"] for row in rows)),
        "position_error_max_m": float(max(row["position_error_max_m"] for row in rows)),
        "attitude_error_max_deg": float(max(row["attitude_error_max_deg"] for row in rows)),
        "source_max_abs_vy_mps": float(max(row["source_max_abs_vy_mps"] for row in rows)),
        "source_contract_failure_cases": source_contract_failures,
        "position_contract_failure_cases": position_failures,
        "attitude_quarantined_cases": attitude_quarantine,
        "nonholonomic_retarget_cases": retarget_cases,
        "nonholonomic_retarget_case_count": len(retarget_cases),
        "v3_split_target_rotation_max_abs_error": split_rotation_max,
        "v3_split_target_translation_max_error_m": split_translation_max,
        "option_b_physical_camera_adapter_audited": False,
        "passed_for_position_retargeting": all(checks.values()),
        "passed_for_attitude_control": all(checks.values()) and not attitude_quarantine,
        "checks": checks,
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    with (args.output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return 0 if result["passed_for_position_retargeting"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
