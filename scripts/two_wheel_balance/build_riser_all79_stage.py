#!/usr/bin/env python3
"""Admit the full all-79 stage through the corrected v2 teacher contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    FullReference,
    SparseTeacher,
    discover_full_stage,
    discover_v3_package,
    monotonic_pose_match,
    normalize_quaternions_wxyz,
    parse_acquisition_time_scale_overrides,
    regenerate_acquisition_attitude_prefix,
    regenerate_acquisition_prefix,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (  # noqa: E402
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        semantic_quat = normalize_quaternions_wxyz(
            np.asarray(handle["semanticQuat"], dtype=np.float64).T
        )
        q_path = np.asarray(handle["qPath"], dtype=np.float64)
        time_s = np.asarray(handle["time"], dtype=np.float64).reshape(-1)
    require(q_path.shape[0] == len(time_s), f"bad q/time shape in {path}")
    require(
        semantic_poses.shape == physical_targets.shape
        and semantic_poses.shape[0] == len(time_s),
        f"bad pose evidence in {path}",
    )
    require(len(semantic_quat) == len(time_s), f"bad attitude evidence in {path}")
    return {
        "semantic_poses": semantic_poses,
        "physical_targets": physical_targets,
        "semantic_quat": semantic_quat,
        "q_path": q_path,
        "time_s": time_s,
    }


def exported_anchor_metrics(
    full: FullReference,
    teacher: SparseTeacher,
    evidence: dict[str, np.ndarray],
) -> dict[str, object]:
    """Validate only samples exported as labels; MATLAB sample zero is initialization."""

    q_error = float(np.max(np.abs(evidence["q_path"][:, :6] - teacher.base_arm_q)))
    time_error = float(np.max(np.abs(evidence["time_s"] - teacher.time_s)))
    source_attitudes = evidence["semantic_quat"][1:]
    dots = np.abs(np.sum(source_attitudes * teacher.dfr_attitudes_wxyz, axis=1))
    export_attitude_error = float(
        np.max(np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))))
    )
    split_rotation_error = float(
        np.max(
            np.abs(
                evidence["semantic_poses"][:, :3, :3]
                - evidence["physical_targets"][:, :3, :3]
            )
        )
    )
    split_translation_error = float(
        np.max(
            np.linalg.norm(
                evidence["semantic_poses"][:, :3, 3]
                - evidence["physical_targets"][:, :3, 3],
                axis=1,
            )
        )
    )
    indices, position_error, attitude_error = monotonic_pose_match(
        full,
        evidence["semantic_poses"][1:, :3, 3],
        source_attitudes,
    )
    return {
        "first_exported_full_index": int(indices[0]),
        "last_exported_full_index": int(indices[-1]),
        "source_q_max_error": q_error,
        "source_time_max_error_s": time_error,
        "source_attitude_export_max_error_deg": export_attitude_error,
        "split_target_rotation_max_abs_error": split_rotation_error,
        "split_target_translation_max_error_m": split_translation_error,
        "full_anchor_position_max_error_m": float(np.max(position_error)),
        "full_anchor_attitude_max_error_deg": float(np.max(attitude_error)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-stage", type=Path, required=True)
    parser.add_argument("--v3-package", type=Path, required=True)
    parser.add_argument("--v3-source-batch", type=Path, required=True)
    parser.add_argument("--riser-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-scale-overrides", default="")
    parser.add_argument("--acquisition-time-scale-overrides", default="")
    parser.add_argument("--maximum-position-anchor-error-m", type=float, default=0.02)
    parser.add_argument("--maximum-attitude-anchor-error-deg", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    full = discover_full_stage(args.full_stage)
    teachers = discover_v3_package(args.v3_package)
    kinematics = UrdfRiserCameraKinematics(args.riser_urdf)
    time_scale_overrides = parse_acquisition_time_scale_overrides(
        args.time_scale_overrides
    )
    acquisition_scale_overrides = parse_acquisition_time_scale_overrides(
        args.acquisition_time_scale_overrides
    )
    overlap = set(time_scale_overrides) & set(acquisition_scale_overrides)
    require(not overlap, f"cases cannot use both retiming modes: {sorted(overlap)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for case in range(1, 80):
        reference = full[case]
        teacher = teachers[case]
        mat_path = args.v3_source_batch / f"episode_{case:04d}" / "teacher_smoke.mat"
        require(mat_path.is_file(), f"missing source evidence {mat_path}")
        evidence = load_mat_evidence(mat_path)
        metrics = exported_anchor_metrics(reference, teacher, evidence)
        require(metrics["source_q_max_error"] <= 2e-6, f"case {case} q mismatch")
        require(metrics["source_time_max_error_s"] <= 2e-8, f"case {case} time mismatch")
        require(
            metrics["source_attitude_export_max_error_deg"] <= 1e-4,
            f"case {case} attitude-label mismatch",
        )
        require(
            metrics["split_target_rotation_max_abs_error"] <= 1e-12
            and metrics["split_target_translation_max_error_m"] <= 1e-12,
            f"case {case} split target changed",
        )
        require(
            metrics["full_anchor_position_max_error_m"]
            <= args.maximum_position_anchor_error_m,
            f"case {case} full-stage position mismatch",
        )
        require(
            metrics["full_anchor_attitude_max_error_deg"]
            <= args.maximum_attitude_anchor_error_deg,
            f"case {case} full-stage attitude mismatch",
        )

        acquisition_end_index = int(metrics["first_exported_full_index"])
        initial_yaw = float(teacher.base_arm_q[0, 2])
        vertical_shift = max(
            0.0,
            0.6 - float(np.min(reference.positions_m[:, 2])),
        )
        shifted_reference = replace(
            reference,
            positions_m=reference.positions_m
            + np.array([0.0, 0.0, vertical_shift]),
        )
        initial_height = float(
            np.clip(shifted_reference.positions_m[0, 2], 0.6, 1.8)
        )
        home_solution = kinematics.solve_position(
            np.array([0.0, 0.0, initial_height]),
            initial_yaw,
            np.zeros(3),
        )
        require(home_solution.reachable, f"case {case} home position is unreachable")
        home_transform = kinematics.world_transform(
            home_solution.base_xy_yaw_riser[:3],
            home_solution.base_xy_yaw_riser[3],
            np.zeros(3),
        )
        home_position = home_transform[:3, 3]
        home_semantic = physical_cam_to_semantic_dfr_quat_wxyz(
            matrix_quaternion_wxyz(home_transform[:3, :3])
        )
        positions, _ = regenerate_acquisition_prefix(
            shifted_reference,
            home_position,
            acquisition_end_index,
        )
        attitudes, _ = regenerate_acquisition_attitude_prefix(
            shifted_reference,
            home_semantic,
            acquisition_end_index,
        )
        time_scale = time_scale_overrides.get(case, 1.0)
        acquisition_scale = acquisition_scale_overrides.get(case, 1.0)
        time_s = reference.time_s.copy()
        if time_scale != 1.0:
            time_s *= time_scale
        elif acquisition_scale != 1.0:
            acquisition_duration = float(time_s[acquisition_end_index])
            time_s[: acquisition_end_index + 1] *= acquisition_scale
            time_s[acquisition_end_index + 1 :] += (
                acquisition_scale - 1.0
            ) * acquisition_duration

        poses = [
            {
                "position": position.tolist(),
                "orientation": attitude[[1, 2, 3, 0]].tolist(),
            }
            for position, attitude in zip(
                positions, attitudes, strict=True
            )
        ]
        filename = f"episode_{case:04d}_split_teacher_v1.json"
        output = args.output_dir / filename
        payload = {
            "poses": poses,
            "time_s": time_s.tolist(),
            "metadata": {
                "source": "corrected_all79_full_stage_v2",
                "source_full_reference": reference.path.name,
                "source_full_reference_sha256": sha256(reference.path),
                "source_teacher_npz": teacher.path.name,
                "source_teacher_npz_sha256": sha256(teacher.path),
                "source_mat_evidence_sha256": sha256(mat_path),
                "scenario": "no_obstacle",
                "quality_status": "accepted",
                "episode_index": case,
                "duration_s": float(time_s[-1]),
                "waypoint_dt": float(np.median(np.diff(reference.time_s))),
                "timing_contract": "explicit_time_s_v1",
                "source_duration_s": float(reference.time_s[-1]),
                "time_scale": time_scale,
                "acquisition_time_scale": acquisition_scale,
                "initial_base_pose_xyyaw": home_solution.base_xy_yaw_riser[:3].tolist(),
                "initial_riser_position_m": float(
                    home_solution.base_xy_yaw_riser[3]
                ),
                "acquisition_end_index": acquisition_end_index,
                "acquisition_contract": "physical_home_to_audited_semantic_start_v2",
                "pre_applied_vertical_shift_m": vertical_shift,
                "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
                "recorded_quaternion_order": "xyzw",
                "observation_ee_frame": "physical_cam_link_fk",
                "riser_use": "semantic_pose_only_no_source_action_labels",
                "physical_gimbal_labels_used_as_actions": False,
                "matlab_initialization_sample_used_as_label": False,
            },
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "case": case,
                "file": filename,
                "samples": len(poses),
                "source_duration_s": float(reference.time_s[-1]),
                "duration_s": float(time_s[-1]),
                "time_scale": time_scale,
                "acquisition_time_scale": acquisition_scale,
                "acquisition_end_index": acquisition_end_index,
                "initial_camera_height_m": float(home_position[2]),
                "pre_applied_vertical_shift_m": vertical_shift,
                "sha256": sha256(output),
                **metrics,
            }
        )

    (args.output_dir / "manifest.txt").write_text(
        "\n".join(item["file"] for item in records) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "cinebotrl_riser_corrected_all79_stage_v2",
        "training_started": False,
        "ppo_authorized": False,
        "source_action_labels_used": False,
        "physical_gimbal_labels_used_as_actions": False,
        "matlab_initialization_sample_used_as_label": False,
        "time_scale_overrides": {
            str(case): scale for case, scale in time_scale_overrides.items()
        },
        "acquisition_time_scale_overrides": {
            str(case): scale for case, scale in acquisition_scale_overrides.items()
        },
        "retimed_cases": sorted(
            set(time_scale_overrides) | set(acquisition_scale_overrides)
        ),
        "accepted_case_count": len(records),
        "rejected_case_count": 0,
        "source_total_duration_s": float(
            sum(item["source_duration_s"] for item in records)
        ),
        "total_duration_s": float(sum(item["duration_s"] for item in records)),
        "worst_metrics": {
            key: max(float(item[key]) for item in records)
            for key in (
                "source_q_max_error",
                "source_time_max_error_s",
                "source_attitude_export_max_error_deg",
                "split_target_rotation_max_abs_error",
                "split_target_translation_max_error_m",
                "full_anchor_position_max_error_m",
                "full_anchor_attitude_max_error_deg",
            )
        },
        "cases": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
