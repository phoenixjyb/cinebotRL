#!/usr/bin/env python3
"""Validate an Isaac corrective-teacher capture and export MATLAB-friendly files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "corrective_teacher_request_v1"
LEARNED_ACTION_MASK = np.asarray([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float32)
REQUIRED_SHAPES = {
    "observations": (None,),
    "applied_actions": (9,),
    "policy_actions": (9,),
    "action_label_mask": (9,),
    "base_position_world_m": (3,),
    "base_quaternion_world_wxyz": (4,),
    "base_linear_velocity_world_mps": (3,),
    "base_angular_velocity_world_radps": (3,),
    "physical_joint_position_rad": (6,),
    "physical_joint_velocity_radps": (6,),
    "cam_position_world_m": (3,),
    "cam_quaternion_world_wxyz": (4,),
    "cam_linear_velocity_world_mps": (3,),
    "cam_angular_velocity_world_radps": (3,),
    "target_cam_position_world_m": (3,),
    "target_cam_quaternion_world_wxyz": (4,),
    "target_semantic_dfr_quaternion_world_wxyz": (4,),
}
SCALAR_KEYS = (
    "rollout_env_id",
    "rollout_step",
    "rollout_waypoint_idx",
    "source_episode_index",
    "first_episode_valid",
    "source_trajectory_metadata_json",
    "trajectory_progress",
    "trajectory_time_remaining_s",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_dfr_to_physical_cam(quat: np.ndarray) -> np.ndarray:
    c = np.sqrt(0.5)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    converted = np.stack(
        [c * (w - z), c * (x + y), c * (y - x), c * (w + z)],
        axis=-1,
    )
    return converted / np.maximum(np.linalg.norm(converted, axis=-1, keepdims=True), 1e-12)


def quaternion_error_deg(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    current = current / np.maximum(np.linalg.norm(current, axis=-1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(current * target, axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def load_and_validate(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, float]]:
    require(path.exists(), f"capture not found: {path}")
    with np.load(path, allow_pickle=False) as raw:
        data = {name: raw[name] for name in raw.files}
    require("metadata" in data, "capture has no metadata")
    metadata = json.loads(str(data["metadata"].item()))
    require(metadata.get("schema") == SCHEMA, f"expected schema {SCHEMA!r}")
    require(metadata.get("action_contract") == "split_base_arm_attitude_v1", "wrong action contract")
    require(
        metadata.get("target_orientation_contract") == "semantic_dfr_to_physical_cam_v1",
        "wrong target orientation contract",
    )
    require(len(metadata.get("physical_joint_names", [])) == 6, "expected six physical joints")
    require(
        metadata.get("physical_joint_roles")
        == ["learned_arm"] * 3 + ["diagnostic_dji_gimbal"] * 3,
        "physical joint ownership roles changed",
    )
    horizon_steps = int(metadata.get("target_horizon_steps", 0))
    require(horizon_steps > 0, "capture has no positive target horizon")
    require(float(metadata.get("target_horizon_dt_s", 0.0)) > 0.0, "capture has no target horizon dt")

    require("observations" in data and data["observations"].ndim == 2, "observations must be 2D")
    rows = int(data["observations"].shape[0])
    require(rows > 0, "capture has no rows")
    for name, tail_shape in REQUIRED_SHAPES.items():
        require(name in data, f"capture missing {name}")
        value = data[name]
        require(value.shape[0] == rows, f"{name} row count {value.shape[0]} != {rows}")
        if tail_shape != (None,):
            require(value.shape[1:] == tail_shape, f"{name} shape {value.shape} != ({rows}, {tail_shape})")
        require(np.isfinite(value).all(), f"{name} contains non-finite values")
    for name in SCALAR_KEYS:
        require(name in data, f"capture missing {name}")
        value = data[name]
        require(
            value.shape == (rows,) or value.shape == (rows, 1),
            f"{name} must have shape ({rows},) or ({rows}, 1)",
        )
        data[name] = value.reshape(rows)
    horizon_shapes = {
        "target_horizon_cam_position_world_m": (rows, horizon_steps, 3),
        "target_horizon_cam_quaternion_world_wxyz": (rows, horizon_steps, 4),
        "target_horizon_semantic_dfr_quaternion_world_wxyz": (rows, horizon_steps, 4),
    }
    for name, expected_shape in horizon_shapes.items():
        require(name in data, f"capture missing {name}")
        require(data[name].shape == expected_shape, f"{name} shape {data[name].shape} != {expected_shape}")
        require(np.isfinite(data[name]).all(), f"{name} contains non-finite values")

    expected_mask = np.tile(LEARNED_ACTION_MASK, (rows, 1))
    require(np.array_equal(data["action_label_mask"], expected_mask), "action label ownership mask changed")
    require(np.all((data["trajectory_progress"] >= 0.0) & (data["trajectory_progress"] <= 1.0)), "progress outside [0,1]")
    require(np.all(data["trajectory_time_remaining_s"] >= 0.0), "negative remaining time")

    quaternion_keys = (
        "base_quaternion_world_wxyz",
        "cam_quaternion_world_wxyz",
        "target_cam_quaternion_world_wxyz",
        "target_semantic_dfr_quaternion_world_wxyz",
    )
    for name in quaternion_keys:
        norm_error = np.max(np.abs(np.linalg.norm(data[name], axis=1) - 1.0))
        require(norm_error <= 1e-4, f"{name} is not normalized; max error={norm_error:g}")
    for name in (
        "target_horizon_cam_quaternion_world_wxyz",
        "target_horizon_semantic_dfr_quaternion_world_wxyz",
    ):
        norm_error = np.max(np.abs(np.linalg.norm(data[name], axis=-1) - 1.0))
        require(norm_error <= 1e-4, f"{name} is not normalized; max error={norm_error:g}")

    converted = semantic_dfr_to_physical_cam(data["target_semantic_dfr_quaternion_world_wxyz"])
    option_b_error = quaternion_error_deg(converted, data["target_cam_quaternion_world_wxyz"])
    require(float(np.max(option_b_error)) <= 0.05, f"Option-B target mismatch max={np.max(option_b_error):.6f} deg")
    converted_horizon = semantic_dfr_to_physical_cam(
        data["target_horizon_semantic_dfr_quaternion_world_wxyz"]
    )
    option_b_horizon_error = quaternion_error_deg(
        converted_horizon,
        data["target_horizon_cam_quaternion_world_wxyz"],
    )
    require(
        float(np.max(option_b_horizon_error)) <= 0.05,
        f"Option-B target horizon mismatch max={np.max(option_b_horizon_error):.6f} deg",
    )

    position_error = data["target_cam_position_world_m"] - data["cam_position_world_m"]
    position_norm = np.linalg.norm(position_error, axis=1)
    orientation_error = quaternion_error_deg(
        data["cam_quaternion_world_wxyz"],
        data["target_cam_quaternion_world_wxyz"],
    )
    statistics = {
        "rows": rows,
        "first_episode_valid_rows": int(np.count_nonzero(data["first_episode_valid"])),
        "position_error_mean_m": float(np.mean(position_norm)),
        "position_error_p95_m": float(np.percentile(position_norm, 95)),
        "position_error_max_m": float(np.max(position_norm)),
        "orientation_error_mean_deg": float(np.mean(orientation_error)),
        "orientation_error_p95_deg": float(np.percentile(orientation_error, 95)),
        "orientation_error_max_deg": float(np.max(orientation_error)),
        "option_b_roundtrip_max_deg": float(np.max(option_b_error)),
        "option_b_horizon_roundtrip_max_deg": float(np.max(option_b_horizon_error)),
    }
    data["position_error_world_m"] = position_error.astype(np.float32)
    data["position_error_norm_m"] = position_norm.astype(np.float32)
    data["orientation_error_deg"] = orientation_error.astype(np.float32)
    return data, metadata, statistics


def append_vector(row: dict[str, object], prefix: str, values: np.ndarray, suffixes: tuple[str, ...]) -> None:
    for suffix, value in zip(suffixes, values, strict=True):
        row[f"{prefix}_{suffix}"] = float(value)


def export_csv(data: dict[str, np.ndarray], metadata: dict[str, object], output: Path) -> None:
    joint_names = tuple(str(name) for name in metadata["physical_joint_names"])
    rows: list[dict[str, object]] = []
    for index in range(data["observations"].shape[0]):
        row: dict[str, object] = {
            "sample_id": index,
            "rollout_env_id": int(data["rollout_env_id"][index]),
            "rollout_step": int(data["rollout_step"][index]),
            "rollout_waypoint_idx": int(data["rollout_waypoint_idx"][index]),
            "source_episode_index": int(data["source_episode_index"][index]),
            "first_episode_valid": int(data["first_episode_valid"][index]),
            "trajectory_progress": float(data["trajectory_progress"][index]),
            "trajectory_time_remaining_s": float(data["trajectory_time_remaining_s"][index]),
            "source_trajectory_metadata_json": str(data["source_trajectory_metadata_json"][index]),
            "position_error_norm_m": float(data["position_error_norm_m"][index]),
            "orientation_error_deg": float(data["orientation_error_deg"][index]),
        }
        append_vector(row, "base_position_world_m", data["base_position_world_m"][index], ("x", "y", "z"))
        append_vector(row, "base_quaternion_world_wxyz", data["base_quaternion_world_wxyz"][index], ("w", "x", "y", "z"))
        append_vector(row, "base_linear_velocity_world_mps", data["base_linear_velocity_world_mps"][index], ("x", "y", "z"))
        append_vector(row, "base_angular_velocity_world_radps", data["base_angular_velocity_world_radps"][index], ("x", "y", "z"))
        append_vector(row, "cam_position_world_m", data["cam_position_world_m"][index], ("x", "y", "z"))
        append_vector(row, "cam_quaternion_world_wxyz", data["cam_quaternion_world_wxyz"][index], ("w", "x", "y", "z"))
        append_vector(row, "cam_linear_velocity_world_mps", data["cam_linear_velocity_world_mps"][index], ("x", "y", "z"))
        append_vector(row, "cam_angular_velocity_world_radps", data["cam_angular_velocity_world_radps"][index], ("x", "y", "z"))
        append_vector(row, "target_cam_position_world_m", data["target_cam_position_world_m"][index], ("x", "y", "z"))
        append_vector(row, "target_cam_quaternion_world_wxyz", data["target_cam_quaternion_world_wxyz"][index], ("w", "x", "y", "z"))
        append_vector(row, "target_semantic_dfr_quaternion_world_wxyz", data["target_semantic_dfr_quaternion_world_wxyz"][index], ("w", "x", "y", "z"))
        append_vector(row, "position_error_world_m", data["position_error_world_m"][index], ("x", "y", "z"))
        for horizon_index in range(int(metadata["target_horizon_steps"])):
            prefix = f"target_horizon_t{horizon_index + 1:02d}"
            append_vector(
                row,
                f"{prefix}_cam_position_world_m",
                data["target_horizon_cam_position_world_m"][index, horizon_index],
                ("x", "y", "z"),
            )
            append_vector(
                row,
                f"{prefix}_cam_quaternion_world_wxyz",
                data["target_horizon_cam_quaternion_world_wxyz"][index, horizon_index],
                ("w", "x", "y", "z"),
            )
            append_vector(
                row,
                f"{prefix}_semantic_dfr_quaternion_world_wxyz",
                data["target_horizon_semantic_dfr_quaternion_world_wxyz"][index, horizon_index],
                ("w", "x", "y", "z"),
            )
        for joint_index, joint_name in enumerate(joint_names):
            row[f"physical_joint_position_rad__{joint_name}"] = float(data["physical_joint_position_rad"][index, joint_index])
            row[f"physical_joint_velocity_radps__{joint_name}"] = float(data["physical_joint_velocity_radps"][index, joint_index])
        for action_index in range(9):
            row[f"policy_action_{action_index}"] = float(data["policy_actions"][index, action_index])
            row[f"applied_action_{action_index}"] = float(data["applied_actions"][index, action_index])
            row[f"action_label_mask_{action_index}"] = int(data["action_label_mask"][index, action_index])
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_export(capture: Path, output_dir: Path) -> dict[str, object]:
    data, source_metadata, statistics = load_and_validate(capture)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "corrective_teacher_request_samples.csv"
    export_csv(data, source_metadata, csv_path)
    manifest = {
        "schema": SCHEMA,
        "status": "valid_request_not_teacher_labels",
        "source_capture": str(capture.resolve()),
        "source_capture_sha256": sha256(capture),
        "samples_csv": str(csv_path.resolve()),
        "samples_csv_sha256": sha256(csv_path),
        "source_metadata": source_metadata,
        "statistics": statistics,
        "teacher_output_contract": {
            "required_identity_columns": ["sample_id", "source_episode_index", "rollout_step"],
            "learned_action_indices": [0, 1, 2, 6, 7, 8],
            "reserved_action_indices": [3, 4, 5],
            "required_label_columns": [f"teacher_action_{index}" for index in (0, 1, 2, 6, 7, 8)],
            "required_quality_columns": [
                "solver_success",
                "physical_cam_position_residual_m",
                "physical_cam_orientation_residual_deg",
                "joint_limit_margin_min_rad",
            ],
            "prohibition": "Do not emit physical DJI gimbal joints as policy labels.",
        },
    }
    manifest_path = output_dir / "corrective_teacher_request_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_export(args.capture, args.output_dir)
    print(json.dumps(manifest["statistics"], indent=2, sort_keys=True))
    print(f"wrote {args.output_dir / 'corrective_teacher_request_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
