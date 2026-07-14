"""Tests for corrective-teacher capture validation and CSV export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "imitation" / "export_corrective_teacher_request.py"
SPEC = importlib.util.spec_from_file_location("export_corrective_teacher_request", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_export_accepts_option_b_capture(tmp_path: Path):
    rows = 2
    identity = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (rows, 1))
    semantic = identity.copy()
    physical = MODULE.semantic_dfr_to_physical_cam(semantic).astype(np.float32)
    metadata = {
        "schema": MODULE.SCHEMA,
        "action_contract": "split_base_arm_attitude_v1",
        "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
        "physical_joint_names": [f"joint_{index}" for index in range(6)],
        "physical_joint_roles": ["learned_arm"] * 3 + ["diagnostic_dji_gimbal"] * 3,
        "target_horizon_steps": 2,
        "target_horizon_dt_s": 0.05,
    }
    capture = tmp_path / "capture.npz"
    np.savez_compressed(
        capture,
        observations=np.zeros((rows, 98), dtype=np.float32),
        applied_actions=np.zeros((rows, 9), dtype=np.float32),
        policy_actions=np.zeros((rows, 9), dtype=np.float32),
        action_label_mask=np.tile(MODULE.LEARNED_ACTION_MASK, (rows, 1)),
        rollout_env_id=np.zeros(rows, dtype=np.int32),
        rollout_step=np.arange(rows, dtype=np.int32),
        rollout_waypoint_idx=np.arange(rows, dtype=np.int32),
        source_episode_index=np.ones(rows, dtype=np.int32),
        first_episode_valid=np.ones(rows, dtype=bool),
        source_trajectory_metadata_json=np.asarray(["{}"] * rows),
        trajectory_progress=np.asarray([0.0, 0.1], dtype=np.float32),
        trajectory_time_remaining_s=np.asarray([5.0, 4.95], dtype=np.float32),
        base_position_world_m=np.zeros((rows, 3), dtype=np.float32),
        base_quaternion_world_wxyz=identity,
        base_linear_velocity_world_mps=np.zeros((rows, 3), dtype=np.float32),
        base_angular_velocity_world_radps=np.zeros((rows, 3), dtype=np.float32),
        physical_joint_position_rad=np.zeros((rows, 6), dtype=np.float32),
        physical_joint_velocity_radps=np.zeros((rows, 6), dtype=np.float32),
        cam_position_world_m=np.zeros((rows, 3), dtype=np.float32),
        cam_quaternion_world_wxyz=physical,
        cam_linear_velocity_world_mps=np.zeros((rows, 3), dtype=np.float32),
        cam_angular_velocity_world_radps=np.zeros((rows, 3), dtype=np.float32),
        target_cam_position_world_m=np.zeros((rows, 3), dtype=np.float32),
        target_cam_quaternion_world_wxyz=physical,
        target_semantic_dfr_quaternion_world_wxyz=semantic,
        target_horizon_cam_position_world_m=np.zeros((rows, 2, 3), dtype=np.float32),
        target_horizon_cam_quaternion_world_wxyz=np.tile(physical[:, None, :], (1, 2, 1)),
        target_horizon_semantic_dfr_quaternion_world_wxyz=np.tile(semantic[:, None, :], (1, 2, 1)),
        metadata=json.dumps(metadata),
    )
    output_dir = tmp_path / "export"
    manifest = MODULE.build_export(capture, output_dir)
    assert manifest["status"] == "valid_request_not_teacher_labels"
    assert manifest["statistics"]["rows"] == rows
    assert (output_dir / "corrective_teacher_request_samples.csv").exists()
    assert (output_dir / "corrective_teacher_request_manifest.json").exists()
