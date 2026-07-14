import json
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.all79_reference import (
    EXPECTED_ACTION_ORDER,
    FullReference,
    load_full_reference,
    load_sparse_teacher,
    monotonic_pose_match,
    parse_acquisition_time_scale_overrides,
    regenerate_acquisition_prefix,
    source_body_velocities,
)


def write_full_reference(path: Path) -> None:
    payload = {
        "poses": [
            {"position": [0.0, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"position": [1.0, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"position": [2.0, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        ],
        "metadata": {"duration_s": 0.2, "waypoint_dt": 0.1},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_sparse_teacher(path: Path, runtime_approved: bool = False) -> None:
    current = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0], [0.1, 0.02, 0.0, 0.1, 1.0, 2.0]])
    next_q = np.array([[0.1, 0.02, 0.0, 0.1, 1.0, 2.0], [0.2, 0.04, 0.0, 0.2, 1.0, 2.0]])
    np.savez_compressed(
        path,
        schema="cinebotrl_gik_monorepo_ee1_split_teacher_v2",
        valid_for_training=np.bool_(True),
        runtime_approved=np.bool_(runtime_approved),
        position_target_link="ee1_tool",
        quaternion_order="wxyz",
        action_order=np.asarray(EXPECTED_ACTION_ORDER),
        q_current_base_arm_6=current,
        q_next_base_arm_6=next_q,
        time_s=np.array([0.1, 0.2]),
        dt_s=np.array([0.1, 0.1]),
        gimbal_attitude_target_world_dfr_quat_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        episode_index=np.int32(1),
    )


def test_full_reference_and_monotonic_match(tmp_path: Path) -> None:
    path = tmp_path / "0001_case.json"
    write_full_reference(path)
    full = load_full_reference(path)
    indices, position_error, attitude_error = monotonic_pose_match(
        full,
        np.array([[0.01, 0.0, 1.0], [1.99, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_allclose(position_error, [0.01, 0.01])
    np.testing.assert_allclose(attitude_error, 0.0)


def test_sparse_teacher_exposes_holonomic_gap(tmp_path: Path) -> None:
    path = tmp_path / "teacher.npz"
    write_sparse_teacher(path)
    teacher = load_sparse_teacher(path)
    velocity = source_body_velocities(teacher)
    np.testing.assert_allclose(velocity[:, 0], 1.0)
    np.testing.assert_allclose(velocity[:, 1], 0.2)
    np.testing.assert_allclose(velocity[:, 2], 0.0)


def test_sparse_teacher_rejects_runtime_approval(tmp_path: Path) -> None:
    path = tmp_path / "teacher.npz"
    write_sparse_teacher(path, runtime_approved=True)
    with pytest.raises(ValueError, match="runtime-approved"):
        load_sparse_teacher(path)


def test_regenerate_acquisition_preserves_semantic_path(tmp_path: Path) -> None:
    path = tmp_path / "0001_case.json"
    write_full_reference(path)
    full = load_full_reference(path)
    targets, semantic_start = regenerate_acquisition_prefix(
        full, np.array([0.5, 0.5, 0.5]), acquisition_end_index=1
    )
    np.testing.assert_allclose(targets[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(semantic_start, full.positions_m[1])
    np.testing.assert_allclose(targets[1:], full.positions_m[1:])


def test_parse_acquisition_time_scale_overrides() -> None:
    assert parse_acquisition_time_scale_overrides("") == {}
    assert parse_acquisition_time_scale_overrides("7:1.25, 23:1.5") == {
        7: 1.25,
        23: 1.5,
    }


@pytest.mark.parametrize(
    "value",
    ("7", "0:1.25", "80:1.25", "7:0.99", "7:nan", "7:1.25,7:1.5"),
)
def test_reject_invalid_acquisition_time_scale_override(value: str) -> None:
    with pytest.raises(ValueError):
        parse_acquisition_time_scale_overrides(value)
