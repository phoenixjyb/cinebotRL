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
    quaternion_slerp_wxyz,
    regenerate_acquisition_attitude_prefix,
    regenerate_acquisition_prefix,
    source_body_velocities,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
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


def write_sparse_teacher(
    path: Path,
    runtime_approved: bool = False,
    teacher_quality_passed: bool = True,
    include_integrity_contract: bool = True,
) -> None:
    current = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0], [0.1, 0.02, 0.0, 0.1, 1.0, 2.0]])
    next_q = np.array([[0.1, 0.02, 0.0, 0.1, 1.0, 2.0], [0.2, 0.04, 0.0, 0.2, 1.0, 2.0]])
    values = {
        "schema": "cinebotrl_gik_monorepo_ee1_split_teacher_v2",
        "trajectory_integrity_passed": np.bool_(True),
        "valid_for_training": np.bool_(teacher_quality_passed),
        "valid_for_candidate_training": np.bool_(teacher_quality_passed),
        "teacher_quality_passed": np.bool_(teacher_quality_passed),
        "teacher_approved_envelope": np.bool_(teacher_quality_passed),
        "runtime_approved": np.bool_(runtime_approved),
        "position_target_link": "ee1_tool",
        "quaternion_order": "wxyz",
        "action_order": np.asarray(EXPECTED_ACTION_ORDER),
        "q_current_base_arm_6": current,
        "q_next_base_arm_6": next_q,
        "time_s": np.array([0.1, 0.2]),
        "dt_s": np.array([0.1, 0.1]),
        "gimbal_attitude_target_world_dfr_quat_wxyz": np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        "desired_time_full_s": np.array([0.0, 0.1, 0.2]),
        "desired_position_full_m": np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
        ),
        "desired_attitude_full_world_dfr_quat_wxyz": np.array(
            [[1.0, 0.0, 0.0, 0.0]] * 3
        ),
        "source_pose_count": np.int32(3),
        "reference_pose_count": np.int32(3),
        "state_count": np.int32(3),
        "action_count": np.int32(2),
        "episode_index": np.int32(1),
    }
    if include_integrity_contract:
        values["trajectory_integrity_contract"] = "exact_source_v1"
    np.savez_compressed(path, **values)


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


def test_sparse_teacher_rejects_pre_exact_source_lineage(tmp_path: Path) -> None:
    path = tmp_path / "teacher.npz"
    write_sparse_teacher(path, include_integrity_contract=False)
    with pytest.raises(ValueError, match="trajectory_integrity_contract"):
        load_sparse_teacher(path)


def test_integrity_canary_is_not_quality_qualified_teacher(tmp_path: Path) -> None:
    path = tmp_path / "teacher.npz"
    write_sparse_teacher(path, teacher_quality_passed=False)
    with pytest.raises(ValueError, match="teacher not valid"):
        load_sparse_teacher(path)


def test_sparse_teacher_rejects_transition_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "teacher.npz"
    write_sparse_teacher(path)
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values["action_count"] = np.int32(1)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="N-1 mismatch"):
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


def test_regenerate_attitude_acquisition_uses_semantic_dfr_slerp(tmp_path: Path) -> None:
    path = tmp_path / "0001_case.json"
    write_full_reference(path)
    full = load_full_reference(path)
    home = np.array([2.0**-0.5, 0.0, 0.0, -2.0**-0.5])
    attitudes, semantic_start = regenerate_acquisition_attitude_prefix(
        full, home, acquisition_end_index=1
    )
    assert abs(float(np.dot(attitudes[0], home))) > 1.0 - 1e-12
    assert abs(float(np.dot(attitudes[1], full.attitudes_wxyz[1]))) > 1.0 - 1e-12
    assert abs(float(np.dot(semantic_start, full.attitudes_wxyz[1]))) > 1.0 - 1e-12
    np.testing.assert_allclose(attitudes[1:], full.attitudes_wxyz[1:])


def test_quaternion_slerp_uses_shortest_path_and_normalizes() -> None:
    start = np.array([1.0, 0.0, 0.0, 0.0])
    end = -np.array([2.0**-0.5, 0.0, 0.0, 2.0**-0.5])
    values = quaternion_slerp_wxyz(start, end, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0)
    assert abs(float(np.dot(values[-1], end))) > 1.0 - 1e-12


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


def test_source_ee1_position_fk_accepts_zeroed_semantic_frame_joints() -> None:
    source_urdf = (
        Path(__file__).resolve().parents[1]
        / "assets_own/sources/recomoProto2-1190_moveit_aa463a.urdf"
    )
    kinematics = UrdfPositionKinematics(
        source_urdf,
        passive_joint_positions={
            "ee1_level_pitch": 0.0,
            "ee1_rot_z": 0.0,
            "ee1_rot_y": 0.0,
            "ee1_rot_x": 0.0,
        },
    )
    position = kinematics.position(
        np.array([0.58322558, 0.53503527, 0.3681535, 0.38171859, 0.49359496, -0.90382045])
    )
    np.testing.assert_allclose(position, [0.0, 0.0, 1.35], atol=1e-7)
