import math
from pathlib import Path

import numpy as np

from rl_platform.tasks.two_wheel_balance.camera_attitude import (
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (
    RISER_CAMERA_CHAIN_JOINTS,
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_attitude import (
    compose_semantic_dfr_rotation,
    rs4_command_to_proxy_joint_order,
)
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.urdf"


def test_riser_camera_chain_is_prismatic_plus_physical_gimbal() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    movable = tuple(item.name for item in kinematics.chain if item.joint_type != "fixed")
    assert movable == RISER_CAMERA_CHAIN_JOINTS
    assert kinematics.riser_lower == 0.0
    assert kinematics.riser_upper == 1.2


def test_camera_height_contract_is_exact_at_zero_gimbal() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    lower = kinematics.relative_transform(0.0, np.zeros(3))[:3, 3]
    upper = kinematics.relative_transform(1.2, np.zeros(3))[:3, 3]
    np.testing.assert_allclose(lower, np.array([0.0, 0.0, 0.6]), atol=1e-8)
    np.testing.assert_allclose(upper, np.array([0.0, 0.0, 1.8]), atol=1e-8)


def test_position_inverse_recovers_base_xy_and_riser() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    expected_state = np.array([1.25, -0.73, 0.61, 0.47])
    gimbal = np.array([0.31, -0.16, 0.24])
    target = kinematics.world_transform(
        expected_state[:3], expected_state[3], gimbal
    )[:3, 3]
    result = kinematics.solve_position(target, expected_state[2], gimbal)
    assert result.reachable
    np.testing.assert_allclose(result.base_xy_yaw_riser, expected_state, atol=1e-8)
    assert result.position_error_m < 1e-10


def test_position_inverse_rejects_height_outside_riser_range() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    result = kinematics.solve_position(
        np.array([0.0, 0.0, 1.95]), 0.0, np.zeros(3)
    )
    assert not result.reachable
    assert result.base_xy_yaw_riser[3] == 1.2
    assert result.position_error_m > 0.14


def test_semantic_attitude_ik_recovers_known_physical_gimbal_pose() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    expected_gimbal = np.array([0.35, -0.25, 0.20])
    physical = kinematics.world_rotation(root_quat, 0.42, expected_gimbal)
    semantic = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical)
    )
    result = kinematics.solve_semantic_attitude(
        root_quat, 0.42, semantic, np.zeros(3)
    )
    achieved = kinematics.world_rotation(root_quat, 0.42, result.gimbal_q)
    assert result.converged
    assert np.linalg.norm(rotation_error_vector(achieved, physical)) < math.radians(0.1)


def test_robust_attitude_ik_recovers_far_pose_without_branch_hint() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    expected_gimbal = np.array([2.6, -1.4, 1.2])
    physical = kinematics.world_rotation(root_quat, 0.42, expected_gimbal)
    semantic = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical)
    )
    result = kinematics.solve_semantic_attitude_robust(
        root_quat, 0.42, semantic, np.zeros(3)
    )
    achieved = kinematics.world_rotation(root_quat, 0.42, result.gimbal_q)
    assert result.converged
    assert np.linalg.norm(rotation_error_vector(achieved, physical)) < math.radians(0.1)


def test_robust_attitude_ik_keeps_continuous_axis_unwrapped_near_seed() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    expected_gimbal = np.deg2rad([14.0, -11.0, -187.0])
    seed = np.deg2rad([9.0, -8.0, -162.0])
    physical = kinematics.world_rotation(root_quat, 0.84, expected_gimbal)
    semantic = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical)
    )
    result = kinematics.solve_semantic_attitude_robust(
        root_quat, 0.84, semantic, seed
    )
    achieved = kinematics.world_rotation(root_quat, 0.84, result.gimbal_q)
    assert result.converged
    assert result.gimbal_q[2] < -math.pi
    assert np.max(np.abs(result.gimbal_q - seed)) < math.radians(30.0)
    assert np.linalg.norm(rotation_error_vector(achieved, physical)) < math.radians(0.1)


def test_rs4_proxy_joint_fk_matches_deployed_semantic_command_mapping() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    basis = Rotation.from_quat(
        [
            0.37126688383868894,
            0.6136484680880162,
            -0.4508086827044413,
            0.5313830917297113,
        ]
    ).as_matrix()
    command = np.array([0.2, -0.3, 0.4])
    base_yaw = -0.25
    world_basis = Rotation.from_euler("z", base_yaw).as_matrix() @ basis
    semantic_rotation = compose_semantic_dfr_rotation(world_basis, command)
    expected_physical = quaternion_matrix_wxyz(
        semantic_dfr_to_physical_cam_quat_wxyz(
            matrix_quaternion_wxyz(semantic_rotation)
        )
    )
    root_quat = np.array(
        [math.cos(base_yaw / 2.0), 0.0, 0.0, math.sin(base_yaw / 2.0)]
    )
    achieved = kinematics.world_rotation(
        root_quat, 0.0, rs4_command_to_proxy_joint_order(command)
    )
    np.testing.assert_allclose(achieved, expected_physical, atol=2e-8)


def test_semantic_attitude_ik_compensates_tilted_balancing_base() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    upright_root = Rotation.from_euler("ZYX", [0.35, 0.0, 0.0]).as_quat()
    upright_root = upright_root[[3, 0, 1, 2]]
    initial_proxy = np.array([0.2, -0.15, 0.4])
    physical_target = kinematics.world_rotation(upright_root, 0.4, initial_proxy)
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical_target)
    )
    tilted_root = Rotation.from_euler(
        "ZYX", [0.35, math.radians(-4.0), math.radians(5.0)]
    ).as_quat()
    tilted_root = tilted_root[[3, 0, 1, 2]]

    result = kinematics.solve_semantic_attitude_robust(
        tilted_root, 0.4, semantic_target, initial_proxy
    )
    achieved = kinematics.world_rotation(tilted_root, 0.4, result.gimbal_q)

    assert result.converged
    assert not np.allclose(result.gimbal_q, initial_proxy)
    assert np.linalg.norm(rotation_error_vector(achieved, physical_target)) < math.radians(0.1)
