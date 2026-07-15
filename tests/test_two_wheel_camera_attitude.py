import math
from pathlib import Path

import numpy as np

from rl_platform.tasks.two_wheel_balance.camera_attitude import (
    UrdfPhysicalCameraKinematics,
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets_own/recomoProto2_two_wheel_whole_body_attitude/recomoProto2_two_wheel_whole_body_attitude.urdf"
ARM_HOME = np.array([0.0, math.pi / 2.0, 3.0 * math.pi / 4.0])


def test_option_b_semantic_dfr_frame_conversion_roundtrip() -> None:
    dfr = np.array([0.8, -0.1, 0.3, 0.5])
    dfr /= np.linalg.norm(dfr)
    physical = semantic_dfr_to_physical_cam_quat_wxyz(dfr)
    recovered = physical_cam_to_semantic_dfr_quat_wxyz(physical)
    assert abs(float(np.dot(dfr, recovered))) > 1.0 - 1e-12


def test_physical_camera_fk_and_attitude_ik_recover_known_gimbal_pose() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    expected_gimbal = np.array([0.35, -0.25, 0.20])
    physical_rotation = kinematics.world_rotation(root_quat, ARM_HOME, expected_gimbal)
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical_rotation)
    )
    result = kinematics.solve_semantic_attitude(
        root_quat, ARM_HOME, semantic_target, np.zeros(3)
    )
    achieved = kinematics.world_rotation(root_quat, ARM_HOME, result.gimbal_q)
    error = np.linalg.norm(rotation_error_vector(achieved, physical_rotation))
    assert result.converged
    assert error < math.radians(0.1)


def test_physical_camera_fk_matches_option_b_target_at_home() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    physical_rotation = kinematics.world_rotation(
        np.array([1.0, 0.0, 0.0, 0.0]), ARM_HOME, np.zeros(3)
    )
    semantic = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical_rotation)
    )
    target = quaternion_matrix_wxyz(semantic_dfr_to_physical_cam_quat_wxyz(semantic))
    np.testing.assert_allclose(target, physical_rotation, atol=1e-10)


def test_gimbal_gravity_effort_is_finite_and_within_declared_limit_at_home() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    effort = kinematics.gimbal_gravitational_effort_nm(
        np.array([1.0, 0.0, 0.0, 0.0]), ARM_HOME, np.zeros(3)
    )
    assert np.isfinite(effort).all()
    assert np.max(np.abs(effort)) < 10.0
