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


def test_continuity_solver_recovers_alternate_euler_branch() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    root_yaw = 0.3681535
    root_quat = np.array([math.cos(root_yaw / 2.0), 0.0, 0.0, math.sin(root_yaw / 2.0)])
    arm = np.array([0.38171859, 0.49359496, -0.90382045])
    semantic_target = np.array([0.48389383, 0.48185483, 0.51560282, -0.51750982])

    local = kinematics.solve_semantic_attitude(
        root_quat, arm, semantic_target, np.zeros(3), maximum_iterations=100
    )
    recovered = kinematics.solve_semantic_attitude_continuous(
        root_quat, arm, semantic_target, np.zeros(3)
    )

    assert not local.converged
    assert recovered.converged
    assert recovered.orientation_error_rad < math.radians(0.1)


def test_continuity_solver_honors_tightened_physical_joint_bounds() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    expected_gimbal = np.array([0.35, -0.25, 0.20])
    physical_rotation = kinematics.world_rotation(root_quat, ARM_HOME, expected_gimbal)
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical_rotation)
    )
    lower = np.maximum(kinematics.gimbal_lower, np.full(3, -0.1))
    upper = np.minimum(kinematics.gimbal_upper, np.full(3, 0.1))

    result = kinematics.solve_semantic_attitude_continuous(
        root_quat,
        ARM_HOME,
        semantic_target,
        np.zeros(3),
        gimbal_lower_bound=lower,
        gimbal_upper_bound=upper,
    )

    assert np.all(result.gimbal_q >= lower)
    assert np.all(result.gimbal_q <= upper)
    assert not result.converged


def test_local_attitude_solver_stays_on_nominal_euler_branch() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    nominal = np.array([0.35, -0.25, 0.20])
    expected = nominal + np.radians([3.0, -2.0, 4.0])
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    physical_rotation = kinematics.world_rotation(root_quat, ARM_HOME, expected)
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical_rotation)
    )

    result = kinematics.solve_semantic_attitude_near_branch(
        root_quat,
        ARM_HOME,
        semantic_target,
        nominal,
        nominal,
        maximum_joint_offset_rad=math.radians(8.0),
    )

    assert result.converged
    assert np.max(np.abs(result.gimbal_q - nominal)) <= math.radians(8.0) + 1e-12
    np.testing.assert_allclose(result.gimbal_q, expected, atol=math.radians(0.1))


def test_local_attitude_solver_rejects_distant_branch_motion() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    nominal = np.zeros(3)
    distant = np.radians([40.0, 20.0, -30.0])
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    physical_rotation = kinematics.world_rotation(root_quat, ARM_HOME, distant)
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(physical_rotation)
    )

    result = kinematics.solve_semantic_attitude_near_branch(
        root_quat,
        ARM_HOME,
        semantic_target,
        nominal,
        nominal,
        maximum_joint_offset_rad=math.radians(5.0),
    )

    assert not result.converged
    assert np.max(np.abs(result.gimbal_q - nominal)) <= math.radians(5.0) + 1e-12


def test_bounded_attitude_feedback_reduces_camera_error() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    actual = np.zeros(3)
    expected = np.radians([5.0, -3.0, 4.0])
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(
            kinematics.world_rotation(root_quat, ARM_HOME, expected)
        )
    )

    result = kinematics.bounded_attitude_feedback_target(
        root_quat,
        ARM_HOME,
        actual,
        semantic_target,
        nominal_gimbal_q=np.zeros(3),
        previous_correction_q=np.zeros(3),
        dt=0.005,
        gain=0.7,
        time_constant_s=0.0,
    )
    before = result.orientation_error_rad
    after = np.linalg.norm(
        rotation_error_vector(
            kinematics.world_rotation(root_quat, ARM_HOME, result.gimbal_q),
            kinematics.world_rotation(root_quat, ARM_HOME, expected),
        )
    )

    assert after < before
    assert np.max(np.abs(result.correction_q)) <= math.radians(15.0) + 1e-12


def test_attitude_feedback_correction_is_low_pass_filtered() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    expected = np.radians([8.0, 0.0, 0.0])
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(
            kinematics.world_rotation(root_quat, ARM_HOME, expected)
        )
    )
    unfiltered = kinematics.bounded_attitude_feedback_target(
        root_quat,
        ARM_HOME,
        np.zeros(3),
        semantic_target,
        np.zeros(3),
        np.zeros(3),
        0.1,
        time_constant_s=0.0,
    )
    filtered = kinematics.bounded_attitude_feedback_target(
        root_quat,
        ARM_HOME,
        np.zeros(3),
        semantic_target,
        np.zeros(3),
        np.zeros(3),
        0.1,
        time_constant_s=0.1,
    )

    assert np.linalg.norm(filtered.correction_q) < np.linalg.norm(
        unfiltered.correction_q
    )


def test_zero_attitude_feedback_gain_returns_nominal_target() -> None:
    kinematics = UrdfPhysicalCameraKinematics(URDF)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    nominal = np.radians([10.0, -5.0, 3.0])
    actual = np.radians([-4.0, 2.0, 1.0])
    semantic_target = physical_cam_to_semantic_dfr_quat_wxyz(
        matrix_quaternion_wxyz(
            kinematics.world_rotation(root_quat, ARM_HOME, nominal)
        )
    )

    result = kinematics.bounded_attitude_feedback_target(
        root_quat,
        ARM_HOME,
        actual,
        semantic_target,
        nominal,
        previous_correction_q=np.zeros(3),
        dt=0.005,
        gain=0.0,
        time_constant_s=0.0,
    )

    np.testing.assert_allclose(result.gimbal_q, nominal, atol=1e-12)
    np.testing.assert_allclose(result.correction_q, 0.0, atol=1e-12)


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
