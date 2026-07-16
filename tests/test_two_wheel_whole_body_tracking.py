from pathlib import Path

import numpy as np

from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    GimbalRootAttitudeFeedbackConfig,
    WholeBodyTrackingConfig,
    bounded_attitude_progress_scale,
    bounded_balance_progress_scale,
    bounded_base_references,
    bounded_phase_progress_scale,
    bounded_dls_arm_target,
    bounded_progress_scale,
    bounded_semantic_arm_target,
    bounded_task_space_base_target,
    equilibrium_pitch_from_world_com,
    filtered_gimbal_root_attitude_command,
    phase_scaled_feedforward,
    quaternion_from_roll_pitch_yaw_wxyz,
    quaternion_from_pitch_yaw_wxyz,
    roll_pitch_from_quaternion_wxyz,
    slew_limited_planar_offset,
    slew_limited_arm_target,
    yaw_from_quaternion_wxyz,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_whole_body"
    / "recomoProto2_two_wheel_whole_body.urdf"
)


def test_yaw_and_base_feedback_signs() -> None:
    assert yaw_from_quaternion_wxyz(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0
    velocity, yaw_rate, diagnostics = bounded_base_references(
        desired_base_q=np.array([0.2, 0.1, 0.1]),
        actual_base_q=np.zeros(3),
        feedforward_v_mps=0.05,
        feedforward_wz_radps=0.0,
        config=WholeBodyTrackingConfig(),
    )
    assert velocity > 0.05
    assert yaw_rate > 0.0
    assert diagnostics["along_track_error_m"] > 0.0
    assert diagnostics["cross_track_error_m"] > 0.0


def test_roll_pitch_quaternion_extraction() -> None:
    roll, pitch = np.radians([2.0, 9.0])
    yaw = np.radians(-17.0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    np.testing.assert_allclose(
        roll_pitch_from_quaternion_wxyz(quaternion), [roll, pitch], atol=1e-12
    )


def test_pitch_yaw_quaternion_round_trip() -> None:
    pitch = np.radians(9.5)
    yaw = np.radians(-31.0)
    quaternion = quaternion_from_pitch_yaw_wxyz(pitch, yaw)

    np.testing.assert_allclose(
        roll_pitch_from_quaternion_wxyz(quaternion), [0.0, pitch], atol=1e-12
    )
    np.testing.assert_allclose(yaw_from_quaternion_wxyz(quaternion), yaw, atol=1e-12)


def test_filtered_gimbal_root_attitude_command_blends_measured_error() -> None:
    actual = quaternion_from_roll_pitch_yaw_wxyz(
        *np.radians([2.0, 6.0, -10.0])
    )
    command, filtered_error, applied_correction = (
        filtered_gimbal_root_attitude_command(
            actual,
            nominal_pitch_rad=np.radians(2.0),
            nominal_yaw_rad=np.radians(-2.0),
            previous_filtered_error_rpy_rad=np.zeros(3),
            dt=0.005,
            config=GimbalRootAttitudeFeedbackConfig(
                gain=0.5, time_constant_s=0.0
            ),
        )
    )

    np.testing.assert_allclose(np.degrees(filtered_error), [2.0, 4.0, -8.0])
    np.testing.assert_allclose(
        np.degrees(applied_correction), [1.0, 2.0, -4.0]
    )
    np.testing.assert_allclose(
        np.degrees(roll_pitch_from_quaternion_wxyz(command)), [1.0, 4.0]
    )
    np.testing.assert_allclose(
        np.degrees(yaw_from_quaternion_wxyz(command)), -6.0
    )


def test_filtered_gimbal_root_attitude_command_bounds_and_filters_error() -> None:
    actual = quaternion_from_roll_pitch_yaw_wxyz(
        0.0, np.radians(-30.0), np.radians(30.0)
    )
    _, filtered_error, applied_correction = filtered_gimbal_root_attitude_command(
        actual,
        nominal_pitch_rad=0.0,
        nominal_yaw_rad=0.0,
        previous_filtered_error_rpy_rad=np.zeros(3),
        dt=0.1,
        config=GimbalRootAttitudeFeedbackConfig(
            gain=0.5,
            time_constant_s=0.1,
            maximum_error_rad=np.radians(12.0),
        ),
    )

    expected_filtered_deg = 12.0 * (1.0 - np.exp(-1.0))
    np.testing.assert_allclose(
        np.degrees(filtered_error),
        [0.0, -expected_filtered_deg, expected_filtered_deg],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.degrees(applied_correction),
        [0.0, -0.5 * expected_filtered_deg, 0.5 * expected_filtered_deg],
        atol=1e-12,
    )


def test_dls_target_reduces_small_position_error() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    state = np.array([0.0, 0.0, 0.0, 0.0, 1.2, 2.0])
    actual_position = kinematics.position(state)
    target_position = actual_position + np.array([0.005, -0.003, 0.004])
    arm_target, diagnostics = bounded_dls_arm_target(
        kinematics,
        actual_base_q=state[:3],
        actual_arm_q=state[3:],
        nominal_arm_q=state[3:],
        target_position_world_m=target_position,
        actual_position_world_m=actual_position,
        config=WholeBodyTrackingConfig(),
    )
    corrected_position = kinematics.position(np.concatenate((state[:3], arm_target)))
    assert np.linalg.norm(corrected_position - target_position) < np.linalg.norm(
        actual_position - target_position
    )
    assert np.max(np.abs(diagnostics["target_correction_rad"])) <= 0.35


def test_semantic_arm_feedback_is_phase_gated_and_slew_limited() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    config = WholeBodyTrackingConfig(maximum_arm_target_rate_radps=0.5)
    base = np.zeros(3)
    arm = np.array([0.0, 1.2, 2.0])
    actual_position = kinematics.position(np.concatenate((base, arm)))
    target_position = actual_position + np.array([0.02, 0.0, 0.01])

    acquisition_target, acquisition_correction = bounded_semantic_arm_target(
        kinematics,
        base,
        arm,
        arm,
        arm,
        target_position,
        actual_position,
        dt=0.02,
        semantic_feedback_enabled=False,
        config=config,
    )
    np.testing.assert_allclose(acquisition_target, arm)
    np.testing.assert_allclose(acquisition_correction, 0.0)

    semantic_target, semantic_correction = bounded_semantic_arm_target(
        kinematics,
        base,
        arm,
        arm,
        arm,
        target_position,
        actual_position,
        dt=0.02,
        semantic_feedback_enabled=True,
        config=config,
    )
    assert np.linalg.norm(semantic_correction) > 0.0
    assert np.max(np.abs(semantic_correction)) <= 0.01 + 1e-12
    assert not np.allclose(semantic_target, arm)


def test_task_space_base_target_absorbs_horizontal_root_tilt_displacement() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    desired_base = np.array([0.4, -0.2, 0.3])
    desired_arm = np.array([-0.1, 1.1, 1.2])
    planar_position = kinematics.position(
        np.concatenate((desired_base, desired_arm))
    )
    retarget_residual = np.array([0.01, -0.02, 0.0])
    target_position = planar_position + retarget_residual
    root_tilt_displacement = np.array([0.16, -0.01, -0.04])

    target, diagnostics = bounded_task_space_base_target(
        kinematics,
        desired_base,
        desired_arm,
        target_position,
        root_tilt_displacement,
        maximum_offset_m=0.4,
    )

    corrected_physical_position = (
        kinematics.position(np.concatenate((target, desired_arm)))
        + root_tilt_displacement
    )
    np.testing.assert_allclose(
        corrected_physical_position[:2], target_position[:2], atol=1e-12
    )
    np.testing.assert_allclose(
        diagnostics["requested_offset_world_m"], [-0.15, -0.01]
    )


def test_task_space_base_target_and_slew_are_bounded() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    desired_base = np.zeros(3)
    desired_arm = np.array([0.0, 1.2, 2.0])
    target_position = kinematics.position(
        np.concatenate((desired_base, desired_arm))
    )
    target, diagnostics = bounded_task_space_base_target(
        kinematics,
        desired_base,
        desired_arm,
        target_position,
        root_tilt_displacement_world_m=np.array([1.0, 0.0, 0.0]),
        maximum_offset_m=0.4,
    )
    np.testing.assert_allclose(target[:2], [-0.4, 0.0])
    assert diagnostics["offset_saturated"] == 1.0

    offset = slew_limited_planar_offset(
        requested_offset_world_m=target[:2],
        previous_offset_world_m=np.zeros(2),
        dt=0.1,
        maximum_rate_mps=0.2,
    )
    np.testing.assert_allclose(offset, [-0.02, 0.0])


def test_arm_target_slew_limit() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    config = WholeBodyTrackingConfig(maximum_arm_target_rate_radps=0.5)
    previous = np.array([0.0, 1.0, 2.0])
    target = slew_limited_arm_target(
        np.array([1.0, -1.0, 0.0]),
        previous,
        dt=0.02,
        kinematics=kinematics,
        config=config,
    )
    np.testing.assert_allclose(target, previous + np.array([0.01, -0.01, -0.01]))


def test_progress_governor_slows_on_tracking_error() -> None:
    config = WholeBodyTrackingConfig()
    assert bounded_progress_scale(0.0, 0.0, config) == 1.0
    np.testing.assert_allclose(bounded_progress_scale(0.25, 0.0, config), 0.1)
    middle = bounded_progress_scale(0.15, 0.0, config)
    assert 0.1 < middle < 1.0


def test_progress_governor_can_fully_pause() -> None:
    config = WholeBodyTrackingConfig(minimum_progress_scale=0.0)
    assert bounded_progress_scale(0.25, 0.0, config) == 0.0


def test_acquisition_progress_ignores_internal_base_compensation_error() -> None:
    config = WholeBodyTrackingConfig(
        progress_error_full_m=0.15, minimum_progress_scale=0.0
    )
    acquisition = bounded_phase_progress_scale(0.16, 0.08, True, config)
    semantic = bounded_phase_progress_scale(0.16, 0.08, False, config)

    assert acquisition > 0.0
    assert semantic == 0.0


def test_balance_progress_governor_preserves_pitch_reserve() -> None:
    slowdown_start = np.radians(10.5)
    full_stop = np.radians(11.5)
    assert bounded_balance_progress_scale(0.0, slowdown_start, full_stop) == 1.0
    np.testing.assert_allclose(
        bounded_balance_progress_scale(
            np.radians(-11.0), slowdown_start, full_stop
        ),
        0.5,
    )
    assert (
        bounded_balance_progress_scale(
            np.radians(12.0), slowdown_start, full_stop
        )
        == 0.0
    )


def test_balance_progress_governor_rejects_invalid_thresholds() -> None:
    with np.testing.assert_raises(ValueError):
        bounded_balance_progress_scale(
            np.radians(10.0), np.radians(11.5), np.radians(10.5)
        )


def test_attitude_progress_governor_pauses_at_quality_gate() -> None:
    slowdown_start = np.radians(4.0)
    full_stop = np.radians(8.0)
    assert (
        bounded_attitude_progress_scale(
            np.radians(2.0), slowdown_start, full_stop
        )
        == 1.0
    )
    np.testing.assert_allclose(
        bounded_attitude_progress_scale(
            np.radians(6.0), slowdown_start, full_stop
        ),
        0.5,
    )
    assert (
        bounded_attitude_progress_scale(
            np.radians(9.0), slowdown_start, full_stop
        )
        == 0.0
    )


def test_attitude_progress_governor_rejects_negative_error() -> None:
    with np.testing.assert_raises(ValueError):
        bounded_attitude_progress_scale(-0.1, 0.2, 0.3)


def test_phase_feedforward_scales_with_governed_progress() -> None:
    velocity, yaw_rate = phase_scaled_feedforward(0.4, -0.2, 0.25)
    assert velocity == 0.1
    assert yaw_rate == -0.05


def test_phase_feedforward_rejects_invalid_progress() -> None:
    with np.testing.assert_raises(ValueError):
        phase_scaled_feedforward(0.4, 0.2, 1.01)


def test_equilibrium_pitch_opposes_forward_com_offset() -> None:
    pitch, com_from_axle = equilibrium_pitch_from_world_com(
        root_position_world_m=np.zeros(3),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        center_of_mass_world_m=np.array([0.1, 0.0, 0.6]),
        wheel_axle_height_m=0.1,
    )
    assert pitch < 0.0
    np.testing.assert_allclose(com_from_axle, [0.1, 0.0, 0.5])
