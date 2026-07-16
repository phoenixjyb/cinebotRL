from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    WholeBodyTrackingConfig,
    bounded_base_references,
    bounded_dls_arm_target,
    bounded_progress_scale,
    continuous_joint_error,
    equilibrium_pitch_from_world_com,
    nearest_equivalent_angle,
    riser_tracking_config,
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


def test_continuous_joint_target_uses_nearest_physical_branch() -> None:
    target = np.deg2rad(401.749127557215)
    measured = np.deg2rad(-317.9332275390625)
    nearest = nearest_equivalent_angle(target, measured)

    assert np.rad2deg(nearest) == pytest.approx(-318.250872442785, abs=1e-9)
    assert np.rad2deg(continuous_joint_error(target, measured)) == pytest.approx(
        -0.3176449037225, abs=1e-9
    )
    assert np.sin(nearest) == pytest.approx(np.sin(target), abs=1e-12)
    assert np.cos(nearest) == pytest.approx(np.cos(target), abs=1e-12)


def test_continuous_joint_helpers_reject_nonfinite_angles() -> None:
    with pytest.raises(ValueError, match="finite"):
        continuous_joint_error(float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite"):
        nearest_equivalent_angle(0.0, float("inf"))


def test_riser_tracking_profile_preserves_limits_and_raises_path_authority() -> None:
    default = WholeBodyTrackingConfig()
    riser = riser_tracking_config()
    assert riser.along_track_kp == 1.6
    assert riser.cross_track_kp == 1.5
    assert riser.yaw_kp == 1.2
    assert riser.maximum_linear_velocity_mps == default.maximum_linear_velocity_mps
    assert riser.maximum_yaw_rate_radps == default.maximum_yaw_rate_radps
    assert riser.progress_error_full_m == default.progress_error_full_m


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


def test_equilibrium_pitch_opposes_forward_com_offset() -> None:
    pitch, com_from_axle = equilibrium_pitch_from_world_com(
        root_position_world_m=np.zeros(3),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        center_of_mass_world_m=np.array([0.1, 0.0, 0.6]),
        wheel_axle_height_m=0.1,
    )
    assert pitch < 0.0
    np.testing.assert_allclose(com_from_axle, [0.1, 0.0, 0.5])
