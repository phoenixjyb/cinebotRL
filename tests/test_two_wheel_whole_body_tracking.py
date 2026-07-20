from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    WholeBodyTrackingConfig,
    bounded_base_references,
    bounded_camera_recovery_progress_scale,
    bounded_camera_lever_arm_base_target,
    bounded_dls_arm_target,
    bounded_progress_scale,
    continuous_joint_error,
    equilibrium_pitch_from_world_com,
    nearest_equivalent_angle,
    riser_tracking_config,
    select_progress_governor_base_error,
    slew_limited_arm_target,
    summarize_progress_governor_base_error,
    summarize_progress_hold,
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


def test_cross_track_direction_follows_feedback_reverse_command() -> None:
    velocity, yaw_rate, diagnostics = bounded_base_references(
        desired_base_q=np.array([-0.5, 0.5, -0.6]),
        actual_base_q=np.zeros(3),
        feedforward_v_mps=0.01,
        feedforward_wz_radps=0.0,
        config=riser_tracking_config(),
    )

    assert velocity == pytest.approx(-0.4)
    assert yaw_rate == pytest.approx(-0.4)
    assert diagnostics["feedforward_direction"] == 1.0
    assert diagnostics["feedback_motion_direction"] == -1.0
    assert diagnostics["direction_recovery_blend"] == 1.0
    assert diagnostics["motion_direction"] == -1.0


def test_healthy_tracking_preserves_feedforward_cross_track_direction() -> None:
    velocity, yaw_rate, diagnostics = bounded_base_references(
        desired_base_q=np.array([0.009375, 0.1, 0.0]),
        actual_base_q=np.zeros(3),
        feedforward_v_mps=0.01,
        feedforward_wz_radps=0.0,
        config=riser_tracking_config(),
    )

    assert velocity == pytest.approx(0.025)
    assert yaw_rate == pytest.approx(0.15)
    assert diagnostics["feedback_motion_direction"] == pytest.approx(0.5)
    assert diagnostics["direction_recovery_blend"] == 0.0
    assert diagnostics["motion_direction"] == 1.0


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


def test_riser_recovery_velocity_cap_is_symmetric_and_default_off() -> None:
    default = riser_tracking_config()
    capped = riser_tracking_config(maximum_linear_velocity_mps=0.2)

    assert default.maximum_linear_velocity_mps == pytest.approx(0.4)
    for desired_x, expected in ((1.0, 0.2), (-1.0, -0.2)):
        velocity, _, _ = bounded_base_references(
            desired_base_q=np.array([desired_x, 0.0, 0.0]),
            actual_base_q=np.zeros(3),
            feedforward_v_mps=0.0,
            feedforward_wz_radps=0.0,
            config=capped,
        )
        assert velocity == pytest.approx(expected)


@pytest.mark.parametrize("limit", [0.0, -0.1, float("nan")])
def test_base_reference_rejects_invalid_velocity_cap(limit: float) -> None:
    with pytest.raises(ValueError, match="base velocity limits"):
        bounded_base_references(
            desired_base_q=np.zeros(3),
            actual_base_q=np.zeros(3),
            feedforward_v_mps=0.0,
            feedforward_wz_radps=0.0,
            config=riser_tracking_config(maximum_linear_velocity_mps=limit),
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


def test_progress_governor_can_hold_only_when_explicitly_configured() -> None:
    default = riser_tracking_config()
    hold = riser_tracking_config(minimum_progress_scale=0.0)

    assert bounded_progress_scale(0.25, 0.0, default) == pytest.approx(0.1)
    assert bounded_progress_scale(0.25, 0.0, hold) == 0.0
    assert bounded_progress_scale(0.15, 0.0, hold) == pytest.approx(0.5)


def test_progress_governor_base_error_defaults_to_nominal_plan_target() -> None:
    assert select_progress_governor_base_error(
        0.22,
        0.11,
        use_commanded_base_target=False,
    ) == pytest.approx(0.22)


def test_progress_governor_base_error_can_use_compensated_command_target() -> None:
    assert select_progress_governor_base_error(
        0.22,
        0.11,
        use_commanded_base_target=True,
    ) == pytest.approx(0.11)


@pytest.mark.parametrize(
    ("nominal", "commanded"),
    [(-0.01, 0.1), (0.1, -0.01), (np.nan, 0.1), (0.1, np.inf)],
)
def test_progress_governor_base_error_rejects_invalid_evidence(
    nominal: float,
    commanded: float,
) -> None:
    with pytest.raises(ValueError, match="progress-governor base errors"):
        select_progress_governor_base_error(
            nominal,
            commanded,
            use_commanded_base_target=True,
        )


def test_progress_governor_base_error_summary_accepts_bounded_candidate() -> None:
    summary = summarize_progress_governor_base_error(
        np.array([0.20, 0.12]),
        np.array([0.15, 0.10]),
        np.array([0.15, 0.10]),
        use_commanded_base_target=True,
        maximum_command_correction_m=0.05,
        expected_sample_count=2,
    )

    assert summary["progress_base_error_telemetry_observed"] is True
    assert summary["progress_base_error_selected_source_matches"] is True
    assert summary["progress_base_error_command_delta_bounded"] is True
    assert summary["selected_vs_nominal_base_progress_error_abs_max_delta_m"] == pytest.approx(0.05)


def test_progress_governor_base_error_summary_rejects_forged_selection() -> None:
    summary = summarize_progress_governor_base_error(
        np.array([0.20, 0.12]),
        np.array([0.15, 0.10]),
        np.array([0.20, 0.12]),
        use_commanded_base_target=True,
        maximum_command_correction_m=0.05,
        expected_sample_count=2,
    )

    assert summary["progress_base_error_selected_source_matches"] is False
    assert summary["progress_base_error_telemetry_observed"] is False


def test_progress_governor_base_error_summary_rejects_over_bound_delta() -> None:
    summary = summarize_progress_governor_base_error(
        np.array([0.20, 0.12]),
        np.array([0.14, 0.10]),
        np.array([0.14, 0.10]),
        use_commanded_base_target=True,
        maximum_command_correction_m=0.05,
        expected_sample_count=2,
    )

    assert summary["progress_base_error_command_delta_bounded"] is False
    assert summary["progress_base_error_telemetry_observed"] is False


@pytest.mark.parametrize("minimum", [-0.01, 1.01])
def test_progress_governor_rejects_invalid_minimum(minimum: float) -> None:
    with pytest.raises(ValueError, match="invalid progress governor"):
        bounded_progress_scale(
            0.25,
            0.0,
            riser_tracking_config(minimum_progress_scale=minimum),
        )


def test_progress_hold_summary_counts_bounded_segments() -> None:
    summary = summarize_progress_hold(
        np.array([1.0, 0.5, 0.0, 0.0, 0.2, 0.0, 1.0])
    )

    assert summary["progress_hold_step_count"] == 3
    assert summary["progress_hold_ratio"] == pytest.approx(3.0 / 7.0)
    assert summary["progress_hold_segment_count"] == 2


@pytest.mark.parametrize(
    "values",
    [np.array([]), np.array([0.0, np.nan]), np.array([-0.1]), np.array([1.1])],
)
def test_progress_hold_summary_rejects_invalid_evidence(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="progress scales"):
        summarize_progress_hold(values)


def test_camera_recovery_governor_is_continuous_bounded_and_saturation_gated() -> None:
    config = riser_tracking_config()
    assert bounded_camera_recovery_progress_scale(0.20, False, config) == 1.0
    assert bounded_camera_recovery_progress_scale(0.12, True, config) == 1.0
    assert bounded_camera_recovery_progress_scale(0.155, True, config) == pytest.approx(
        0.2
    )
    middle = bounded_camera_recovery_progress_scale(0.1425, True, config)
    assert middle == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("error", "config"),
    [
        (-0.01, riser_tracking_config()),
        (
            0.15,
            riser_tracking_config(
                camera_recovery_error_start_m=0.16,
                camera_recovery_error_full_m=0.13,
            ),
        ),
        (
            0.15,
            riser_tracking_config(minimum_camera_recovery_scale=0.0),
        ),
    ],
)
def test_camera_recovery_governor_rejects_invalid_contract(
    error: float, config: WholeBodyTrackingConfig
) -> None:
    with pytest.raises(ValueError):
        bounded_camera_recovery_progress_scale(error, True, config)


def test_camera_lever_arm_compensation_is_zero_when_offsets_match() -> None:
    target, diagnostics = bounded_camera_lever_arm_base_target(
        desired_base_q=np.array([1.0, 2.0, 0.3]),
        actual_base_q=np.array([3.0, 4.0, -0.2]),
        target_camera_position_world_m=np.array([1.1, 1.9, 1.8]),
        actual_camera_position_world_m=np.array([3.1, 3.9, 1.6]),
        gain=1.0,
        maximum_correction_m=0.05,
    )

    np.testing.assert_allclose(target, [1.0, 2.0, 0.3])
    assert diagnostics["correction_norm_m"] == pytest.approx(0.0)
    assert diagnostics["saturated"] is False


def test_disabled_camera_lever_arm_compensation_preserves_planned_base() -> None:
    desired = np.array([0.4, -0.2, 0.7])
    target, diagnostics = bounded_camera_lever_arm_base_target(
        desired_base_q=desired,
        actual_base_q=np.array([1.0, 1.0, -0.4]),
        target_camera_position_world_m=np.array([0.4, -0.2, 1.8]),
        actual_camera_position_world_m=np.array([1.08, 0.94, 1.6]),
        gain=0.0,
        maximum_correction_m=0.05,
    )

    np.testing.assert_array_equal(target, desired)
    assert diagnostics["lever_error_norm_m"] == pytest.approx(0.1)
    assert diagnostics["raw_correction_norm_m"] == pytest.approx(0.0)
    assert diagnostics["correction_norm_m"] == pytest.approx(0.0)
    assert diagnostics["saturated"] is False


def test_camera_lever_arm_compensation_preserves_yaw_and_norm_bounds_xy() -> None:
    target, diagnostics = bounded_camera_lever_arm_base_target(
        desired_base_q=np.array([0.0, 0.0, 0.7]),
        actual_base_q=np.array([1.0, 1.0, -0.4]),
        target_camera_position_world_m=np.array([0.0, 0.0, 1.8]),
        actual_camera_position_world_m=np.array([1.08, 0.94, 1.6]),
        gain=1.0,
        maximum_correction_m=0.05,
    )

    np.testing.assert_allclose(target, [-0.04, 0.03, 0.7], atol=1e-12)
    np.testing.assert_allclose(
        diagnostics["lever_error_xy_m"], [0.08, -0.06], atol=1e-12
    )
    assert diagnostics["raw_correction_norm_m"] == pytest.approx(0.1)
    assert diagnostics["correction_norm_m"] == pytest.approx(0.05)
    assert diagnostics["saturated"] is True


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"gain": 1.01}, "gain"),
        ({"maximum_correction_m": 0.0}, "maximum"),
        ({"actual_camera_position_world_m": np.ones(2)}, "shape"),
        (
            {"actual_camera_position_world_m": np.array([np.nan, 0.0, 1.0])},
            "finite",
        ),
    ],
)
def test_camera_lever_arm_compensation_rejects_invalid_contract(
    updates: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "desired_base_q": np.zeros(3),
        "actual_base_q": np.zeros(3),
        "target_camera_position_world_m": np.array([0.0, 0.0, 1.0]),
        "actual_camera_position_world_m": np.array([0.0, 0.0, 1.0]),
        "gain": 1.0,
        "maximum_correction_m": 0.05,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        bounded_camera_lever_arm_base_target(**values)


def test_equilibrium_pitch_opposes_forward_com_offset() -> None:
    pitch, com_from_axle = equilibrium_pitch_from_world_com(
        root_position_world_m=np.zeros(3),
        root_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        center_of_mass_world_m=np.array([0.1, 0.0, 0.6]),
        wheel_axle_height_m=0.1,
    )
    assert pitch < 0.0
    np.testing.assert_allclose(com_from_axle, [0.1, 0.0, 0.5])
