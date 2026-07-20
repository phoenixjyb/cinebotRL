import pytest

from rl_platform.tasks.two_wheel_balance.riser_recovery_evidence import (
    LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA,
    RECOVERY_TELEMETRY_SCHEMA,
    VELOCITY_FEEDBACK_TELEMETRY_SCHEMA,
    LongitudinalAuthorityTelemetryAccumulator,
    RecoveryTelemetryAccumulator,
    VelocityFeedbackTelemetryAccumulator,
)


def test_healthy_samples_remain_inactive_and_command_neutral() -> None:
    telemetry = RecoveryTelemetryAccumulator()
    for _ in range(20):
        telemetry.step(
            recovery_blend=0.0,
            motion_direction=1.0,
            feedback_motion_direction=-1.0,
            candidate_yaw_rate_rad_s=0.1,
            legacy_yaw_rate_rad_s=0.1,
            maximum_yaw_rate_rad_s=0.4,
        )

    summary = telemetry.summary()
    assert summary["schema"] == RECOVERY_TELEMETRY_SCHEMA
    assert summary["policy_rate_sample_count"] == 20
    assert summary["activation_step_count"] == 0
    assert summary["activation_segment_count"] == 0
    assert summary["motion_direction_sign_change_count"] == 0
    assert summary["candidate_vs_legacy_yaw_delta_abs_max_rad_s"] == 0.0


def test_synthetic_recovery_counts_activation_chatter_and_saturation() -> None:
    telemetry = RecoveryTelemetryAccumulator()
    samples = [
        (0.0, 1.0, 1.0, 0.1, 0.1),
        (0.5, 1.0, 1.0, 0.2, 0.1),
        (1.0, -1.0, -1.0, -0.4, 0.2),
        (1.0, 1.0, 1.0, 0.4, -0.4),
        (0.0, 1.0, 1.0, 0.0, 0.0),
        (0.25, -1.0, -1.0, -0.3, -0.1),
    ]
    for blend, motion, feedback, candidate, legacy in samples:
        telemetry.step(
            recovery_blend=blend,
            motion_direction=motion,
            feedback_motion_direction=feedback,
            candidate_yaw_rate_rad_s=candidate,
            legacy_yaw_rate_rad_s=legacy,
            maximum_yaw_rate_rad_s=0.4,
        )

    summary = telemetry.summary()
    assert summary["policy_rate_sample_count"] == 6
    assert summary["activation_step_count"] == 4
    assert summary["full_authority_step_count"] == 2
    assert summary["activation_segment_count"] == 2
    assert summary["motion_direction_sign_change_count"] == 2
    assert summary["feedback_direction_sign_change_count"] == 2
    assert summary["consecutive_active_motion_direction_chatter_count"] == 2
    assert summary["candidate_yaw_saturation_step_count"] == 2
    assert summary["legacy_yaw_saturation_step_count"] == 1
    assert summary["candidate_vs_legacy_delta_nonzero_step_count"] == 4
    assert summary["candidate_vs_legacy_yaw_delta_abs_max_rad_s"] == pytest.approx(
        0.8
    )
    assert summary["recovery_blend_max"] == 1.0


def test_recovery_telemetry_rejects_invalid_samples() -> None:
    with pytest.raises(ValueError, match="deadband"):
        RecoveryTelemetryAccumulator(direction_deadband=0.0)

    telemetry = RecoveryTelemetryAccumulator()
    with pytest.raises(ValueError, match="invalid"):
        telemetry.step(
            recovery_blend=1.1,
            motion_direction=1.0,
            feedback_motion_direction=1.0,
            candidate_yaw_rate_rad_s=0.0,
            legacy_yaw_rate_rad_s=0.0,
            maximum_yaw_rate_rad_s=0.4,
        )


def test_velocity_feedback_telemetry_localizes_false_wheel_tracking() -> None:
    telemetry = VelocityFeedbackTelemetryAccumulator()
    telemetry.step(
        root_velocity_mps=-0.4,
        wheel_velocity_mps=-0.4,
        effective_reference_mps=-0.4,
        pitch_reference_rad=-0.02,
        total_pitch_reference_rad=-0.01,
        applied_pitch_bias_rad=0.01,
        common_action=0.2,
    )
    telemetry.step(
        root_velocity_mps=0.05,
        wheel_velocity_mps=-0.38,
        effective_reference_mps=-0.4,
        pitch_reference_rad=-0.1,
        total_pitch_reference_rad=-0.08,
        applied_pitch_bias_rad=0.02,
        common_action=0.5,
    )
    summary = telemetry.summary()
    assert summary["schema"] == VELOCITY_FEEDBACK_TELEMETRY_SCHEMA
    assert summary["policy_rate_sample_count"] == 2
    assert summary["opposite_direction_step_count"] == 1
    assert summary["wheel_false_tracking_step_count"] == 1
    assert summary["root_reference_error_rms_mps"] > summary[
        "wheel_reference_error_rms_mps"
    ]
    assert summary["root_wheel_mismatch_abs_max_mps"] == pytest.approx(0.43)
    assert summary["total_pitch_reference_abs_max_rad"] == pytest.approx(0.08)


def test_velocity_feedback_telemetry_rejects_nonfinite_samples() -> None:
    telemetry = VelocityFeedbackTelemetryAccumulator()
    with pytest.raises(ValueError, match="finite"):
        telemetry.step(
            root_velocity_mps=float("nan"),
            wheel_velocity_mps=0.0,
            effective_reference_mps=0.0,
            pitch_reference_rad=0.0,
            total_pitch_reference_rad=0.0,
            applied_pitch_bias_rad=0.0,
            common_action=0.0,
        )


def test_longitudinal_authority_localizes_opposing_reversal_memory() -> None:
    telemetry = LongitudinalAuthorityTelemetryAccumulator()
    telemetry.step(
        controller_updated=True,
        effective_reference_mps=-0.16,
        previous_effective_reference_mps=0.16,
        wheel_velocity_mps=-0.10,
        pitch_rad=0.02,
        pitch_rate_rad_s=-0.10,
        total_pitch_reference_rad=-0.04,
        total_pitch_limit_rad=0.104719755,
        common_action=-0.06,
        vx_integral_before=0.30,
        vx_integral_after=-0.0012,
        integral_reset=True,
        pitch_contribution=0.20,
        pitch_rate_contribution=-0.26,
        wheel_velocity_contribution=-0.0001,
    )
    telemetry.step(
        controller_updated=False,
        effective_reference_mps=-0.16,
        previous_effective_reference_mps=0.16,
        wheel_velocity_mps=-0.15,
        pitch_rad=0.01,
        pitch_rate_rad_s=-0.02,
        total_pitch_reference_rad=-0.03,
        total_pitch_limit_rad=0.104719755,
        common_action=-0.02,
        vx_integral_before=-0.0012,
        vx_integral_after=-0.0012,
        integral_reset=False,
        pitch_contribution=0.05,
        pitch_rate_contribution=-0.03,
        wheel_velocity_contribution=-0.0001,
    )

    summary = telemetry.summary()
    assert summary["schema"] == LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA
    assert summary["policy_rate_sample_count"] == 2
    assert summary["controller_update_count"] == 1
    assert summary["held_controller_command_step_count"] == 1
    assert summary["reference_sign_change_count"] == 1
    assert summary["opposing_integral_sign_change_count"] == 1
    assert summary["integral_reset_count"] == 1
    assert summary["velocity_deficit_step_count"] == 1
    assert summary["velocity_deficit_mean_mps"] == pytest.approx(0.06)
    assert summary["deficit_pitch_contribution_mean"] == pytest.approx(0.20)
    assert summary["deficit_pitch_rate_contribution_mean"] == pytest.approx(-0.26)


def test_longitudinal_authority_rejects_invalid_or_repeated_reset() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        LongitudinalAuthorityTelemetryAccumulator(reference_deadband_mps=0.0)
    telemetry = LongitudinalAuthorityTelemetryAccumulator()
    kwargs = dict(
        effective_reference_mps=-0.16,
        previous_effective_reference_mps=0.16,
        wheel_velocity_mps=-0.10,
        pitch_rad=0.0,
        pitch_rate_rad_s=0.0,
        total_pitch_reference_rad=0.0,
        total_pitch_limit_rad=0.104719755,
        common_action=0.0,
        vx_integral_before=0.3,
        vx_integral_after=0.0,
        integral_reset=True,
        pitch_contribution=0.0,
        pitch_rate_contribution=0.0,
        wheel_velocity_contribution=0.0,
    )
    with pytest.raises(ValueError, match="controller update"):
        telemetry.step(controller_updated=False, **kwargs)
    with pytest.raises(ValueError, match="finite"):
        telemetry.step(
            controller_updated=True,
            **{**kwargs, "pitch_rate_rad_s": float("nan")},
        )
