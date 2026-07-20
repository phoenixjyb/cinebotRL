import pytest

from rl_platform.tasks.two_wheel_balance.riser_recovery_evidence import (
    RECOVERY_TELEMETRY_SCHEMA,
    VELOCITY_FEEDBACK_TELEMETRY_SCHEMA,
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
