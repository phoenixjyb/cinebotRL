"""Policy-rate evidence for recovery-gated steering without changing commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


RECOVERY_TELEMETRY_SCHEMA = "riser_recovery_direction_policy_rate_v1"
VELOCITY_FEEDBACK_TELEMETRY_SCHEMA = (
    "riser_root_vs_wheel_velocity_policy_rate_v1"
)
LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA = (
    "riser_longitudinal_authority_policy_rate_v1"
)


def _sign(value: float, deadband: float) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


@dataclass
class RecoveryTelemetryAccumulator:
    """Aggregate every policy step while keeping the existing 1 Hz trace small."""

    direction_deadband: float = 1e-6
    sample_count: int = 0
    activation_step_count: int = 0
    full_authority_step_count: int = 0
    activation_segment_count: int = 0
    motion_direction_sign_change_count: int = 0
    feedback_direction_sign_change_count: int = 0
    consecutive_active_motion_direction_chatter_count: int = 0
    candidate_yaw_saturation_step_count: int = 0
    legacy_yaw_saturation_step_count: int = 0
    candidate_vs_legacy_delta_nonzero_step_count: int = 0
    candidate_vs_legacy_yaw_delta_abs_max_rad_s: float = 0.0
    recovery_blend_max: float = 0.0
    _active_last_step: bool = field(default=False, init=False, repr=False)
    _previous_motion_sign: int = field(default=0, init=False, repr=False)
    _previous_feedback_sign: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.direction_deadband) or self.direction_deadband <= 0.0:
            raise ValueError("recovery telemetry deadband must be finite and positive")

    def step(
        self,
        *,
        recovery_blend: float,
        motion_direction: float,
        feedback_motion_direction: float,
        candidate_yaw_rate_rad_s: float,
        legacy_yaw_rate_rad_s: float,
        maximum_yaw_rate_rad_s: float,
    ) -> None:
        values = (
            recovery_blend,
            motion_direction,
            feedback_motion_direction,
            candidate_yaw_rate_rad_s,
            legacy_yaw_rate_rad_s,
            maximum_yaw_rate_rad_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("recovery telemetry samples must be finite")
        if not 0.0 <= recovery_blend <= 1.0 or maximum_yaw_rate_rad_s <= 0.0:
            raise ValueError("invalid recovery telemetry sample")

        self.sample_count += 1
        self.recovery_blend_max = max(self.recovery_blend_max, recovery_blend)
        active = recovery_blend > 1e-12
        if not active:
            self._active_last_step = False
            self._previous_motion_sign = 0
            self._previous_feedback_sign = 0
            return

        self.activation_step_count += 1
        self.full_authority_step_count += int(recovery_blend >= 1.0 - 1e-12)
        if not self._active_last_step:
            self.activation_segment_count += 1

        motion_sign = _sign(motion_direction, self.direction_deadband)
        feedback_sign = _sign(feedback_motion_direction, self.direction_deadband)
        if self._active_last_step:
            motion_changed = (
                motion_sign != 0
                and self._previous_motion_sign != 0
                and motion_sign != self._previous_motion_sign
            )
            feedback_changed = (
                feedback_sign != 0
                and self._previous_feedback_sign != 0
                and feedback_sign != self._previous_feedback_sign
            )
            self.motion_direction_sign_change_count += int(motion_changed)
            self.feedback_direction_sign_change_count += int(feedback_changed)
            self.consecutive_active_motion_direction_chatter_count += int(
                motion_changed
            )

        yaw_limit = maximum_yaw_rate_rad_s - 1e-9
        self.candidate_yaw_saturation_step_count += int(
            abs(candidate_yaw_rate_rad_s) >= yaw_limit
        )
        self.legacy_yaw_saturation_step_count += int(
            abs(legacy_yaw_rate_rad_s) >= yaw_limit
        )
        delta = abs(candidate_yaw_rate_rad_s - legacy_yaw_rate_rad_s)
        self.candidate_vs_legacy_yaw_delta_abs_max_rad_s = max(
            self.candidate_vs_legacy_yaw_delta_abs_max_rad_s, delta
        )
        self.candidate_vs_legacy_delta_nonzero_step_count += int(delta > 1e-12)
        self._active_last_step = True
        self._previous_motion_sign = motion_sign
        self._previous_feedback_sign = feedback_sign

    def summary(self) -> dict[str, object]:
        return {
            "schema": RECOVERY_TELEMETRY_SCHEMA,
            "policy_rate_sample_count": self.sample_count,
            "activation_step_count": self.activation_step_count,
            "full_authority_step_count": self.full_authority_step_count,
            "activation_segment_count": self.activation_segment_count,
            "motion_direction_sign_change_count": (
                self.motion_direction_sign_change_count
            ),
            "feedback_direction_sign_change_count": (
                self.feedback_direction_sign_change_count
            ),
            "consecutive_active_motion_direction_chatter_count": (
                self.consecutive_active_motion_direction_chatter_count
            ),
            "candidate_yaw_saturation_step_count": (
                self.candidate_yaw_saturation_step_count
            ),
            "legacy_yaw_saturation_step_count": (
                self.legacy_yaw_saturation_step_count
            ),
            "candidate_vs_legacy_delta_nonzero_step_count": (
                self.candidate_vs_legacy_delta_nonzero_step_count
            ),
            "candidate_vs_legacy_yaw_delta_abs_max_rad_s": (
                self.candidate_vs_legacy_yaw_delta_abs_max_rad_s
            ),
            "recovery_blend_max": self.recovery_blend_max,
        }


@dataclass
class VelocityFeedbackTelemetryAccumulator:
    """Compare scored root motion with wheel-derived controller feedback."""

    sign_deadband_mps: float = 0.02
    wheel_tracking_tolerance_mps: float = 0.05
    root_lag_threshold_mps: float = 0.15
    sample_count: int = 0
    opposite_direction_step_count: int = 0
    wheel_false_tracking_step_count: int = 0
    root_wheel_mismatch_abs_max_mps: float = 0.0
    root_velocity_abs_max_mps: float = 0.0
    wheel_velocity_abs_max_mps: float = 0.0
    root_reference_error_abs_max_mps: float = 0.0
    wheel_reference_error_abs_max_mps: float = 0.0
    effective_reference_abs_max_mps: float = 0.0
    pitch_reference_abs_max_rad: float = 0.0
    total_pitch_reference_abs_max_rad: float = 0.0
    applied_pitch_bias_abs_max_rad: float = 0.0
    common_action_abs_max: float = 0.0
    _mismatch_squared_sum: float = field(default=0.0, init=False, repr=False)
    _root_error_squared_sum: float = field(default=0.0, init=False, repr=False)
    _wheel_error_squared_sum: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        thresholds = (
            self.sign_deadband_mps,
            self.wheel_tracking_tolerance_mps,
            self.root_lag_threshold_mps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in thresholds):
            raise ValueError("velocity telemetry thresholds must be finite and positive")

    def step(
        self,
        *,
        root_velocity_mps: float,
        wheel_velocity_mps: float,
        effective_reference_mps: float,
        pitch_reference_rad: float,
        total_pitch_reference_rad: float,
        applied_pitch_bias_rad: float,
        common_action: float,
    ) -> None:
        values = (
            root_velocity_mps,
            wheel_velocity_mps,
            effective_reference_mps,
            pitch_reference_rad,
            total_pitch_reference_rad,
            applied_pitch_bias_rad,
            common_action,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("velocity telemetry samples must be finite")
        mismatch = root_velocity_mps - wheel_velocity_mps
        root_error = effective_reference_mps - root_velocity_mps
        wheel_error = effective_reference_mps - wheel_velocity_mps
        self.sample_count += 1
        self._mismatch_squared_sum += mismatch * mismatch
        self._root_error_squared_sum += root_error * root_error
        self._wheel_error_squared_sum += wheel_error * wheel_error
        self.root_wheel_mismatch_abs_max_mps = max(
            self.root_wheel_mismatch_abs_max_mps, abs(mismatch)
        )
        self.root_velocity_abs_max_mps = max(
            self.root_velocity_abs_max_mps, abs(root_velocity_mps)
        )
        self.wheel_velocity_abs_max_mps = max(
            self.wheel_velocity_abs_max_mps, abs(wheel_velocity_mps)
        )
        self.root_reference_error_abs_max_mps = max(
            self.root_reference_error_abs_max_mps, abs(root_error)
        )
        self.wheel_reference_error_abs_max_mps = max(
            self.wheel_reference_error_abs_max_mps, abs(wheel_error)
        )
        self.effective_reference_abs_max_mps = max(
            self.effective_reference_abs_max_mps, abs(effective_reference_mps)
        )
        self.pitch_reference_abs_max_rad = max(
            self.pitch_reference_abs_max_rad, abs(pitch_reference_rad)
        )
        self.total_pitch_reference_abs_max_rad = max(
            self.total_pitch_reference_abs_max_rad,
            abs(total_pitch_reference_rad),
        )
        self.applied_pitch_bias_abs_max_rad = max(
            self.applied_pitch_bias_abs_max_rad, abs(applied_pitch_bias_rad)
        )
        self.common_action_abs_max = max(
            self.common_action_abs_max, abs(common_action)
        )
        root_sign = _sign(root_velocity_mps, self.sign_deadband_mps)
        wheel_sign = _sign(wheel_velocity_mps, self.sign_deadband_mps)
        self.opposite_direction_step_count += int(
            root_sign != 0 and wheel_sign != 0 and root_sign != wheel_sign
        )
        self.wheel_false_tracking_step_count += int(
            abs(wheel_error) <= self.wheel_tracking_tolerance_mps
            and abs(root_error) >= self.root_lag_threshold_mps
        )

    def summary(self) -> dict[str, object]:
        denominator = max(self.sample_count, 1)
        return {
            "schema": VELOCITY_FEEDBACK_TELEMETRY_SCHEMA,
            "policy_rate_sample_count": self.sample_count,
            "root_wheel_mismatch_rms_mps": math.sqrt(
                self._mismatch_squared_sum / denominator
            ),
            "root_wheel_mismatch_abs_max_mps": (
                self.root_wheel_mismatch_abs_max_mps
            ),
            "root_reference_error_rms_mps": math.sqrt(
                self._root_error_squared_sum / denominator
            ),
            "wheel_reference_error_rms_mps": math.sqrt(
                self._wheel_error_squared_sum / denominator
            ),
            "root_reference_error_abs_max_mps": (
                self.root_reference_error_abs_max_mps
            ),
            "wheel_reference_error_abs_max_mps": (
                self.wheel_reference_error_abs_max_mps
            ),
            "root_velocity_abs_max_mps": self.root_velocity_abs_max_mps,
            "wheel_velocity_abs_max_mps": self.wheel_velocity_abs_max_mps,
            "effective_reference_abs_max_mps": (
                self.effective_reference_abs_max_mps
            ),
            "pitch_reference_abs_max_rad": self.pitch_reference_abs_max_rad,
            "total_pitch_reference_abs_max_rad": (
                self.total_pitch_reference_abs_max_rad
            ),
            "applied_pitch_bias_abs_max_rad": (
                self.applied_pitch_bias_abs_max_rad
            ),
            "common_action_abs_max": self.common_action_abs_max,
            "opposite_direction_step_count": self.opposite_direction_step_count,
            "opposite_direction_ratio": (
                self.opposite_direction_step_count / denominator
            ),
            "wheel_false_tracking_step_count": (
                self.wheel_false_tracking_step_count
            ),
            "wheel_false_tracking_ratio": (
                self.wheel_false_tracking_step_count / denominator
            ),
            "sign_deadband_mps": self.sign_deadband_mps,
            "wheel_tracking_tolerance_mps": self.wheel_tracking_tolerance_mps,
            "root_lag_threshold_mps": self.root_lag_threshold_mps,
        }


@dataclass
class LongitudinalAuthorityTelemetryAccumulator:
    """Expose PI-memory and held inner-LQR authority at policy rate."""

    reference_deadband_mps: float = 0.05
    deficit_tolerance_mps: float = 0.03
    sample_count: int = 0
    controller_update_count: int = 0
    reference_sign_change_count: int = 0
    opposing_integral_sign_change_count: int = 0
    integral_reset_count: int = 0
    velocity_deficit_step_count: int = 0
    total_pitch_limit_step_count: int = 0
    velocity_deficit_abs_max_mps: float = 0.0
    vx_integral_before_abs_max: float = 0.0
    vx_integral_after_abs_max: float = 0.0
    pitch_abs_max_rad: float = 0.0
    pitch_rate_abs_max_rad_s: float = 0.0
    total_pitch_reference_abs_max_rad: float = 0.0
    common_action_abs_max: float = 0.0
    _velocity_deficit_sum_mps: float = field(
        default=0.0, init=False, repr=False
    )
    _deficit_pitch_contribution_sum: float = field(
        default=0.0, init=False, repr=False
    )
    _deficit_pitch_rate_contribution_sum: float = field(
        default=0.0, init=False, repr=False
    )
    _deficit_wheel_velocity_contribution_sum: float = field(
        default=0.0, init=False, repr=False
    )

    def __post_init__(self) -> None:
        values = (self.reference_deadband_mps, self.deficit_tolerance_mps)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "longitudinal authority thresholds must be finite and positive"
            )

    def step(
        self,
        *,
        controller_updated: bool,
        effective_reference_mps: float,
        previous_effective_reference_mps: float,
        wheel_velocity_mps: float,
        pitch_rad: float,
        pitch_rate_rad_s: float,
        total_pitch_reference_rad: float,
        total_pitch_limit_rad: float,
        common_action: float,
        vx_integral_before: float,
        vx_integral_after: float,
        integral_reset: bool,
        pitch_contribution: float,
        pitch_rate_contribution: float,
        wheel_velocity_contribution: float,
    ) -> None:
        values = (
            effective_reference_mps,
            previous_effective_reference_mps,
            wheel_velocity_mps,
            pitch_rad,
            pitch_rate_rad_s,
            total_pitch_reference_rad,
            total_pitch_limit_rad,
            common_action,
            vx_integral_before,
            vx_integral_after,
            pitch_contribution,
            pitch_rate_contribution,
            wheel_velocity_contribution,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("longitudinal authority samples must be finite")
        if total_pitch_limit_rad <= 0.0:
            raise ValueError("total pitch limit must be positive")
        if integral_reset and not controller_updated:
            raise ValueError("integral reset must coincide with a controller update")

        self.sample_count += 1
        self.controller_update_count += int(controller_updated)
        self.vx_integral_before_abs_max = max(
            self.vx_integral_before_abs_max, abs(vx_integral_before)
        )
        self.vx_integral_after_abs_max = max(
            self.vx_integral_after_abs_max, abs(vx_integral_after)
        )
        self.pitch_abs_max_rad = max(self.pitch_abs_max_rad, abs(pitch_rad))
        self.pitch_rate_abs_max_rad_s = max(
            self.pitch_rate_abs_max_rad_s, abs(pitch_rate_rad_s)
        )
        self.total_pitch_reference_abs_max_rad = max(
            self.total_pitch_reference_abs_max_rad,
            abs(total_pitch_reference_rad),
        )
        self.common_action_abs_max = max(
            self.common_action_abs_max, abs(common_action)
        )
        self.total_pitch_limit_step_count += int(
            abs(total_pitch_reference_rad) >= total_pitch_limit_rad - 1e-9
        )

        reference_sign = _sign(
            effective_reference_mps, self.reference_deadband_mps
        )
        previous_sign = _sign(
            previous_effective_reference_mps, self.reference_deadband_mps
        )
        sign_changed = bool(
            controller_updated
            and reference_sign != 0
            and previous_sign != 0
            and reference_sign != previous_sign
        )
        opposing_integral = bool(
            sign_changed and vx_integral_before * effective_reference_mps < 0.0
        )
        self.reference_sign_change_count += int(sign_changed)
        self.opposing_integral_sign_change_count += int(opposing_integral)
        self.integral_reset_count += int(integral_reset)

        projected_deficit = (
            reference_sign
            * (effective_reference_mps - wheel_velocity_mps)
            if reference_sign != 0
            else 0.0
        )
        velocity_deficit = max(projected_deficit, 0.0)
        if velocity_deficit > self.deficit_tolerance_mps:
            self.velocity_deficit_step_count += 1
            self._velocity_deficit_sum_mps += velocity_deficit
            self._deficit_pitch_contribution_sum += pitch_contribution
            self._deficit_pitch_rate_contribution_sum += (
                pitch_rate_contribution
            )
            self._deficit_wheel_velocity_contribution_sum += (
                wheel_velocity_contribution
            )
            self.velocity_deficit_abs_max_mps = max(
                self.velocity_deficit_abs_max_mps, velocity_deficit
            )

    def summary(self) -> dict[str, object]:
        sample_denominator = max(self.sample_count, 1)
        deficit_denominator = max(self.velocity_deficit_step_count, 1)
        return {
            "schema": LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA,
            "policy_rate_sample_count": self.sample_count,
            "controller_update_count": self.controller_update_count,
            "held_controller_command_step_count": (
                self.sample_count - self.controller_update_count
            ),
            "reference_sign_change_count": self.reference_sign_change_count,
            "opposing_integral_sign_change_count": (
                self.opposing_integral_sign_change_count
            ),
            "integral_reset_count": self.integral_reset_count,
            "velocity_deficit_step_count": self.velocity_deficit_step_count,
            "velocity_deficit_ratio": (
                self.velocity_deficit_step_count / sample_denominator
            ),
            "velocity_deficit_mean_mps": (
                self._velocity_deficit_sum_mps / deficit_denominator
            ),
            "velocity_deficit_abs_max_mps": (
                self.velocity_deficit_abs_max_mps
            ),
            "deficit_pitch_contribution_mean": (
                self._deficit_pitch_contribution_sum / deficit_denominator
            ),
            "deficit_pitch_rate_contribution_mean": (
                self._deficit_pitch_rate_contribution_sum
                / deficit_denominator
            ),
            "deficit_wheel_velocity_contribution_mean": (
                self._deficit_wheel_velocity_contribution_sum
                / deficit_denominator
            ),
            "vx_integral_before_abs_max": self.vx_integral_before_abs_max,
            "vx_integral_after_abs_max": self.vx_integral_after_abs_max,
            "pitch_abs_max_rad": self.pitch_abs_max_rad,
            "pitch_rate_abs_max_rad_s": self.pitch_rate_abs_max_rad_s,
            "total_pitch_reference_abs_max_rad": (
                self.total_pitch_reference_abs_max_rad
            ),
            "total_pitch_limit_step_count": (
                self.total_pitch_limit_step_count
            ),
            "common_action_abs_max": self.common_action_abs_max,
            "reference_deadband_mps": self.reference_deadband_mps,
            "deficit_tolerance_mps": self.deficit_tolerance_mps,
        }
