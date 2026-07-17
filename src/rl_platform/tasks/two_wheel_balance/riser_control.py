"""Deterministic safety primitives for the motorized camera riser."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiserLimits:
    lower_m: float = 0.0
    upper_m: float = 1.2
    maximum_velocity_mps: float = 1.0
    maximum_acceleration_mps2: float = 2.0
    maximum_jerk_mps3: float = 8.0

    def __post_init__(self) -> None:
        values = (
            self.lower_m,
            self.upper_m,
            self.maximum_velocity_mps,
            self.maximum_acceleration_mps2,
            self.maximum_jerk_mps3,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("riser limits must be finite")
        if self.upper_m <= self.lower_m:
            raise ValueError("riser upper limit must exceed lower limit")
        if min(values[2:]) <= 0.0:
            raise ValueError("riser dynamic limits must be positive")


@dataclass(frozen=True)
class RiserReferenceState:
    position_m: float
    velocity_mps: float
    acceleration_mps2: float
    jerk_mps3: float


RISER_THERMAL_FORCE_CONTRACT = "leadshine_400w_first_order_monitor_v1"


@dataclass
class RiserMotorThermalMonitor:
    """Estimate normalized winding load from measured linear actuator force."""

    continuous_force_n: float = 292.3970042486123
    peak_force_n: float = 877.1910127458367
    thermal_time_constant_s: float = 30.0
    thermal_load: float = 0.0
    maximum_thermal_load: float = 0.0
    maximum_abs_force_n: float = 0.0
    peak_force_violation_count: int = 0
    sample_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.continuous_force_n,
            self.peak_force_n,
            self.thermal_time_constant_s,
            self.thermal_load,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("thermal monitor parameters must be finite")
        if (
            self.continuous_force_n <= 0.0
            or self.peak_force_n <= self.continuous_force_n
            or self.thermal_time_constant_s <= 0.0
            or self.thermal_load < 0.0
        ):
            raise ValueError("invalid thermal monitor parameters")
        self.maximum_thermal_load = max(
            self.maximum_thermal_load, self.thermal_load
        )

    def step(self, applied_force_n: float, dt_s: float) -> float:
        """Advance the first-order I-squared thermal state by one sample."""

        force = abs(float(applied_force_n))
        dt = float(dt_s)
        if not math.isfinite(force) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("thermal monitor sample must be finite with positive dt")
        alpha = -math.expm1(-dt / self.thermal_time_constant_s)
        normalized_heating = (force / self.continuous_force_n) ** 2
        self.thermal_load += alpha * (normalized_heating - self.thermal_load)
        self.maximum_thermal_load = max(
            self.maximum_thermal_load, self.thermal_load
        )
        self.maximum_abs_force_n = max(self.maximum_abs_force_n, force)
        self.peak_force_violation_count += int(force > self.peak_force_n + 1e-9)
        self.sample_count += 1
        return self.thermal_load

    @property
    def passed(self) -> bool:
        return (
            self.sample_count > 0
            and self.maximum_thermal_load <= 1.0 + 1e-9
            and self.peak_force_violation_count == 0
        )


def balance_progress_scale(
    absolute_pitch_rad: float,
    *,
    slow_start_rad: float = math.radians(3.0),
    stop_rad: float = math.radians(8.0),
    minimum_scale: float = 0.0,
) -> float:
    """Reduce riser progress before pitch reaches the balance stop boundary."""

    pitch = abs(float(absolute_pitch_rad))
    if not all(
        math.isfinite(value)
        for value in (pitch, slow_start_rad, stop_rad, minimum_scale)
    ):
        raise ValueError("progress governor inputs must be finite")
    if not 0.0 <= minimum_scale <= 1.0 or not 0.0 <= slow_start_rad < stop_rad:
        raise ValueError("invalid progress governor bounds")
    if pitch <= slow_start_rad:
        return 1.0
    if pitch >= stop_rad:
        return minimum_scale
    blend = (pitch - slow_start_rad) / (stop_rad - slow_start_rad)
    return 1.0 + blend * (minimum_scale - 1.0)


def required_stopping_distance(
    velocity_mps: float,
    maximum_deceleration_mps2: float,
    response_delay_s: float = 0.0,
) -> float:
    """Return delay travel plus constant-deceleration braking distance."""

    values = (velocity_mps, maximum_deceleration_mps2, response_delay_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("stopping-envelope inputs must be finite")
    if maximum_deceleration_mps2 <= 0.0 or response_delay_s < 0.0:
        raise ValueError("deceleration must be positive and delay non-negative")
    speed = abs(float(velocity_mps))
    return speed * response_delay_s + speed**2 / (
        2.0 * maximum_deceleration_mps2
    )


def safe_velocity_for_stopping_distance(
    stopping_distance_m: float,
    maximum_deceleration_mps2: float,
    response_delay_s: float = 0.0,
) -> float:
    """Invert the stopping-distance model for a non-negative speed limit."""

    values = (stopping_distance_m, maximum_deceleration_mps2, response_delay_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("stopping-envelope inputs must be finite")
    if stopping_distance_m < 0.0:
        raise ValueError("stopping distance must be non-negative")
    if maximum_deceleration_mps2 <= 0.0 or response_delay_s < 0.0:
        raise ValueError("deceleration must be positive and delay non-negative")
    delayed_speed = maximum_deceleration_mps2 * response_delay_s
    return max(
        0.0,
        math.sqrt(
            delayed_speed**2
            + 2.0 * maximum_deceleration_mps2 * stopping_distance_m
        )
        - delayed_speed,
    )


def safe_riser_velocity_bounds(
    position_m: float,
    *,
    hard_lower_m: float,
    hard_upper_m: float,
    maximum_velocity_mps: float,
    maximum_deceleration_mps2: float,
    response_delay_s: float = 0.0,
    hard_margin_m: float = 0.0,
) -> tuple[float, float]:
    """Return direction-aware velocity bounds that stop before hard margins."""

    values = (
        position_m,
        hard_lower_m,
        hard_upper_m,
        maximum_velocity_mps,
        maximum_deceleration_mps2,
        response_delay_s,
        hard_margin_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("riser velocity-envelope inputs must be finite")
    if hard_upper_m <= hard_lower_m:
        raise ValueError("hard upper limit must exceed hard lower limit")
    if not hard_lower_m <= position_m <= hard_upper_m:
        raise ValueError("riser position lies outside hard limits")
    if maximum_velocity_mps <= 0.0 or maximum_deceleration_mps2 <= 0.0:
        raise ValueError("riser velocity and deceleration limits must be positive")
    if response_delay_s < 0.0 or hard_margin_m < 0.0:
        raise ValueError("delay and hard margin must be non-negative")
    if 2.0 * hard_margin_m >= hard_upper_m - hard_lower_m:
        raise ValueError("hard margins consume the complete riser travel")

    lower_distance = max(0.0, position_m - hard_lower_m - hard_margin_m)
    upper_distance = max(0.0, hard_upper_m - hard_margin_m - position_m)
    downward_speed = min(
        maximum_velocity_mps,
        safe_velocity_for_stopping_distance(
            lower_distance,
            maximum_deceleration_mps2,
            response_delay_s,
        ),
    )
    upward_speed = min(
        maximum_velocity_mps,
        safe_velocity_for_stopping_distance(
            upper_distance,
            maximum_deceleration_mps2,
            response_delay_s,
        ),
    )
    return -downward_speed, upward_speed


@dataclass(frozen=True)
class QuinticRiserMove:
    """Zero-velocity/acceleration endpoint move with analytic dynamic bounds."""

    start_m: float
    target_m: float
    duration_s: float
    limits: RiserLimits = RiserLimits()

    # Maxima of derivatives of 10u^3 - 15u^4 + 6u^5 on u in [0, 1].
    _VELOCITY_FACTOR = 1.875
    _ACCELERATION_FACTOR = 5.773502691896258
    _JERK_FACTOR = 60.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.start_m, self.target_m, self.duration_s)
        ):
            raise ValueError("riser move values must be finite")
        if self.duration_s <= 0.0:
            raise ValueError("riser move duration must be positive")
        if not (
            self.limits.lower_m <= self.start_m <= self.limits.upper_m
            and self.limits.lower_m <= self.target_m <= self.limits.upper_m
        ):
            raise ValueError("riser move endpoints exceed travel limits")

    @classmethod
    def for_peak_velocity(
        cls,
        start_m: float,
        target_m: float,
        peak_velocity_mps: float,
        limits: RiserLimits = RiserLimits(),
    ) -> "QuinticRiserMove":
        distance = abs(float(target_m) - float(start_m))
        if not math.isfinite(peak_velocity_mps) or not (
            0.0 < peak_velocity_mps <= limits.maximum_velocity_mps
        ):
            raise ValueError("requested peak velocity exceeds riser contract")
        if distance <= 0.0:
            raise ValueError("riser move endpoints must differ")
        duration = max(
            cls._VELOCITY_FACTOR * distance / peak_velocity_mps,
            math.sqrt(
                cls._ACCELERATION_FACTOR
                * distance
                / limits.maximum_acceleration_mps2
            ),
            (
                cls._JERK_FACTOR * distance / limits.maximum_jerk_mps3
            )
            ** (1.0 / 3.0),
        )
        return cls(float(start_m), float(target_m), duration, limits)

    def sample(self, elapsed_s: float) -> RiserReferenceState:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        u = min(max(float(elapsed_s) / self.duration_s, 0.0), 1.0)
        distance = self.target_m - self.start_m
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_d1 = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        blend_d2 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
        blend_d3 = 60.0 - 360.0 * u + 360.0 * u**2
        return RiserReferenceState(
            self.start_m + distance * blend,
            distance * blend_d1 / self.duration_s,
            distance * blend_d2 / self.duration_s**2,
            distance * blend_d3 / self.duration_s**3,
        )
