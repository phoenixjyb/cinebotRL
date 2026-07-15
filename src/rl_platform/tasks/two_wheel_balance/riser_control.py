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
