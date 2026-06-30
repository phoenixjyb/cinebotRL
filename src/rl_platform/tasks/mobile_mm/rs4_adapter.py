"""Pure utilities for the proposed RS4 attitude-rate action contract.

This module does not change the active Isaac environment.  It defines the
rate-command mapping needed by ``rs4_attitude_rate_v1`` so dataset builders and
future sim adapters can share one tested convention.

Conventions:
* Policy attitude channels are ordered ``[yaw, pitch, roll]``.
* Local gimbal vectors are ordered ``[roll, pitch, yaw]``.
* The default RS4 command vector applies the documented deployment mapping
  ``rs4_axis_map_from_gimbal=(2, 0, 1)``, producing ``[yaw, roll, pitch]`` from
  local ``[roll, pitch, yaw]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


POLICY_ATTITUDE_RATE_ORDER = ("yaw", "pitch", "roll")
LOCAL_GIMBAL_RATE_ORDER = ("roll", "pitch", "yaw")
DEFAULT_RS4_AXIS_MAP_FROM_GIMBAL = (2, 0, 1)


@dataclass(frozen=True)
class Rs4RateAdapterConfig:
    """Scaling and mapping for RS4 attitude-rate commands."""

    max_yaw_rate_deg_s: float = 90.0
    max_pitch_rate_deg_s: float = 90.0
    max_roll_rate_deg_s: float = 45.0
    max_yaw_accel_deg_s2: float = 360.0
    max_pitch_accel_deg_s2: float = 360.0
    max_roll_accel_deg_s2: float = 180.0
    enable_roll: bool = False
    rs4_axis_map_from_gimbal: tuple[int, int, int] = DEFAULT_RS4_AXIS_MAP_FROM_GIMBAL

    def __post_init__(self) -> None:
        if sorted(self.rs4_axis_map_from_gimbal) != [0, 1, 2]:
            raise ValueError(
                "rs4_axis_map_from_gimbal must be a permutation of local "
                "[roll, pitch, yaw] indices"
            )

    @property
    def max_policy_order_rates(self) -> np.ndarray:
        """Max rates for policy order ``[yaw, pitch, roll]``."""

        return np.asarray(
            [self.max_yaw_rate_deg_s, self.max_pitch_rate_deg_s, self.max_roll_rate_deg_s],
            dtype=np.float32,
        )

    @property
    def max_policy_order_accels(self) -> np.ndarray:
        """Max accelerations for policy order ``[yaw, pitch, roll]``."""

        return np.asarray(
            [self.max_yaw_accel_deg_s2, self.max_pitch_accel_deg_s2, self.max_roll_accel_deg_s2],
            dtype=np.float32,
        )


def _as_rate_array(values: np.ndarray | list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3, got shape {arr.shape}")
    return arr


def normalized_policy_rates_to_deg_s(
    normalized_rates: np.ndarray | list[float] | tuple[float, ...],
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> np.ndarray:
    """Scale normalized policy ``[yaw, pitch, roll]`` rates into deg/s.

    Roll defaults to zero because the currently verified deployment path is a
    yaw/pitch velocity loop.  Set ``enable_roll=True`` only after RS4 mixed-mode
    roll behavior is validated.
    """

    rates = np.clip(_as_rate_array(normalized_rates, name="normalized_rates"), -1.0, 1.0)
    scaled = rates * config.max_policy_order_rates
    if not config.enable_roll:
        scaled = scaled.copy()
        scaled[..., 2] = 0.0
    return scaled.astype(np.float32)


def policy_order_to_local_gimbal_order(policy_rates_deg_s: np.ndarray) -> np.ndarray:
    """Convert ``[yaw, pitch, roll]`` to local gimbal ``[roll, pitch, yaw]``."""

    rates = _as_rate_array(policy_rates_deg_s, name="policy_rates_deg_s")
    return rates[..., [2, 1, 0]].astype(np.float32)


def local_gimbal_order_to_policy_order(local_rates_deg_s: np.ndarray) -> np.ndarray:
    """Convert local gimbal ``[roll, pitch, yaw]`` to policy ``[yaw, pitch, roll]``."""

    rates = _as_rate_array(local_rates_deg_s, name="local_rates_deg_s")
    return rates[..., [2, 1, 0]].astype(np.float32)


def local_gimbal_to_rs4_axis_order(
    local_rates_deg_s: np.ndarray | list[float] | tuple[float, ...],
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> np.ndarray:
    """Map local gimbal ``[roll, pitch, yaw]`` into RS4 command axis order."""

    rates = _as_rate_array(local_rates_deg_s, name="local_rates_deg_s")
    return rates[..., list(config.rs4_axis_map_from_gimbal)].astype(np.float32)


def policy_rates_to_rs4_command_deg_s(
    normalized_rates: np.ndarray | list[float] | tuple[float, ...],
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> np.ndarray:
    """Full mapping from policy ``[yaw, pitch, roll]`` to RS4 command deg/s."""

    policy_rates = normalized_policy_rates_to_deg_s(normalized_rates, config)
    local_rates = policy_order_to_local_gimbal_order(policy_rates)
    return local_gimbal_to_rs4_axis_order(local_rates, config)


def clamp_policy_rate_delta(
    desired_policy_rates_deg_s: np.ndarray | list[float] | tuple[float, ...],
    previous_policy_rates_deg_s: np.ndarray | list[float] | tuple[float, ...],
    dt_s: float,
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> np.ndarray:
    """Apply per-axis acceleration limits in policy order ``[yaw, pitch, roll]``."""

    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    desired = _as_rate_array(desired_policy_rates_deg_s, name="desired_policy_rates_deg_s")
    previous = _as_rate_array(previous_policy_rates_deg_s, name="previous_policy_rates_deg_s")
    max_delta = config.max_policy_order_accels * float(dt_s)
    delta = np.clip(desired - previous, -max_delta, max_delta)
    limited = previous + delta
    if not config.enable_roll:
        limited = limited.copy()
        limited[..., 2] = 0.0
    return limited.astype(np.float32)


def integrate_policy_attitude_deg(
    current_attitude_deg: np.ndarray | list[float] | tuple[float, ...],
    policy_rates_deg_s: np.ndarray | list[float] | tuple[float, ...],
    dt_s: float,
) -> np.ndarray:
    """Integrate policy-order attitude ``[yaw, pitch, roll]`` by deg/s rates.

    This is a simple sim/dataset helper, not a replacement for the real RS4
    controller.  Yaw is wrapped to ``[-180, 180)`` while pitch/roll are left
    unclamped because final limits need hardware confirmation.
    """

    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    attitude = _as_rate_array(current_attitude_deg, name="current_attitude_deg")
    rates = _as_rate_array(policy_rates_deg_s, name="policy_rates_deg_s")
    out = attitude + rates * float(dt_s)
    out = out.astype(np.float32)
    out[..., 0] = (out[..., 0] + 180.0) % 360.0 - 180.0
    return out

