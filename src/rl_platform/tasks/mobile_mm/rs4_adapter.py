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
import torch


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


def _as_quaternion_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[-1] != 4:
        raise ValueError(f"{name} must have last dimension 4 in wxyz order, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if not np.isfinite(arr).all() or np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains non-finite or zero-length quaternion values")
    return arr / norms


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _relative_quaternion_policy_rotvec_deg(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = _quat_multiply_wxyz(_quat_conjugate_wxyz(current), target)
    relative *= np.where(relative[..., :1] < 0.0, -1.0, 1.0)
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., :1], -1.0, 1.0))
    axis = vector / np.maximum(vector_norm, 1e-12)
    rotvec_local_deg = np.rad2deg(axis * angle)
    return rotvec_local_deg[..., [2, 1, 0]]


def quaternion_residual_policy_rates_deg_s(
    current_quat_wxyz: np.ndarray,
    target_quat_wxyz: np.ndarray,
    *,
    response_horizon_s: float,
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-attitude error into bounded local yaw/pitch/roll rates.

    The relative rotation is computed in the current camera frame as
    ``q_current^-1 * q_target``.  Its rotation vector is singularity-free and
    ordered into policy channels as local ``[yaw, pitch, roll] = [z, y, x]``.
    The returned tuple is ``(bounded_rates_deg_s, residual_rotvec_deg)``.
    """

    if response_horizon_s <= 0.0:
        raise ValueError(f"response_horizon_s must be positive, got {response_horizon_s}")
    current = _as_quaternion_array(current_quat_wxyz, name="current_quat_wxyz")
    target = _as_quaternion_array(target_quat_wxyz, name="target_quat_wxyz")
    if current.shape != target.shape:
        raise ValueError(f"current and target quaternion shapes differ: {current.shape} vs {target.shape}")

    policy_residual_deg = _relative_quaternion_policy_rotvec_deg(current, target)
    desired_rates = policy_residual_deg / float(response_horizon_s)
    bounded = np.clip(
        desired_rates,
        -config.max_policy_order_rates,
        config.max_policy_order_rates,
    )
    if not config.enable_roll:
        bounded = bounded.copy()
        bounded[..., 2] = 0.0
    return bounded.astype(np.float32), policy_residual_deg.astype(np.float32)


def quaternion_residual_policy_rates_rad_s_torch(
    current_quat_wxyz: torch.Tensor,
    target_quat_wxyz: torch.Tensor,
    *,
    response_horizon_s: float,
    max_policy_rates_rad_s: torch.Tensor,
    enable_roll: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded local camera-error rates for the live split adapter."""

    if response_horizon_s <= 0.0:
        raise ValueError(f"response_horizon_s must be positive, got {response_horizon_s}")
    if current_quat_wxyz.shape != target_quat_wxyz.shape or current_quat_wxyz.shape[-1] != 4:
        raise ValueError(
            "current and target quaternions must have the same [...,4] shape, got "
            f"{tuple(current_quat_wxyz.shape)} and {tuple(target_quat_wxyz.shape)}"
        )
    if max_policy_rates_rad_s.shape != (3,):
        raise ValueError(f"max_policy_rates_rad_s must have shape (3,), got {tuple(max_policy_rates_rad_s.shape)}")

    current = torch.nn.functional.normalize(current_quat_wxyz, dim=-1)
    target = torch.nn.functional.normalize(target_quat_wxyz, dim=-1)
    current_conjugate = current.clone()
    current_conjugate[..., 1:] *= -1.0

    w1, x1, y1, z1 = current_conjugate.unbind(dim=-1)
    w2, x2, y2, z2 = target.unbind(dim=-1)
    relative = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    relative = relative * torch.where(relative[..., :1] < 0.0, -1.0, 1.0)
    vector = relative[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(relative[..., :1], -1.0, 1.0))
    axis = vector / torch.clamp(vector_norm, min=1e-12)
    policy_residual = (axis * angle)[..., [2, 1, 0]]
    desired_rates = policy_residual / float(response_horizon_s)
    desired_rates = torch.clamp(desired_rates, -max_policy_rates_rad_s, max_policy_rates_rad_s)
    if not enable_roll:
        desired_rates = desired_rates.clone()
        desired_rates[..., 2] = 0.0
    return desired_rates, policy_residual


def quaternion_world_error_rotvec_rad_torch(
    current_quat_wxyz: torch.Tensor,
    target_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return the shortest current-to-target rotation vector in world axes."""

    if current_quat_wxyz.shape != target_quat_wxyz.shape or current_quat_wxyz.shape[-1] != 4:
        raise ValueError(
            "current and target quaternions must have the same [...,4] shape, got "
            f"{tuple(current_quat_wxyz.shape)} and {tuple(target_quat_wxyz.shape)}"
        )
    current = torch.nn.functional.normalize(current_quat_wxyz, dim=-1)
    target = torch.nn.functional.normalize(target_quat_wxyz, dim=-1)
    current_conjugate = current.clone()
    current_conjugate[..., 1:] *= -1.0
    w1, x1, y1, z1 = target.unbind(dim=-1)
    w2, x2, y2, z2 = current_conjugate.unbind(dim=-1)
    relative = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    relative = relative * torch.where(relative[..., :1] < 0.0, -1.0, 1.0)
    vector = relative[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(relative[..., :1], -1.0, 1.0))
    return vector / torch.clamp(vector_norm, min=1e-12) * angle


def quaternion_feedforward_policy_rates_deg_s(
    target_quat_wxyz: np.ndarray,
    dt_s: np.ndarray | float,
) -> np.ndarray:
    """Compute singularity-free local target angular velocity in policy order."""

    target = _as_quaternion_array(target_quat_wxyz, name="target_quat_wxyz")
    if target.ndim != 2:
        raise ValueError(f"target_quat_wxyz must be [N,4], got shape {target.shape}")
    if target.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float32)
    dt = np.asarray(dt_s, dtype=np.float32)
    if dt.ndim == 0:
        dt = np.full(target.shape[0], float(dt), dtype=np.float32)
    if dt.shape != (target.shape[0],):
        raise ValueError(f"dt_s must be scalar or shape {(target.shape[0],)}, got {dt.shape}")
    if not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise ValueError("dt_s must contain finite positive values")

    rates = np.zeros((target.shape[0], 3), dtype=np.float32)
    rotvec = _relative_quaternion_policy_rotvec_deg(target[:-1], target[1:])
    rates[:-1] = rotvec / dt[:-1, None]
    rates[-1] = rates[-2]
    return rates


def quaternion_tracking_policy_rates_deg_s(
    current_quat_wxyz: np.ndarray,
    target_quat_wxyz: np.ndarray,
    *,
    dt_s: np.ndarray | float,
    response_horizon_s: float,
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build bounded feed-forward plus attitude-residual teacher rates.

    Returns ``(desired_rates, feedforward_rates, residual_rotvec_deg)`` before
    acceleration slew limiting.  All rate arrays use policy order
    ``[yaw, pitch, roll]``.
    """

    feedback, residual = quaternion_residual_policy_rates_deg_s(
        current_quat_wxyz,
        target_quat_wxyz,
        response_horizon_s=response_horizon_s,
        config=config,
    )
    feedforward = quaternion_feedforward_policy_rates_deg_s(target_quat_wxyz, dt_s)
    desired = np.clip(
        feedforward + feedback,
        -config.max_policy_order_rates,
        config.max_policy_order_rates,
    )
    if not config.enable_roll:
        desired[..., 2] = 0.0
        feedforward[..., 2] = 0.0
    return desired.astype(np.float32), feedforward.astype(np.float32), residual.astype(np.float32)


def slew_limit_policy_rate_sequence_deg_s(
    desired_rates_deg_s: np.ndarray,
    dt_s: np.ndarray | float,
    config: Rs4RateAdapterConfig = Rs4RateAdapterConfig(),
) -> np.ndarray:
    """Apply the runtime acceleration limits to a complete teacher sequence."""

    desired = _as_rate_array(desired_rates_deg_s, name="desired_rates_deg_s")
    if desired.ndim != 2:
        raise ValueError(f"desired_rates_deg_s must be [N,3], got shape {desired.shape}")
    dt = np.asarray(dt_s, dtype=np.float32)
    if dt.ndim == 0:
        dt = np.full(desired.shape[0], float(dt), dtype=np.float32)
    if dt.shape != (desired.shape[0],):
        raise ValueError(f"dt_s must be scalar or shape {(desired.shape[0],)}, got {dt.shape}")
    if not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise ValueError("dt_s must contain finite positive values")

    output = np.zeros_like(desired, dtype=np.float32)
    previous = np.zeros(3, dtype=np.float32)
    for index in range(desired.shape[0]):
        previous = clamp_policy_rate_delta(desired[index], previous, float(dt[index]), config)
        output[index] = previous
    return output


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
