"""Pure contract helpers and deterministic metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_NAMES = ("a_common", "a_yaw")
OBSERVATION_NAMES = (
    "pitch",
    "pitch_rate",
    "mean_wheel_position",
    "mean_wheel_velocity",
    "wheel_velocity_difference",
    "yaw_rate",
    "vx_ref",
    "wz_ref",
    "previous_a_common",
    "previous_a_yaw",
)


def mix_common_yaw_effort(actions: np.ndarray, torque_limit: float) -> np.ndarray:
    """Map normalized common/yaw actions to left/right wheel effort."""
    clipped = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
    if clipped.shape[-1] != 2:
        raise ValueError(f"expected action shape (..., 2), got {clipped.shape}")
    common = clipped[..., 0]
    yaw = clipped[..., 1]
    # With both positive wheel velocities driving +X and left wheel at +Y,
    # positive body +Z yaw requires left-forward/right-backward effort.
    wheel = np.stack((common + yaw, common - yaw), axis=-1)
    return np.clip(wheel, -1.0, 1.0) * float(torque_limit)


@dataclass(frozen=True)
class BalanceContract:
    physics_dt: float = 0.001
    decimation: int = 5
    torque_limit_nm: float = 20.0

    @property
    def policy_hz(self) -> float:
        return 1.0 / (self.physics_dt * self.decimation)

    @property
    def action_dim(self) -> int:
        return len(ACTION_NAMES)

    @property
    def observation_dim(self) -> int:
        return len(OBSERVATION_NAMES)
