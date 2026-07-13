"""Pure contract helpers and deterministic metric aggregation."""

from __future__ import annotations

import math
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

LQR_STATE_NAMES = OBSERVATION_NAMES[:6]


def mix_common_yaw_effort(actions: np.ndarray, torque_limit: float) -> np.ndarray:
    """Map normalized common/yaw actions to left/right wheel effort."""
    clipped = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
    if clipped.shape[-1] != 2:
        raise ValueError(f"expected action shape (..., 2), got {clipped.shape}")
    common = clipped[..., 0]
    yaw = clipped[..., 1]
    # With both +Y wheel axes, positive velocity drives +X. Positive body +Z
    # yaw therefore requires the right wheel forward and left wheel backward.
    wheel = np.stack((common - yaw, common + yaw), axis=-1)
    return np.clip(wheel, -1.0, 1.0) * float(torque_limit)


def compose_pd_residual_action(
    pitch: np.ndarray,
    pitch_rate: np.ndarray,
    residual_actions: np.ndarray,
    *,
    kp: float = 1.0,
    kd: float = 0.2,
    pd_limit: float = 0.5,
    residual_scale: float = 0.15,
) -> np.ndarray:
    """Compose a bounded PD common action with learned common/yaw residuals."""
    residual = np.clip(np.asarray(residual_actions, dtype=np.float64), -1.0, 1.0)
    if residual.shape[-1] != 2:
        raise ValueError(f"expected residual shape (..., 2), got {residual.shape}")
    pd_common = np.clip(
        kp * np.asarray(pitch) + kd * np.asarray(pitch_rate),
        -pd_limit,
        pd_limit,
    )
    applied = residual_scale * residual
    applied[..., 0] += pd_common
    return np.clip(applied, -1.0, 1.0)


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


@dataclass(frozen=True)
class CascadedLQRConfig:
    """Deployable outer-loop contract around the frozen balance LQR."""

    wheel_radius_m: float = 0.1016
    wheel_track_m: float = 0.620
    vx_kp: float = 0.6
    vx_ki: float = 0.0
    wz_kp: float = 0.25
    wz_ki: float = 0.0
    wz_feedforward: float = 0.6
    wheel_difference_kp: float = 0.0
    pitch_reference_limit_rad: float = math.radians(6.0)
    vx_integral_limit: float = 0.5
    wz_integral_limit: float = 1.0
    action_limit: float = 0.8


def cascaded_lqr_action(
    states: np.ndarray,
    vx_ref: np.ndarray,
    wz_ref: np.ndarray,
    gain: np.ndarray,
    integrals: np.ndarray,
    *,
    control_dt: float,
    config: CascadedLQRConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Track chassis commands with bounded outer loops and a frozen inner LQR."""
    states = np.asarray(states, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    vx_ref = np.asarray(vx_ref, dtype=np.float64).reshape(-1)
    wz_ref = np.asarray(wz_ref, dtype=np.float64).reshape(-1)
    integrals = np.asarray(integrals, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(LQR_STATE_NAMES):
        raise ValueError(f"expected states shape (N, {len(LQR_STATE_NAMES)}), got {states.shape}")
    batch = states.shape[0]
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")
    if vx_ref.shape != (batch,) or wz_ref.shape != (batch,):
        raise ValueError("command arrays must match the state batch")
    if integrals.shape != (batch, 2):
        raise ValueError(f"expected integral shape {(batch, 2)}, got {integrals.shape}")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if config.wheel_radius_m <= 0.0 or config.wheel_track_m <= 0.0:
        raise ValueError("wheel geometry must be positive")

    vx_estimate = config.wheel_radius_m * states[:, 3]
    vx_error = vx_ref - vx_estimate
    wz_error = wz_ref - states[:, 5]
    next_integrals = integrals.copy()
    next_integrals[:, 0] = np.clip(
        next_integrals[:, 0] + control_dt * vx_error,
        -config.vx_integral_limit,
        config.vx_integral_limit,
    )
    next_integrals[:, 1] = np.clip(
        next_integrals[:, 1] + control_dt * wz_error,
        -config.wz_integral_limit,
        config.wz_integral_limit,
    )

    # A forward command first pulls the wheels back to create a forward lean;
    # the frozen balance LQR then drives the wheels under the falling body.
    pitch_reference = np.clip(
        config.vx_kp * vx_error + config.vx_ki * next_integrals[:, 0],
        -config.pitch_reference_limit_rad,
        config.pitch_reference_limit_rad,
    )
    tracking_states = states.copy()
    tracking_states[:, 0] -= pitch_reference
    tracking_states[:, 3] -= vx_ref / config.wheel_radius_m
    tracking_states[:, 4] -= (
        config.wheel_track_m / config.wheel_radius_m
    ) * wz_ref
    actions = lqr_action(
        tracking_states,
        gain,
        action_limit=config.action_limit,
    )
    wheel_difference_target = (
        config.wheel_track_m / config.wheel_radius_m
    ) * wz_ref
    wheel_difference_error = wheel_difference_target - states[:, 4]
    actions[:, 1] += (
        config.wz_kp * wz_error
        + config.wz_ki * next_integrals[:, 1]
        + config.wz_feedforward * wz_ref
        + config.wheel_difference_kp * wheel_difference_error
    )
    actions = np.clip(actions, -config.action_limit, config.action_limit)
    diagnostics = {
        "vx_estimate": vx_estimate,
        "vx_error": vx_error,
        "wz_error": wz_error,
        "wheel_difference_error": wheel_difference_error,
        "pitch_reference": pitch_reference,
    }
    return actions, next_integrals, diagnostics


@dataclass(frozen=True)
class DiscreteLQRResult:
    gain: np.ndarray
    riccati: np.ndarray
    solver: str
    iterations: int
    residual_max_abs: float
    closed_loop_eigenvalues: np.ndarray


def controllability_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return [B, AB, ..., A^(n-1)B] for a discrete linear system."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A must be square, got {a.shape}")
    if b.ndim != 2 or b.shape[0] != a.shape[0]:
        raise ValueError(f"B must have {a.shape[0]} rows, got {b.shape}")
    blocks = [b]
    for _ in range(1, a.shape[0]):
        blocks.append(a @ blocks[-1])
    return np.concatenate(blocks, axis=1)


def solve_discrete_lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 20_000,
) -> DiscreteLQRResult:
    """Solve the infinite-horizon discrete LQR problem without SciPy."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    state_dim = a.shape[0] if a.ndim == 2 else 0
    action_dim = b.shape[1] if b.ndim == 2 else 0
    if a.shape != (state_dim, state_dim) or b.shape != (state_dim, action_dim):
        raise ValueError(f"incompatible A/B shapes: {a.shape}, {b.shape}")
    if q.shape != a.shape or r.shape != (action_dim, action_dim):
        raise ValueError(f"incompatible Q/R shapes: {q.shape}, {r.shape}")
    if not all(np.isfinite(value).all() for value in (a, b, q, r)):
        raise ValueError("LQR matrices must be finite")
    if not np.allclose(q, q.T) or not np.allclose(r, r.T):
        raise ValueError("Q and R must be symmetric")
    if np.linalg.eigvalsh(q).min() < -1e-10:
        raise ValueError("Q must be positive semidefinite")
    if np.linalg.eigvalsh(r).min() <= 0.0:
        raise ValueError("R must be positive definite")

    p = q.copy()
    iterations = 0
    solver = "fixed_point"
    for iterations in range(1, max_iterations + 1):
        regularized_r = r + b.T @ p @ b
        gain = np.linalg.solve(regularized_r, b.T @ p @ a)
        p_next = q + a.T @ p @ a - a.T @ p @ b @ gain
        p_next = 0.5 * (p_next + p_next.T)
        if np.max(np.abs(p_next - p)) <= tolerance:
            p = p_next
            break
        p = p_next
    else:
        try:
            from scipy.linalg import solve_discrete_are
        except ImportError as exc:
            raise RuntimeError(
                f"discrete Riccati iteration did not converge in {max_iterations} steps"
            ) from exc
        p = solve_discrete_are(a, b, q, r)
        solver = "scipy_solve_discrete_are"
        iterations = 0

    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    residual = q + a.T @ p @ a - a.T @ p @ b @ gain - p
    eigenvalues = np.linalg.eigvals(a - b @ gain)
    return DiscreteLQRResult(
        gain=gain,
        riccati=p,
        solver=solver,
        iterations=iterations,
        residual_max_abs=float(np.max(np.abs(residual))),
        closed_loop_eigenvalues=eigenvalues,
    )


def lqr_action(
    states: np.ndarray,
    gain: np.ndarray,
    *,
    action_limit: float = 1.0,
) -> np.ndarray:
    """Apply u=-Kx and enforce the normalized two-wheel action contract."""
    states = np.asarray(states, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if states.shape[-1] != gain.shape[1]:
        raise ValueError(f"state/gain mismatch: {states.shape}, {gain.shape}")
    if not 0.0 < action_limit <= 1.0:
        raise ValueError(f"action_limit must be in (0, 1], got {action_limit}")
    return np.clip(-(states @ gain.T), -action_limit, action_limit)
