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


def recovery_window_steps(
    push_forces_n: np.ndarray,
    push_start_step: int,
    push_end_step: int,
) -> tuple[bool, int, int]:
    """Select measurement/recovery windows for initial-state or push gates."""
    forces = np.asarray(push_forces_n, dtype=np.float64)
    if forces.size == 0 or not np.isfinite(forces).all():
        raise ValueError("push forces must be a non-empty finite array")
    if push_start_step < 0 or push_end_step < push_start_step:
        raise ValueError("invalid push window")
    initial_condition_only = bool(np.allclose(forces, 0.0))
    if initial_condition_only:
        return True, 0, 0
    return False, push_start_step, push_end_step


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
    vx_ki: float = 0.05
    wz_kp: float = 0.25
    wz_ki: float = 0.10
    wz_feedforward: float = 0.6
    wheel_difference_kp: float = 0.0
    pitch_reference_limit_rad: float = math.radians(6.0)
    vx_integral_limit: float = 0.5
    wz_integral_limit: float = 2.0
    pitch_bias_adaptation_rate: float = 5.0
    pitch_bias_limit_rad: float = math.radians(4.0)
    pitch_bias_command_threshold: float = 0.01
    pitch_bias_pitch_rate_threshold: float = 0.05
    vx_reference_slew_rate_m_s2: float = 0.0
    wz_reference_slew_rate_rad_s2: float = 0.0
    path_progress_governor_enabled: bool = False
    governor_bias_start_rad: float = math.radians(0.5)
    governor_bias_full_rad: float = math.radians(2.5)
    governor_minimum_progress_scale: float = 0.75
    action_limit: float = 0.8


@dataclass(frozen=True)
class PlantVariation:
    """Deterministic simulation variation for controller robustness gates."""

    name: str
    mass_scale: float = 1.0
    target_total_mass_kg: float | None = None
    inertia_scale: float = 1.0
    com_offset_x_m: float = 0.0
    com_offset_z_m: float = 0.0
    static_friction: float | None = None
    dynamic_friction: float | None = None
    torque_scale: float = 1.0
    action_delay_steps: int = 0

    def __post_init__(self) -> None:
        values = [
            self.mass_scale,
            self.inertia_scale,
            self.com_offset_x_m,
            self.com_offset_z_m,
            self.torque_scale,
        ]
        values.extend(
            value
            for value in (
                self.target_total_mass_kg,
                self.static_friction,
                self.dynamic_friction,
            )
            if value is not None
        )
        if not self.name or not all(math.isfinite(value) for value in values):
            raise ValueError("plant variation must have a name and finite values")
        if self.mass_scale <= 0.0 or self.inertia_scale <= 0.0:
            raise ValueError("mass and inertia scales must be positive")
        if self.target_total_mass_kg is not None and self.target_total_mass_kg <= 0.0:
            raise ValueError("target total mass must be positive")
        if self.target_total_mass_kg is not None and self.mass_scale != 1.0:
            raise ValueError("set either mass scale or target total mass, not both")
        if (self.static_friction is None) != (self.dynamic_friction is None):
            raise ValueError("static and dynamic friction must both be set or both be preserved")
        if self.static_friction is not None and not (
            0.0 < self.dynamic_friction <= self.static_friction
        ):
            raise ValueError("friction must satisfy 0 < dynamic <= static")
        if not 0.0 < self.torque_scale <= 1.0:
            raise ValueError("torque scale must be in (0, 1]")
        if self.action_delay_steps < 0:
            raise ValueError("action delay must be non-negative")


def diagnostic_plant_variations() -> tuple[PlantVariation, ...]:
    """Return the bounded v1 matrix used before any randomized training."""
    return (
        PlantVariation("nominal"),
        PlantVariation("mass_0p85", mass_scale=0.85),
        PlantVariation("mass_1p15", mass_scale=1.15),
        PlantVariation("mass_1p25_stress", mass_scale=1.25),
        PlantVariation("com_x_minus_0p03", com_offset_x_m=-0.03),
        PlantVariation("com_x_plus_0p03", com_offset_x_m=0.03),
        PlantVariation("com_z_minus_0p05", com_offset_z_m=-0.05),
        PlantVariation("com_z_plus_0p05", com_offset_z_m=0.05),
        PlantVariation("inertia_0p8", inertia_scale=0.8),
        PlantVariation("inertia_1p2", inertia_scale=1.2),
        PlantVariation("friction_low", static_friction=0.65, dynamic_friction=0.55),
        PlantVariation("friction_high", static_friction=1.1, dynamic_friction=1.0),
        PlantVariation("torque_0p8", torque_scale=0.8),
        PlantVariation("delay_10ms", action_delay_steps=2),
        PlantVariation("delay_20ms", action_delay_steps=4),
        PlantVariation(
            "corner_heavy_high_com_low_grip_low_torque_delay",
            mass_scale=1.15,
            inertia_scale=1.2,
            com_offset_z_m=0.05,
            static_friction=0.65,
            dynamic_friction=0.55,
            torque_scale=0.8,
            action_delay_steps=4,
        ),
    )


def provisional_plant_variations() -> tuple[PlantVariation, ...]:
    """Return the guessed v1 operating envelope pending hardware measurement."""
    return (
        PlantVariation("nominal"),
        PlantVariation("mass_0p95", mass_scale=0.95),
        PlantVariation("mass_1p05", mass_scale=1.05),
        PlantVariation("com_x_minus_0p02", com_offset_x_m=-0.02),
        PlantVariation("com_x_plus_0p02", com_offset_x_m=0.02),
        PlantVariation("com_z_minus_0p03", com_offset_z_m=-0.03),
        PlantVariation("com_z_plus_0p03", com_offset_z_m=0.03),
        PlantVariation("inertia_0p85", inertia_scale=0.85),
        PlantVariation("inertia_1p15", inertia_scale=1.15),
        PlantVariation("friction_low", static_friction=0.70, dynamic_friction=0.60),
        PlantVariation("friction_high", static_friction=1.00, dynamic_friction=0.90),
        PlantVariation("torque_0p9", torque_scale=0.90),
        PlantVariation("delay_10ms", action_delay_steps=2),
        PlantVariation(
            "corner_provisional_v1",
            mass_scale=1.05,
            inertia_scale=1.15,
            com_offset_x_m=-0.02,
            com_offset_z_m=0.03,
            static_friction=0.70,
            dynamic_friction=0.60,
            torque_scale=0.90,
            action_delay_steps=2,
        ),
    )


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
    valid_controller_state_shapes = ((batch, 2), (batch, 3), (batch, 4), (batch, 6))
    if integrals.shape not in valid_controller_state_shapes:
        raise ValueError(
            "expected controller state shape "
            f"{(batch, 2)}, {(batch, 3)}, {(batch, 4)}, or {(batch, 6)}, got {integrals.shape}"
        )
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if config.wheel_radius_m <= 0.0 or config.wheel_track_m <= 0.0:
        raise ValueError("wheel geometry must be positive")
    if (
        config.vx_integral_limit <= 0.0
        or config.wz_integral_limit <= 0.0
        or config.pitch_reference_limit_rad <= 0.0
        or config.pitch_bias_limit_rad <= 0.0
        or config.action_limit <= 0.0
    ):
        raise ValueError("controller limits must be positive")
    if (
        config.pitch_bias_adaptation_rate < 0.0
        or config.vx_reference_slew_rate_m_s2 < 0.0
        or config.wz_reference_slew_rate_rad_s2 < 0.0
    ):
        raise ValueError("adaptation and reference slew rates must be non-negative")
    if not (
        0.0 <= config.governor_bias_start_rad < config.governor_bias_full_rad
        and 0.0 < config.governor_minimum_progress_scale <= 1.0
    ):
        raise ValueError("invalid path progress governor limits")

    stored_pitch_bias = (
        integrals[:, 2].copy()
        if integrals.shape[1] >= 3
        else np.zeros(batch, dtype=np.float64)
    )
    path_progress_scale = np.ones(batch, dtype=np.float64)
    if config.path_progress_governor_enabled:
        bias_severity = np.clip(
            (np.abs(stored_pitch_bias) - config.governor_bias_start_rad)
            / (config.governor_bias_full_rad - config.governor_bias_start_rad),
            0.0,
            1.0,
        )
        reinforces_equilibrium_bias = stored_pitch_bias * vx_ref > 0.0
        path_progress_scale[reinforces_equilibrium_bias] -= (
            bias_severity[reinforces_equilibrium_bias]
            * (1.0 - config.governor_minimum_progress_scale)
        )
    governed_vx_ref = vx_ref * path_progress_scale
    governed_wz_ref = wz_ref * path_progress_scale

    effective_vx_ref = governed_vx_ref.copy()
    effective_wz_ref = governed_wz_ref.copy()
    if integrals.shape[1] == 6:
        if config.vx_reference_slew_rate_m_s2 > 0.0:
            maximum_delta = config.vx_reference_slew_rate_m_s2 * control_dt
            effective_vx_ref = integrals[:, 4] + np.clip(
                governed_vx_ref - integrals[:, 4], -maximum_delta, maximum_delta
            )
        if config.wz_reference_slew_rate_rad_s2 > 0.0:
            maximum_delta = config.wz_reference_slew_rate_rad_s2 * control_dt
            effective_wz_ref = integrals[:, 5] + np.clip(
                governed_wz_ref - integrals[:, 5], -maximum_delta, maximum_delta
            )
    vx_estimate = config.wheel_radius_m * states[:, 3]
    vx_error = effective_vx_ref - vx_estimate
    wz_error = effective_wz_ref - states[:, 5]
    candidate_integrals = integrals.copy()
    candidate_integrals[:, 0] = np.clip(
        candidate_integrals[:, 0] + control_dt * vx_error,
        -config.vx_integral_limit,
        config.vx_integral_limit,
    )
    candidate_integrals[:, 1] = np.clip(
        candidate_integrals[:, 1] + control_dt * wz_error,
        -config.wz_integral_limit,
        config.wz_integral_limit,
    )
    if integrals.shape[1] == 6:
        candidate_integrals[:, 4] = effective_vx_ref
        candidate_integrals[:, 5] = effective_wz_ref

    # A forward command first pulls the wheels back to create a forward lean;
    # the frozen balance LQR then drives the wheels under the falling body.
    pitch_reference_unclipped = (
        config.vx_kp * vx_error + config.vx_ki * candidate_integrals[:, 0]
    )
    vx_integrator_blocked = (
        (pitch_reference_unclipped > config.pitch_reference_limit_rad) & (vx_error > 0.0)
    ) | (
        (pitch_reference_unclipped < -config.pitch_reference_limit_rad) & (vx_error < 0.0)
    )
    next_integrals = candidate_integrals.copy()
    next_integrals[vx_integrator_blocked, 0] = integrals[vx_integrator_blocked, 0]
    pitch_bias = np.zeros(batch, dtype=np.float64)
    pitch_bias_adapting = np.zeros(batch, dtype=bool)
    pitch_bias_calibrated = np.zeros(batch, dtype=bool)
    if integrals.shape[1] >= 3:
        pitch_bias = integrals[:, 2].copy()
        zero_command = (
            (np.abs(vx_ref) <= config.pitch_bias_command_threshold)
            & (np.abs(wz_ref) <= config.pitch_bias_command_threshold)
        )
        pitch_bias_calibrated = (
            integrals[:, 3] >= 0.5
            if integrals.shape[1] >= 4
            else np.zeros(batch, dtype=bool)
        )
        pitch_bias_adapting = (
            ~pitch_bias_calibrated
            & zero_command
            & (np.abs(states[:, 1]) <= config.pitch_bias_pitch_rate_threshold)
        )
        adaptation_alpha = np.clip(
            config.pitch_bias_adaptation_rate * control_dt, 0.0, 1.0
        )
        pitch_bias[pitch_bias_adapting] += adaptation_alpha * (
            states[pitch_bias_adapting, 0] - pitch_bias[pitch_bias_adapting]
        )
        pitch_bias = np.clip(
            pitch_bias, -config.pitch_bias_limit_rad, config.pitch_bias_limit_rad
        )
        next_integrals[:, 2] = pitch_bias
        if integrals.shape[1] >= 4:
            pitch_bias_calibrated |= ~zero_command
            next_integrals[:, 3] = pitch_bias_calibrated.astype(np.float64)
    applied_pitch_bias = np.where(pitch_bias_adapting, 0.0, pitch_bias)
    pitch_reference = np.clip(
        config.vx_kp * vx_error + config.vx_ki * next_integrals[:, 0],
        -config.pitch_reference_limit_rad,
        config.pitch_reference_limit_rad,
    )
    tracking_states = states.copy()
    tracking_states[:, 0] -= applied_pitch_bias + pitch_reference
    tracking_states[:, 3] -= effective_vx_ref / config.wheel_radius_m
    tracking_states[:, 4] -= (
        config.wheel_track_m / config.wheel_radius_m
    ) * effective_wz_ref
    actions = lqr_action(
        tracking_states,
        gain,
        action_limit=config.action_limit,
    )
    wheel_difference_target = (
        config.wheel_track_m / config.wheel_radius_m
    ) * effective_wz_ref
    wheel_difference_error = wheel_difference_target - states[:, 4]
    yaw_correction = (
        config.wz_kp * wz_error
        + config.wz_ki * next_integrals[:, 1]
        + config.wz_feedforward * effective_wz_ref
        + config.wheel_difference_kp * wheel_difference_error
    )
    yaw_unclipped = actions[:, 1] + yaw_correction
    wz_integrator_blocked = (
        (yaw_unclipped > config.action_limit) & (wz_error > 0.0)
    ) | ((yaw_unclipped < -config.action_limit) & (wz_error < 0.0))
    next_integrals[wz_integrator_blocked, 1] = integrals[wz_integrator_blocked, 1]
    yaw_correction = (
        config.wz_kp * wz_error
        + config.wz_ki * next_integrals[:, 1]
        + config.wz_feedforward * effective_wz_ref
        + config.wheel_difference_kp * wheel_difference_error
    )
    actions[:, 1] += yaw_correction
    actions = np.clip(actions, -config.action_limit, config.action_limit)
    diagnostics = {
        "requested_vx_ref": vx_ref,
        "requested_wz_ref": wz_ref,
        "governed_vx_ref": governed_vx_ref,
        "governed_wz_ref": governed_wz_ref,
        "path_progress_scale": path_progress_scale,
        "vx_estimate": vx_estimate,
        "effective_vx_ref": effective_vx_ref,
        "effective_wz_ref": effective_wz_ref,
        "vx_error": vx_error,
        "wz_error": wz_error,
        "wheel_difference_error": wheel_difference_error,
        "pitch_reference": pitch_reference,
        "pitch_bias": pitch_bias,
        "applied_pitch_bias": applied_pitch_bias,
        "pitch_bias_adapting": pitch_bias_adapting,
        "pitch_bias_calibrated": pitch_bias_calibrated,
        "vx_integrator_blocked": vx_integrator_blocked,
        "wz_integrator_blocked": wz_integrator_blocked,
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
