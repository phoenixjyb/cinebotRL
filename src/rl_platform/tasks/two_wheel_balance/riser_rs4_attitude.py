"""Pure RS4 attitude-command planning for the two-wheel riser robot.

The DJI command is a world camera attitude, not a physical motor-joint target.
This module therefore keeps the command envelope separate from the legacy CAD
gimbal joint limits used by Isaac articulation and rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_attitude import quaternion_matrix_wxyz, rotation_error_vector
from .riser_reference import CorrectedRiserReference, bidirectional_path_heading


# Current ronin_rs4_driver configuration in DJI DFR command order
# [yaw, roll, pitch]. The driver marks these values as requiring final
# on-device validation, so they are a simulation envelope, not hardware proof.
RS4_DFR_COMMAND_LOWER_RAD = np.deg2rad(np.array([-180.0, -95.0, -112.0]))
RS4_DFR_COMMAND_UPPER_RAD = np.deg2rad(np.array([180.0, 240.0, 214.0]))
RS4_HARD_RATE_LIMIT_DEG_S = 360.0
RS4_FILMING_RATE_LIMIT_DEG_S = 24.0
ACCEPTED62_BASE_YAW_RATE_RAD_S = 0.25
ACCEPTED62_BODY_BASIS_QUAT_XYZW = np.array(
    [
        0.37126688383868894,
        0.6136484680880162,
        -0.4508086827044413,
        0.5313830917297113,
    ]
)


def accepted62_body_basis_rotation() -> np.ndarray:
    return Rotation.from_quat(ACCEPTED62_BODY_BASIS_QUAT_XYZW).as_matrix()


@dataclass(frozen=True)
class Rs4AttitudePlan:
    base_yaw_rad: np.ndarray
    command_yaw_roll_pitch_rad: np.ndarray
    command_feasible: np.ndarray
    reconstruction_error_rad: np.ndarray
    command_rate_yaw_roll_pitch_deg_s: np.ndarray


def rs4_command_to_proxy_joint_order(
    command_yaw_roll_pitch_rad: np.ndarray,
) -> np.ndarray:
    """Map Ronin [yaw, roll, pitch] into legacy-named proxy joint order."""

    command = np.asarray(command_yaw_roll_pitch_rad, dtype=np.float64)
    if command.shape[-1] != 3:
        raise ValueError("RS4 command must have last dimension 3")
    return command[..., [2, 1, 0]]


def proxy_joint_to_rs4_command_order(proxy_joint_rad: np.ndarray) -> np.ndarray:
    """Map proxy [pitch, roll, yaw] back to Ronin command order."""

    proxy = np.asarray(proxy_joint_rad, dtype=np.float64)
    if proxy.shape[-1] != 3:
        raise ValueError("proxy joint vector must have last dimension 3")
    return proxy[..., [2, 1, 0]]


def proxy_joint_rates_rad_s(proxy_joint_rad: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Differentiate proxy joints while treating the yaw proxy as cyclic."""

    proxy = np.asarray(proxy_joint_rad, dtype=np.float64)
    time = np.asarray(time_s, dtype=np.float64)
    if proxy.ndim != 2 or proxy.shape[1] != 3 or time.shape != (len(proxy),):
        raise ValueError("proxy joints and time must have shapes (N,3) and (N,)")
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must contain at least two increasing samples")
    delta = np.diff(proxy, axis=0)
    delta[:, 2] = _wrap_to_pi(delta[:, 2])
    return delta / np.diff(time)[:, None]


def unwrap_proxy_joint_yaw(proxy_joint_rad: np.ndarray) -> np.ndarray:
    """Return continuous proxy targets while preserving equivalent FK."""

    proxy = np.asarray(proxy_joint_rad, dtype=np.float64)
    if proxy.ndim != 2 or proxy.shape[1] != 3 or not np.isfinite(proxy).all():
        raise ValueError("proxy joints must be finite shape (N,3)")
    result = proxy.copy()
    result[:, 2] = np.unwrap(result[:, 2])
    return result


def _wrap_to_pi(value: np.ndarray | float) -> np.ndarray | float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def bounded_path_yaw_schedule(
    reference: CorrectedRiserReference,
    *,
    maximum_yaw_rate_rad_s: float = 0.4,
) -> np.ndarray:
    """Rate-limit the bidirectional path heading from the recorded initial yaw."""

    if maximum_yaw_rate_rad_s <= 0.0 or not math.isfinite(maximum_yaw_rate_rad_s):
        raise ValueError("maximum yaw rate must be finite and positive")
    desired = bidirectional_path_heading(
        reference.positions_m[:, :2], reference.initial_base_yaw_rad
    )
    yaw = np.empty_like(desired)
    yaw[0] = reference.initial_base_yaw_rad
    for index in range(1, len(yaw)):
        dt = float(reference.time_s[index] - reference.time_s[index - 1])
        maximum_delta = maximum_yaw_rate_rad_s * dt
        error = float(_wrap_to_pi(desired[index] - yaw[index - 1]))
        yaw[index] = yaw[index - 1] + np.clip(
            error, -maximum_delta, maximum_delta
        )
    return yaw


def fit_corpus_centered_body_basis(
    references: dict[int, CorrectedRiserReference],
    yaw_schedules: dict[int, np.ndarray],
) -> np.ndarray:
    """Fit a fixed chassis-to-semantic-DFR zero-command bracket rotation."""

    if set(references) != set(yaw_schedules) or not references:
        raise ValueError("references and yaw schedules must have the same non-empty cases")
    body_targets = []
    for case, reference in sorted(references.items()):
        yaw = np.asarray(yaw_schedules[case], dtype=np.float64)
        if yaw.shape != reference.time_s.shape or not np.isfinite(yaw).all():
            raise ValueError(f"bad yaw schedule for case {case}")
        for base_yaw, quaternion in zip(
            yaw, reference.semantic_dfr_quat_wxyz, strict=True
        ):
            body_targets.append(
                Rotation.from_euler("z", -float(base_yaw)).as_matrix()
                @ quaternion_matrix_wxyz(quaternion)
            )
    return Rotation.from_matrix(np.stack(body_targets)).mean().as_matrix()


def compose_semantic_dfr_rotation(
    world_basis_rotation: np.ndarray,
    command_yaw_roll_pitch_rad: np.ndarray,
) -> np.ndarray:
    """Apply the deployed mapping yaw<-X, roll<-Y, pitch<-Z."""

    basis = np.asarray(world_basis_rotation, dtype=np.float64)
    command = np.asarray(command_yaw_roll_pitch_rad, dtype=np.float64)
    if basis.shape != (3, 3) or command.shape != (3,):
        raise ValueError("basis and command must have shapes (3,3) and (3,)")
    yaw, roll, pitch = command
    relative = Rotation.from_euler("ZYX", [pitch, roll, yaw]).as_matrix()
    return basis @ relative


def _command_candidates(relative_rotation: np.ndarray) -> list[np.ndarray]:
    pitch, roll, yaw = Rotation.from_matrix(relative_rotation).as_euler("ZYX")
    # ZYX has a second Euler branch away from singularity. Generate periodic
    # equivalents as well so the solver can preserve command continuity.
    branches = (
        np.array([yaw, roll, pitch]),
        np.array([yaw + math.pi, math.pi - roll, pitch + math.pi]),
        np.array([yaw + math.pi, -math.pi - roll, pitch + math.pi]),
    )
    candidates = []
    for branch in branches:
        for yaw_turns in range(-2, 3):
            for roll_turns in range(-1, 2):
                for pitch_turns in range(-1, 2):
                    candidates.append(
                        branch
                        + 2.0
                        * math.pi
                        * np.array([yaw_turns, roll_turns, pitch_turns])
                    )
    return candidates


def _valid_command_candidates(
    relative_rotation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[np.ndarray]:
    valid = []
    for candidate in _command_candidates(relative_rotation):
        if np.all(candidate >= lower - 1e-10) and np.all(candidate <= upper + 1e-10):
            if not any(np.allclose(candidate, item, atol=1e-10) for item in valid):
                valid.append(candidate)
    return valid


def _command_delta(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = np.asarray(current) - np.asarray(previous)
    delta = delta.copy()
    delta[..., 0] = _wrap_to_pi(delta[..., 0])
    return delta


def resolve_rs4_position_command(
    world_basis_rotation: np.ndarray,
    target_semantic_dfr_rotation: np.ndarray,
    previous_command_yaw_roll_pitch_rad: np.ndarray,
    *,
    lower_rad: np.ndarray = RS4_DFR_COMMAND_LOWER_RAD,
    upper_rad: np.ndarray = RS4_DFR_COMMAND_UPPER_RAD,
) -> tuple[np.ndarray, bool]:
    """Resolve one continuous, bounded DJI DFR position command."""

    basis = np.asarray(world_basis_rotation, dtype=np.float64)
    target = np.asarray(target_semantic_dfr_rotation, dtype=np.float64)
    previous = np.asarray(previous_command_yaw_roll_pitch_rad, dtype=np.float64)
    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    if basis.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("basis and target rotations must have shape (3,3)")
    if previous.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("command and limits must have shape (3,)")
    if not np.all(lower < upper):
        raise ValueError("command lower limits must be below upper limits")

    candidates = _command_candidates(basis.T @ target)
    valid = _valid_command_candidates(basis.T @ target, lower, upper)
    if valid:
        return min(
            valid,
            key=lambda item: float(np.linalg.norm(_command_delta(item, previous))),
        ), True

    def violation(candidate: np.ndarray) -> tuple[float, float]:
        clipped = np.clip(candidate, lower, upper)
        return (
            float(np.linalg.norm(candidate - clipped)),
            float(np.linalg.norm(clipped - previous)),
        )

    closest = min(candidates, key=violation)
    return np.clip(closest, lower, upper), False


def resolve_rs4_position_command_sequence(
    world_basis_rotations: np.ndarray,
    target_semantic_dfr_rotations: np.ndarray,
    time_s: np.ndarray,
    *,
    lower_rad: np.ndarray = RS4_DFR_COMMAND_LOWER_RAD,
    upper_rad: np.ndarray = RS4_DFR_COMMAND_UPPER_RAD,
    preferred_rate_deg_s: float = RS4_FILMING_RATE_LIMIT_DEG_S,
    hard_rate_deg_s: float = RS4_HARD_RATE_LIMIT_DEG_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Globally choose Euler branches that minimize command-rate discontinuity."""

    basis = np.asarray(world_basis_rotations, dtype=np.float64)
    target = np.asarray(target_semantic_dfr_rotations, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    count = len(times)
    if basis.shape != (count, 3, 3) or target.shape != (count, 3, 3):
        raise ValueError("basis and target rotations must have shape (N,3,3)")
    if count < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time must contain at least two increasing samples")
    if preferred_rate_deg_s <= 0.0 or hard_rate_deg_s <= 0.0:
        raise ValueError("command rates must be positive")

    choices: list[np.ndarray] = []
    feasible = np.empty(count, dtype=bool)
    for index in range(count):
        relative = basis[index].T @ target[index]
        valid = _valid_command_candidates(relative, lower, upper)
        feasible[index] = bool(valid)
        if not valid:
            fallback, _ = resolve_rs4_position_command(
                basis[index], target[index], np.zeros(3), lower_rad=lower, upper_rad=upper
            )
            valid = [fallback]
        choices.append(np.stack(valid))

    preferred_rate_rad_s = math.radians(preferred_rate_deg_s)
    hard_rate_rad_s = math.radians(hard_rate_deg_s)
    previous_cost = np.sum(
        (_command_delta(choices[0], np.zeros(3)) / preferred_rate_rad_s) ** 2,
        axis=1,
    )
    backpointers: list[np.ndarray] = []
    for index in range(1, count):
        dt = float(times[index] - times[index - 1])
        delta = _command_delta(
            choices[index][:, None, :], choices[index - 1][None, :, :]
        )
        rate = delta / dt
        transition = np.sum((rate / preferred_rate_rad_s) ** 2, axis=2)
        hard_excess = np.maximum(np.abs(rate) - hard_rate_rad_s, 0.0)
        transition += 1e6 * np.sum(
            (hard_excess / hard_rate_rad_s) ** 2, axis=2
        )
        total = transition + previous_cost[None, :]
        best_previous = np.argmin(total, axis=1)
        backpointers.append(best_previous)
        previous_cost = total[np.arange(len(choices[index])), best_previous]

    selection = np.empty(count, dtype=np.int64)
    selection[-1] = int(np.argmin(previous_cost))
    for index in range(count - 1, 0, -1):
        selection[index - 1] = backpointers[index - 1][selection[index]]
    command = np.stack(
        [choices[index][selection[index]] for index in range(count)]
    )
    return command, feasible


def plan_rs4_attitude_commands(
    reference: CorrectedRiserReference,
    body_basis_rotation: np.ndarray,
    base_yaw_rad: np.ndarray,
) -> Rs4AttitudePlan:
    """Resolve a complete semantic-DFR trajectory into DJI position commands."""

    body_basis = np.asarray(body_basis_rotation, dtype=np.float64)
    base_yaw = np.asarray(base_yaw_rad, dtype=np.float64)
    if body_basis.shape != (3, 3):
        raise ValueError("body basis rotation must have shape (3,3)")
    if base_yaw.shape != reference.time_s.shape or not np.isfinite(base_yaw).all():
        raise ValueError("base yaw must match reference time")

    count = len(base_yaw)
    world_basis = np.empty((count, 3, 3), dtype=np.float64)
    target = np.empty((count, 3, 3), dtype=np.float64)
    error = np.empty(count, dtype=np.float64)
    for index, (yaw, quaternion) in enumerate(
        zip(base_yaw, reference.semantic_dfr_quat_wxyz, strict=True)
    ):
        world_basis[index] = (
            Rotation.from_euler("z", float(yaw)).as_matrix() @ body_basis
        )
        target[index] = quaternion_matrix_wxyz(quaternion)
    command, feasible = resolve_rs4_position_command_sequence(
        world_basis, target, reference.time_s
    )
    for index in range(count):
        achieved = compose_semantic_dfr_rotation(world_basis[index], command[index])
        error[index] = float(
            np.linalg.norm(rotation_error_vector(achieved, target[index]))
        )

    rate = np.rad2deg(
        proxy_joint_rates_rad_s(
            rs4_command_to_proxy_joint_order(command), reference.time_s
        )[..., [2, 1, 0]]
    )
    return Rs4AttitudePlan(base_yaw, command, feasible, error, rate)
