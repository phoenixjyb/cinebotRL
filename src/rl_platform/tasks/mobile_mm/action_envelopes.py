"""Named arm action envelopes for Proto2 policy and teacher-label experiments."""

from __future__ import annotations

from typing import Final

import numpy as np


ARM_ENVELOPE_PROTO2_SAFE_V1: Final[str] = "proto2_safe_v1"
ARM_ENVELOPE_TEACHER_WIDE_V1: Final[str] = "teacher_wide_v1"
ARM_ENVELOPE_PROFILES: Final[tuple[str, ...]] = (
    ARM_ENVELOPE_PROTO2_SAFE_V1,
    ARM_ENVELOPE_TEACHER_WIDE_V1,
)

ARM_SAFE_HOME: Final[np.ndarray] = np.array(
    [0.0, 1.0, -1.2, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
ARM_ACTION_RADIUS_PROTO2_SAFE: Final[np.ndarray] = np.array(
    [1.0, 0.45, 0.8, 1.0, 0.8, 0.8],
    dtype=np.float32,
)

# Offline fallback copied from assets_own/recomoProto2-1190_moveit.urdf. Isaac
# envs should pass live USD/URDF limits instead of relying on these constants.
ARM_URDF_LOWER_HINT: Final[np.ndarray] = np.array(
    [-3.14159265, -1.57079633, -2.35619449, -3.14159265, -3.2, -3.2],
    dtype=np.float32,
)
ARM_URDF_UPPER_HINT: Final[np.ndarray] = np.array(
    [3.14159265, 1.57079633, 2.35619449, 3.14159265, 1.57079633, 1.57079633],
    dtype=np.float32,
)


def validate_arm_envelope_profile(profile: str) -> str:
    """Return a normalized profile name or raise for unknown profiles."""
    if profile not in ARM_ENVELOPE_PROFILES:
        valid = ", ".join(ARM_ENVELOPE_PROFILES)
        raise ValueError(f"unknown arm action envelope profile {profile!r}; valid: {valid}")
    return profile


def get_arm_action_envelope(
    profile: str,
    *,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    margin: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve physical lower/upper joint targets for a normalized arm action.

    ``proto2_safe_v1`` preserves the existing conservative home +/- radius
    policy envelope. ``teacher_wide_v1`` is intentionally experimental: it uses
    URDF/USD joint limits minus margin so GIK labels can be tested with minimal
    clipping before deciding whether the wider space is trainable.
    """
    profile = validate_arm_envelope_profile(profile)
    lower = np.asarray(
        ARM_URDF_LOWER_HINT if joint_lower is None else joint_lower,
        dtype=np.float32,
    )
    upper = np.asarray(
        ARM_URDF_UPPER_HINT if joint_upper is None else joint_upper,
        dtype=np.float32,
    )
    lower_margin = lower + np.float32(margin)
    upper_margin = upper - np.float32(margin)

    if profile == ARM_ENVELOPE_PROTO2_SAFE_V1:
        return (
            np.maximum(lower_margin, ARM_SAFE_HOME - ARM_ACTION_RADIUS_PROTO2_SAFE),
            np.minimum(upper_margin, ARM_SAFE_HOME + ARM_ACTION_RADIUS_PROTO2_SAFE),
        )

    if profile == ARM_ENVELOPE_TEACHER_WIDE_V1:
        return lower_margin, upper_margin

    raise AssertionError(f"unhandled arm envelope profile: {profile}")


def normalize_arm_targets(
    q_arm: np.ndarray,
    *,
    profile: str,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    margin: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize physical six-joint arm targets into policy action rows."""
    lower, upper = get_arm_action_envelope(
        profile,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        margin=margin,
    )
    q_arm = np.asarray(q_arm, dtype=np.float32)
    denom = np.maximum(upper - lower, np.float32(1e-6))
    raw = 2.0 * (q_arm - lower[None, :]) / denom[None, :] - 1.0
    valid = np.abs(raw) <= (1.0 + 1e-6)
    return np.clip(raw, -1.0, 1.0).astype(np.float32), raw.astype(np.float32), valid
