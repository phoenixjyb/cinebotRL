"""Centralised task specification for the Cinebot RL environment.

The values below are derived from `assets_own/mobile_arm_whole_body` so the
simulation stack, Isaac Lab tasks, and ROS control nodes share a single source
of truth. Update this module whenever mechanical or sensing specs change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float
    max_velocity: float
    max_effort: float


ARM_JOINT_ORDER: List[str] = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
]

ARM_JOINT_LIMITS: Dict[str, JointLimit] = {
    "left_arm_joint1": JointLimit(-2.8798, 2.8798, 1.6, 40.0),
    "left_arm_joint2": JointLimit(0.0, 3.2289, 1.6, 40.0),
    "left_arm_joint3": JointLimit(-3.3161, 0.0, 4.0, 27.0),
    "left_arm_joint4": JointLimit(-2.8798, 2.8798, 4.0, 7.0),
    "left_arm_joint5": JointLimit(-1.6581, 1.6581, 4.0, 7.0),
    "left_arm_joint6": JointLimit(-2.8798, 2.8798, 4.0, 7.0),
}

BASE_CMD_KEYS: Tuple[str, ...] = ("v_x", "omega_z")
BASE_CMD_LIMITS: Dict[str, Tuple[float, float, float]] = {
    # (min, max, rate_limit) - provisional until chassis testing supplies data.
    "v_x": (-1.5, 1.5, 0.75),
    "omega_z": (-2.5, 2.5, 1.0),
}


@dataclass
class ObservationChannel:
    """Describes one slice of the observation vector."""

    name: str
    description: str
    dimension: int
    source: str  # e.g. "sim/rigid_body", "ros/joint_state"


def default_observation_schema() -> List[ObservationChannel]:
    """Canonical observation layout for whole-body control."""

    return [
        ObservationChannel(
            "base_planar_pose",
            "Planar pose (x, y, yaw) of `chassis_center_link` in odom/world frame",
            3,
            "sim/odometry",
        ),
        ObservationChannel(
            "base_planar_twist",
            "Planar velocity (vx, vy, omega) reported by odom tracker",
            3,
            "sim/odometry",
        ),
        ObservationChannel(
            "arm_joint_positions",
            "Joint positions `[q1..q6]` in radians following ARM_JOINT_ORDER",
            len(ARM_JOINT_ORDER),
            "sim/joint_state",
        ),
        ObservationChannel(
            "arm_joint_velocities",
            "Joint velocities `[dq1..dq6]` in rad/s following ARM_JOINT_ORDER",
            len(ARM_JOINT_ORDER),
            "sim/joint_state",
        ),
        ObservationChannel(
            "ee_pose_error",
            "6D pose error between desired camera waypoint and `left_gripper_link`",
            6,
            "planner/trajectory_tracker",
        ),
        ObservationChannel(
            "manipulability",
            "Scalar manipulability index at current joint configuration",
            1,
            "analytics/manipulability",
        ),
    ]


@dataclass
class ActionSpec:
    """Defines action space bounds for coordinated base + arm control."""

    name: str
    dimension: int
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    rate_limit: Tuple[float, ...]


def default_action_spec() -> List[ActionSpec]:
    """Derived action limits for diff-drive base and 6-DoF arm."""

    base_lower = tuple(BASE_CMD_LIMITS[key][0] for key in BASE_CMD_KEYS)
    base_upper = tuple(BASE_CMD_LIMITS[key][1] for key in BASE_CMD_KEYS)
    base_rates = tuple(BASE_CMD_LIMITS[key][2] for key in BASE_CMD_KEYS)

    vel_limits = [ARM_JOINT_LIMITS[joint].max_velocity for joint in ARM_JOINT_ORDER]
    velocity_lower = tuple(-v for v in vel_limits)
    velocity_upper = tuple(vel_limits)
    velocity_rate = tuple(v * 0.5 for v in vel_limits)

    position_lower = tuple(ARM_JOINT_LIMITS[joint].lower for joint in ARM_JOINT_ORDER)
    position_upper = tuple(ARM_JOINT_LIMITS[joint].upper for joint in ARM_JOINT_ORDER)

    return [
        ActionSpec(
            name="base_planar_cmd",
            dimension=len(BASE_CMD_KEYS),
            lower=base_lower,
            upper=base_upper,
            rate_limit=base_rates,
        ),
        ActionSpec(
            name="arm_joint_position_targets",
            dimension=len(ARM_JOINT_ORDER),
            lower=position_lower,
            upper=position_upper,
            rate_limit=tuple(v * 0.25 for v in vel_limits),
        ),
        ActionSpec(
            name="arm_joint_velocity_cmd",
            dimension=len(ARM_JOINT_ORDER),
            lower=velocity_lower,
            upper=velocity_upper,
            rate_limit=velocity_rate,
        ),
    ]


@dataclass
class RewardTerm:
    key: str
    weight: float
    description: str


DEFAULT_REWARD_TERMS: List[RewardTerm] = [
    RewardTerm("pose_tracking", 1.0, "Negative norm of base + EE pose error"),
    RewardTerm(
        "velocity_penalty",
        -0.02,
        "Quadratic penalty on joint/base speeds exceeding comfort bands",
    ),
    RewardTerm(
        "smoothness",
        -0.05,
        "Penalise action deltas to avoid jerky camera motion",
    ),
    RewardTerm(
        "manipulability_bonus",
        0.1,
        "Reward maintaining manipulability > threshold",
    ),
    RewardTerm(
        "collision_penalty",
        -5.0,
        "Large penalty for contacts with environment or self-collision",
    ),
]


def default_reward_terms() -> List[RewardTerm]:
    """Return mutable copy of reward configuration."""

    return list(DEFAULT_REWARD_TERMS)


def build_observation(state: Dict[str, float]) -> List[float]:
    """Stub for converting simulator/robot state dict into flat observation vector."""

    raise NotImplementedError("Implement observation encoding once state plumbing is defined")


def compute_reward(signals: Dict[str, float]) -> float:
    """Stub for combining reward signals using `default_reward_terms`."""

    raise NotImplementedError("Hook reward calculation after terms are validated in sim")


def enforce_action_limits(action: List[float]) -> List[float]:
    """Placeholder for clamping/rate-limiting actions before sending to low-level controller."""

    raise NotImplementedError("Implement actuator-aware clamping once hardware specs are finalised")


# ============================================================================
# Isaac Lab Task Registration
# ============================================================================

def register_isaac_lab_tasks():
    """Register custom Isaac Lab tasks.
    
    This function should be called during Isaac Lab environment initialization
    to make our custom tasks available for training.
    """
    try:
        import gymnasium as gym
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg

        def register_once(task_id: str, entry_point: str) -> None:
            if task_id in gym.envs.registry:
                return
            gym.register(id=task_id, entry_point=entry_point)

        # Primary tracking task. Do not pass cfg in kwargs; gym.make() passes
        # num_envs and trajectory overrides directly into the environment.
        register_once(
            "MobileMMTrackEE-v0",
            "rl_platform.tasks.mobile_mm:MobileMMTrackEEEnv",
        )
        print("[task_spec] Registered Isaac Lab task: MobileMMTrackEE-v0")

        # Explicit Proto2 alias for the active recomoProto2-1190 USD baseline.
        register_once(
            "RecomoProto2TrackEE-v0",
            "rl_platform.tasks.mobile_mm:MobileMMTrackEEEnv",
        )
        print("[task_spec] Registered Isaac Lab task: RecomoProto2TrackEE-v0")

        # RecomoProto1 compatibility task. The implementation currently points
        # to the Proto2 USD, but the ID is retained for older scripts.
        from rl_platform.tasks.recomoproto1 import RecomoProto1TrackEEEnv, RecomoProto1TrackEEEnvCfg  # noqa: F401
        register_once(
            "RecomoProto1TrackEE-v0",
            "rl_platform.tasks.recomoproto1:RecomoProto1TrackEEEnv",
        )
        print("[task_spec] Registered Isaac Lab task: RecomoProto1TrackEE-v0")

        # Independent low-level balance task. It intentionally does not alias
        # or modify the mobile-manipulator tracking environment.
        register_once(
            "RecomoTwoWheelBalance-v0",
            "rl_platform.tasks.two_wheel_balance:RecomoTwoWheelBalanceEnv",
        )
        print("[task_spec] Registered Isaac Lab task: RecomoTwoWheelBalance-v0")

    except ImportError as e:
        print(f"[task_spec] Could not register Isaac Lab tasks: {e}")
        print("[task_spec] This is expected if running outside Isaac Lab environment")
