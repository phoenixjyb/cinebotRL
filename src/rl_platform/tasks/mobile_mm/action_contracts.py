"""Named action contracts for the Proto2 mobile-manipulator tasks.

The current task has two different 9D action semantics under discussion:

* ``sim_6joint_gimbal_v1`` is the existing Isaac contract and remains the
  default.  It controls six URDF arm/wrist joints plus body-frame base velocity.
* ``rs4_attitude_rate_v1`` is a proposed deployment-oriented contract for a
  3-DOF Realman arm plus DJI RS4/RS5 attitude/rate commands.
* ``split_base_arm_attitude_v1`` gives the policy ownership of only the arm and
  chassis. A separate camera-attitude adapter owns the physical gimbal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionChannel:
    """One normalized policy output channel."""

    index: int
    name: str
    semantic: str
    unit: str
    normalized_min: float = -1.0
    normalized_max: float = 1.0


@dataclass(frozen=True)
class ActionContract:
    """Explicit policy action-space semantics."""

    name: str
    action_dim: int
    channels: tuple[ActionChannel, ...]
    arm_indices: tuple[int, ...]
    gimbal_or_attitude_indices: tuple[int, ...]
    base_indices: tuple[int, int, int]
    deployment_ready: bool
    description: str

    def __post_init__(self) -> None:
        if len(self.channels) != self.action_dim:
            raise ValueError(f"{self.name}: {len(self.channels)} channels != action_dim {self.action_dim}")
        expected_indices = tuple(range(self.action_dim))
        actual_indices = tuple(channel.index for channel in self.channels)
        if actual_indices != expected_indices:
            raise ValueError(f"{self.name}: channel indices must be contiguous {expected_indices}, got {actual_indices}")
        for group_name, indices in (
            ("arm_indices", self.arm_indices),
            ("gimbal_or_attitude_indices", self.gimbal_or_attitude_indices),
            ("base_indices", self.base_indices),
        ):
            if any(index < 0 or index >= self.action_dim for index in indices):
                raise ValueError(f"{self.name}: {group_name} out of range: {indices}")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    def describe(self) -> str:
        names = ", ".join(self.channel_names)
        return f"{self.name} ({self.action_dim}D): [{names}]"


SIM_6JOINT_GIMBAL_V1 = ActionContract(
    name="sim_6joint_gimbal_v1",
    action_dim=9,
    channels=(
        ActionChannel(0, "joint6_arm_yaw", "URDF arm J1 absolute position target", "rad"),
        ActionChannel(1, "joint5_arm_pitch", "URDF arm J2 absolute position target", "rad"),
        ActionChannel(2, "joint4_elbow_pitch", "URDF arm J3 absolute position target", "rad"),
        ActionChannel(3, "joint3_gimbal_yaw", "URDF wrist/gimbal J4 absolute position target", "rad"),
        ActionChannel(4, "joint2_gimbal_roll", "URDF wrist/gimbal J5 absolute position target", "rad"),
        ActionChannel(5, "joint1_gimbal_pitch", "URDF wrist/gimbal J6 absolute position target", "rad"),
        ActionChannel(6, "base_vx", "body-frame base forward velocity command", "m/s"),
        ActionChannel(7, "base_vy", "body-frame base lateral velocity command", "m/s"),
        ActionChannel(8, "base_wz", "body-frame base yaw-rate command", "rad/s"),
    ),
    arm_indices=(0, 1, 2, 3, 4, 5),
    gimbal_or_attitude_indices=(3, 4, 5),
    base_indices=(6, 7, 8),
    deployment_ready=False,
    description=(
        "Existing Isaac simulation contract. Internally self-consistent, but not "
        "proven equivalent to the DJI RS4/RS5 deployment command surface."
    ),
)


RS4_ATTITUDE_RATE_V1 = ActionContract(
    name="rs4_attitude_rate_v1",
    action_dim=9,
    channels=(
        ActionChannel(0, "arm_yaw", "Realman arm yaw target", "rad"),
        ActionChannel(1, "arm_pitch", "Realman arm pitch target", "rad"),
        ActionChannel(2, "arm_elbow", "Realman elbow target", "rad"),
        ActionChannel(3, "rs4_yaw_rate", "DJI RS4/RS5 camera attitude yaw-rate command", "deg/s"),
        ActionChannel(4, "rs4_pitch_rate", "DJI RS4/RS5 camera attitude pitch-rate command", "deg/s"),
        ActionChannel(5, "rs4_roll_rate", "DJI RS4/RS5 camera attitude roll-rate command; may be masked in v1", "deg/s"),
        ActionChannel(6, "base_vx", "body-frame base forward velocity command", "m/s"),
        ActionChannel(7, "base_vy", "body-frame base lateral velocity command", "m/s"),
        ActionChannel(8, "base_wz", "body-frame base yaw-rate command", "rad/s"),
    ),
    arm_indices=(0, 1, 2),
    gimbal_or_attitude_indices=(3, 4, 5),
    base_indices=(6, 7, 8),
    deployment_ready=False,
    description=(
        "Proposed RS4-aware experimental contract. Do not use for training until "
        "the simulator adapter, dataset schema, and hardware axis/sign mapping are validated."
    ),
)


SPLIT_BASE_ARM_ATTITUDE_V1 = ActionContract(
    name="split_base_arm_attitude_v1",
    action_dim=9,
    channels=(
        ActionChannel(0, "arm_yaw", "Realman arm yaw target", "rad"),
        ActionChannel(1, "arm_pitch", "Realman arm pitch target", "rad"),
        ActionChannel(2, "arm_elbow", "Realman elbow target", "rad"),
        ActionChannel(3, "attitude_adapter_reserved_yaw", "ignored; attitude adapter owns yaw", "none"),
        ActionChannel(4, "attitude_adapter_reserved_pitch", "ignored; attitude adapter owns pitch", "none"),
        ActionChannel(5, "attitude_adapter_reserved_roll", "ignored; attitude adapter owns roll", "none"),
        ActionChannel(6, "base_vx", "body-frame base forward velocity command", "m/s"),
        ActionChannel(7, "base_vy", "body-frame base lateral velocity command", "m/s"),
        ActionChannel(8, "base_wz", "body-frame base yaw-rate command", "rad/s"),
    ),
    arm_indices=(0, 1, 2),
    gimbal_or_attitude_indices=(3, 4, 5),
    base_indices=(6, 7, 8),
    deployment_ready=False,
    description=(
        "Split-teacher contract. The policy learns arm and chassis only; the "
        "Option-B physical-camera attitude adapter controls the simulated gimbal."
    ),
)


ACTION_CONTRACTS: dict[str, ActionContract] = {
    SIM_6JOINT_GIMBAL_V1.name: SIM_6JOINT_GIMBAL_V1,
    RS4_ATTITUDE_RATE_V1.name: RS4_ATTITUDE_RATE_V1,
    SPLIT_BASE_ARM_ATTITUDE_V1.name: SPLIT_BASE_ARM_ATTITUDE_V1,
}

# Current default remains unchanged.
DEFAULT_ACTION_CONTRACT = SIM_6JOINT_GIMBAL_V1
DEFAULT_ACTION_CONTRACT_NAME = DEFAULT_ACTION_CONTRACT.name


def available_action_contract_names() -> tuple[str, ...]:
    """Return supported action contract names in stable display order."""

    return tuple(ACTION_CONTRACTS.keys())


def get_action_contract(name: str) -> ActionContract:
    """Resolve a named action contract or raise a useful error."""

    try:
        return ACTION_CONTRACTS[name]
    except KeyError as exc:
        valid = ", ".join(available_action_contract_names())
        raise ValueError(f"unknown action contract {name!r}; valid contracts: {valid}") from exc
