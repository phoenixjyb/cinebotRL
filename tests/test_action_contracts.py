"""Tests for named mobile-manipulator action contracts."""

from __future__ import annotations

import pytest

from rl_platform.tasks.mobile_mm.action_contracts import (
    ACTION_CONTRACTS,
    DEFAULT_ACTION_CONTRACT,
    RS4_ATTITUDE_RATE_V1,
    SIM_6JOINT_GIMBAL_V1,
)


def test_default_contract_preserves_existing_sim_semantics():
    assert DEFAULT_ACTION_CONTRACT is SIM_6JOINT_GIMBAL_V1
    assert DEFAULT_ACTION_CONTRACT.name == "sim_6joint_gimbal_v1"
    assert DEFAULT_ACTION_CONTRACT.action_dim == 9
    assert DEFAULT_ACTION_CONTRACT.arm_indices == (0, 1, 2, 3, 4, 5)
    assert DEFAULT_ACTION_CONTRACT.gimbal_or_attitude_indices == (3, 4, 5)
    assert DEFAULT_ACTION_CONTRACT.base_indices == (6, 7, 8)
    assert DEFAULT_ACTION_CONTRACT.channel_names == (
        "joint6_arm_yaw",
        "joint5_arm_pitch",
        "joint4_elbow_pitch",
        "joint3_gimbal_yaw",
        "joint2_gimbal_roll",
        "joint1_gimbal_pitch",
        "base_vx",
        "base_vy",
        "base_wz",
    )


def test_rs4_contract_is_named_but_not_default_or_deployment_ready():
    assert RS4_ATTITUDE_RATE_V1.name == "rs4_attitude_rate_v1"
    assert RS4_ATTITUDE_RATE_V1.action_dim == 9
    assert RS4_ATTITUDE_RATE_V1.arm_indices == (0, 1, 2)
    assert RS4_ATTITUDE_RATE_V1.gimbal_or_attitude_indices == (3, 4, 5)
    assert RS4_ATTITUDE_RATE_V1.base_indices == (6, 7, 8)
    assert RS4_ATTITUDE_RATE_V1.channel_names == (
        "arm_yaw",
        "arm_pitch",
        "arm_elbow",
        "rs4_yaw_rate",
        "rs4_pitch_rate",
        "rs4_roll_rate",
        "base_vx",
        "base_vy",
        "base_wz",
    )
    assert not RS4_ATTITUDE_RATE_V1.deployment_ready
    assert DEFAULT_ACTION_CONTRACT is not RS4_ATTITUDE_RATE_V1


def test_contract_registry_contains_unique_names():
    assert ACTION_CONTRACTS[SIM_6JOINT_GIMBAL_V1.name] is SIM_6JOINT_GIMBAL_V1
    assert ACTION_CONTRACTS[RS4_ATTITUDE_RATE_V1.name] is RS4_ATTITUDE_RATE_V1
    assert len(ACTION_CONTRACTS) == 2


@pytest.mark.parametrize("contract", [SIM_6JOINT_GIMBAL_V1, RS4_ATTITUDE_RATE_V1])
def test_contract_indices_and_channels_are_consistent(contract):
    assert tuple(channel.index for channel in contract.channels) == tuple(range(contract.action_dim))
    assert len(set(contract.channel_names)) == contract.action_dim
    assert contract.base_indices == (6, 7, 8)
    assert contract.describe().startswith(f"{contract.name} (9D):")

