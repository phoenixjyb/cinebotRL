"""Pure tests for the two-wheel action and observation contract."""

import numpy as np

from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    BalanceContract,
    compose_pd_residual_action,
    mix_common_yaw_effort,
)


def test_contract_dimensions_and_rate() -> None:
    contract = BalanceContract()
    assert contract.action_dim == 2
    assert contract.observation_dim == 10
    assert contract.policy_hz == 200.0
    assert ACTION_NAMES == ("a_common", "a_yaw")
    assert "vx" not in OBSERVATION_NAMES


def test_common_effort_drives_both_wheels_equally() -> None:
    effort = mix_common_yaw_effort(np.array([[0.5, 0.0]]), 20.0)
    np.testing.assert_allclose(effort, [[10.0, 10.0]])


def test_yaw_effort_is_antisymmetric() -> None:
    effort = mix_common_yaw_effort(np.array([[0.0, 0.25]]), 20.0)
    np.testing.assert_allclose(effort, [[-5.0, 5.0]])


def test_mixer_clips_each_wheel() -> None:
    effort = mix_common_yaw_effort(np.array([[1.0, 1.0]]), 20.0)
    np.testing.assert_allclose(effort, [[0.0, 20.0]])


def test_pd_residual_zero_matches_proven_feedback() -> None:
    action = compose_pd_residual_action(
        np.array([0.1]),
        np.array([0.2]),
        np.zeros((1, 2)),
    )
    np.testing.assert_allclose(action, [[0.14, 0.0]])


def test_pd_residual_is_bounded_and_scaled() -> None:
    action = compose_pd_residual_action(
        np.array([0.0]),
        np.array([0.0]),
        np.array([[1.0, -1.0]]),
    )
    np.testing.assert_allclose(action, [[0.15, -0.15]])
