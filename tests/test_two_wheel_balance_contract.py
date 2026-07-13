"""Pure tests for the two-wheel action and observation contract."""

import numpy as np

from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    OBSERVATION_NAMES,
    BalanceContract,
    CascadedLQRConfig,
    cascaded_lqr_action,
    compose_pd_residual_action,
    controllability_matrix,
    lqr_action,
    mix_common_yaw_effort,
    solve_discrete_lqr,
)


def test_contract_dimensions_and_rate() -> None:
    contract = BalanceContract()
    assert contract.action_dim == 2
    assert contract.observation_dim == 10
    assert contract.policy_hz == 200.0
    assert ACTION_NAMES == ("a_common", "a_yaw")
    assert LQR_STATE_NAMES == OBSERVATION_NAMES[:6]
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


def test_discrete_lqr_stabilizes_double_integrator() -> None:
    a = np.array([[1.0, 0.01], [0.0, 1.0]])
    b = np.array([[0.00005], [0.01]])
    result = solve_discrete_lqr(a, b, np.diag([10.0, 1.0]), np.diag([0.1]))
    assert np.linalg.matrix_rank(controllability_matrix(a, b)) == 2
    assert result.solver in {"fixed_point", "scipy_solve_discrete_are"}
    assert np.max(np.abs(result.closed_loop_eigenvalues)) < 1.0
    assert result.residual_max_abs < 1e-8


def test_lqr_action_uses_negative_feedback_and_clips() -> None:
    gain = np.array([[-2.0, 1.0], [0.5, -3.0]])
    action = lqr_action(np.array([[1.0, -1.0]]), gain, action_limit=0.8)
    np.testing.assert_allclose(action, [[0.8, -0.8]])


def test_cascaded_lqr_zero_command_matches_inner_lqr() -> None:
    gain = np.array(
        [[-4.0, -2.0, 0.0, 0.00025, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0065, 0.0]]
    )
    states = np.array([[0.01, -0.02, 0.0, 0.1, -0.1, 0.03]])
    action, integrals, diagnostics = cascaded_lqr_action(
        states,
        np.zeros(1),
        np.zeros(1),
        gain,
        np.zeros((1, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(vx_kp=0.0, vx_ki=0.0, wz_kp=0.0, wz_ki=0.0),
    )
    np.testing.assert_allclose(action, lqr_action(states, gain, action_limit=0.8))
    assert integrals.shape == (1, 2)
    assert diagnostics["pitch_reference"].shape == (1,)


def test_cascaded_lqr_signed_commands_produce_signed_actions() -> None:
    gain = np.array(
        [[-4.0, -2.0, 0.0, 0.00025, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0065, 0.0]]
    )
    actions, _, diagnostics = cascaded_lqr_action(
        np.zeros((2, len(LQR_STATE_NAMES))),
        np.array([0.2, -0.2]),
        np.array([0.4, -0.4]),
        gain,
        np.zeros((2, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(),
    )
    assert actions[0, 0] < 0.0
    assert actions[1, 0] > 0.0
    assert actions[0, 1] > 0.0
    assert actions[1, 1] < 0.0
    assert diagnostics["pitch_reference"][0] > 0.0
    assert diagnostics["pitch_reference"][1] < 0.0


def test_cascaded_lqr_enforces_integral_reference_and_action_limits() -> None:
    gain = np.array(
        [[-20.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    )
    config = CascadedLQRConfig(vx_kp=10.0, vx_ki=10.0, wz_kp=10.0, wz_ki=10.0)
    actions, integrals, diagnostics = cascaded_lqr_action(
        np.zeros((1, len(LQR_STATE_NAMES))),
        np.array([10.0]),
        np.array([10.0]),
        gain,
        np.zeros((1, 2)),
        control_dt=10.0,
        config=config,
    )
    assert abs(integrals[0, 0]) <= config.vx_integral_limit
    assert abs(integrals[0, 1]) <= config.wz_integral_limit
    assert abs(diagnostics["pitch_reference"][0]) <= config.pitch_reference_limit_rad
    assert np.max(np.abs(actions)) <= config.action_limit


def test_cascaded_lqr_wheel_difference_feedback_is_damping() -> None:
    gain = np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES)))
    states = np.zeros((2, len(LQR_STATE_NAMES)))
    target_difference = 0.620 / 0.1016 * 0.4
    states[1, 4] = 2.0 * target_difference
    actions, _, diagnostics = cascaded_lqr_action(
        states,
        np.zeros(2),
        np.full(2, 0.4),
        gain,
        np.zeros((2, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(vx_kp=0.0, wz_kp=0.0, wheel_difference_kp=0.1),
    )
    assert actions[0, 1] > 0.0
    assert actions[1, 1] < 0.0
    assert diagnostics["wheel_difference_error"][0] > 0.0
    assert diagnostics["wheel_difference_error"][1] < 0.0


def test_cascaded_lqr_yaw_feedforward_follows_command_sign() -> None:
    actions, _, _ = cascaded_lqr_action(
        np.zeros((2, len(LQR_STATE_NAMES))),
        np.zeros(2),
        np.array([0.4, -0.4]),
        np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES))),
        np.zeros((2, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(
            vx_kp=0.0,
            wz_kp=0.0,
            wheel_difference_kp=0.0,
            wz_feedforward=0.5,
        ),
    )
    np.testing.assert_allclose(actions[:, 1], [0.2, -0.2])
