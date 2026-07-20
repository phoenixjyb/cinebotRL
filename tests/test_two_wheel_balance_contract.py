"""Pure tests for the two-wheel action and observation contract."""

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    OBSERVATION_NAMES,
    BalanceContract,
    CascadedLQRConfig,
    PlantVariation,
    allocate_common_yaw_action,
    cascaded_lqr_config,
    cascaded_lqr_action,
    compose_pd_residual_action,
    controllability_matrix,
    diagnostic_plant_variations,
    lqr_action,
    mix_common_yaw_effort,
    provisional_plant_variations,
    recovery_window_steps,
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


def test_initial_condition_recovery_starts_at_step_zero() -> None:
    assert recovery_window_steps(np.zeros(6), 200, 220) == (True, 0, 0)


def test_push_recovery_starts_after_push_window() -> None:
    assert recovery_window_steps(np.array([-60.0, 60.0]), 200, 220) == (
        False,
        200,
        220,
    )


def test_common_effort_drives_both_wheels_equally() -> None:
    effort = mix_common_yaw_effort(np.array([[0.5, 0.0]]), 20.0)
    np.testing.assert_allclose(effort, [[10.0, 10.0]])


def test_yaw_effort_is_antisymmetric() -> None:
    effort = mix_common_yaw_effort(np.array([[0.0, 0.25]]), 20.0)
    np.testing.assert_allclose(effort, [[-5.0, 5.0]])


def test_mixer_clips_each_wheel() -> None:
    effort = mix_common_yaw_effort(np.array([[1.0, 1.0]]), 20.0)
    np.testing.assert_allclose(effort, [[0.0, 20.0]])


def test_mixer_reports_hidden_combined_action_clipping() -> None:
    wheel, effective, saturated = allocate_common_yaw_action(np.array([[0.8, 0.8]]))
    np.testing.assert_allclose(wheel, [[0.0, 1.0]])
    np.testing.assert_allclose(effective, [[0.5, 0.5]])
    np.testing.assert_array_equal(saturated, [[False, True]])


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


def test_cascaded_lqr_exposes_exact_longitudinal_action_contributions() -> None:
    gain = np.array(
        [[-4.0, -1.5, 0.0, 0.0002, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0065, 0.0]]
    )
    states = np.array([[0.13, -0.34, 0.0, 4.3, 0.0, 0.0]])
    action, _, diagnostics = cascaded_lqr_action(
        states,
        np.zeros(1),
        np.zeros(1),
        gain,
        np.zeros((1, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(vx_kp=0.0, vx_ki=0.0, wz_kp=0.0, wz_ki=0.0),
    )

    contributions = diagnostics["common_action_state_contributions"][0]
    assert contributions[0] > 0.0
    assert contributions[1] < 0.0
    assert contributions[3] < 0.0
    assert diagnostics["common_action_unclipped"][0] == pytest.approx(
        np.sum(contributions)
    )
    assert action[0, 0] == pytest.approx(np.sum(contributions))


def test_cascaded_lqr_root_feedback_changes_only_outer_velocity_error() -> None:
    gain = np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES)))
    states = np.zeros((1, len(LQR_STATE_NAMES)))
    states[0, 3] = -0.38 / 0.1016
    reference = np.array([-0.4])
    _, _, legacy = cascaded_lqr_action(
        states,
        reference,
        np.zeros(1),
        gain,
        np.zeros((1, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(),
    )
    _, _, candidate = cascaded_lqr_action(
        states,
        reference,
        np.zeros(1),
        gain,
        np.zeros((1, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(),
        outer_vx_feedback_m_s=np.array([0.05]),
    )
    np.testing.assert_allclose(legacy["wheel_vx_estimate"], [-0.38])
    np.testing.assert_allclose(candidate["wheel_vx_estimate"], [-0.38])
    np.testing.assert_allclose(candidate["outer_vx_feedback"], [0.05])
    np.testing.assert_allclose(legacy["vx_error"], [-0.02])
    np.testing.assert_allclose(candidate["vx_error"], [-0.45])
    assert abs(candidate["pitch_reference"][0]) > abs(
        legacy["pitch_reference"][0]
    )
    assert not legacy["outer_vx_feedback_is_root"][0]
    assert candidate["outer_vx_feedback_is_root"][0]

    explicit_wheel_action, explicit_wheel_state, explicit_wheel = (
        cascaded_lqr_action(
            states,
            reference,
            np.zeros(1),
            gain,
            np.zeros((1, 2)),
            control_dt=0.02,
            config=CascadedLQRConfig(),
            outer_vx_feedback_m_s=np.array([-0.38]),
        )
    )
    legacy_action, legacy_state, _ = cascaded_lqr_action(
        states,
        reference,
        np.zeros(1),
        gain,
        np.zeros((1, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(),
    )
    np.testing.assert_allclose(explicit_wheel_action, legacy_action)
    np.testing.assert_allclose(explicit_wheel_state, legacy_state)
    np.testing.assert_allclose(explicit_wheel["vx_error"], legacy["vx_error"])


def test_cascaded_lqr_rejects_invalid_root_velocity_feedback() -> None:
    with pytest.raises(ValueError, match="outer velocity feedback"):
        cascaded_lqr_action(
            np.zeros((1, len(LQR_STATE_NAMES))),
            np.zeros(1),
            np.zeros(1),
            np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES))),
            np.zeros((1, 2)),
            control_dt=0.02,
            config=CascadedLQRConfig(),
            outer_vx_feedback_m_s=np.array([float("nan")]),
        )


def test_cascaded_lqr_blocks_integrators_that_drive_further_into_limits() -> None:
    gain = np.zeros((2, len(LQR_STATE_NAMES)))
    _, integrals, diagnostics = cascaded_lqr_action(
        np.zeros((1, len(LQR_STATE_NAMES))),
        np.array([1.0]),
        np.array([1.0]),
        gain,
        np.zeros((1, 2)),
        control_dt=0.02,
        config=CascadedLQRConfig(
            vx_kp=1.0,
            vx_ki=1.0,
            wz_kp=1.0,
            wz_ki=1.0,
            wz_feedforward=1.0,
            pitch_reference_limit_rad=0.1,
            action_limit=0.8,
        ),
    )
    np.testing.assert_allclose(integrals, np.zeros((1, 2)))
    assert diagnostics["vx_integrator_blocked"][0]
    assert diagnostics["wz_integrator_blocked"][0]


def test_cascaded_lqr_slews_references_and_persists_effective_commands() -> None:
    gain = np.zeros((2, len(LQR_STATE_NAMES)))
    config = CascadedLQRConfig(
        vx_reference_slew_rate_m_s2=0.5,
        wz_reference_slew_rate_rad_s2=1.0,
    )
    _, controller_state, diagnostics = cascaded_lqr_action(
        np.zeros((1, len(LQR_STATE_NAMES))),
        np.array([0.2]),
        np.array([-0.4]),
        gain,
        np.zeros((1, 6)),
        control_dt=0.02,
        config=config,
    )
    np.testing.assert_allclose(diagnostics["effective_vx_ref"], [0.01])
    np.testing.assert_allclose(diagnostics["effective_wz_ref"], [-0.02])
    np.testing.assert_allclose(controller_state[:, 4:], [[0.01, -0.02]])

    _, controller_state, diagnostics = cascaded_lqr_action(
        np.zeros((1, len(LQR_STATE_NAMES))),
        np.array([0.2]),
        np.array([-0.4]),
        gain,
        controller_state,
        control_dt=0.02,
        config=config,
    )
    np.testing.assert_allclose(diagnostics["effective_vx_ref"], [0.02])
    np.testing.assert_allclose(diagnostics["effective_wz_ref"], [-0.04])
    np.testing.assert_allclose(controller_state[:, 4:], [[0.02, -0.04]])


def test_cascaded_lqr_governor_retimes_only_bias_reinforcing_motion() -> None:
    gain = np.zeros((2, len(LQR_STATE_NAMES)))
    controller_state = np.zeros((3, 6))
    controller_state[:, 2] = np.radians([3.0, 3.0, 0.25])
    controller_state[:, 3] = 1.0
    _, _, diagnostics = cascaded_lqr_action(
        np.zeros((3, len(LQR_STATE_NAMES))),
        np.array([0.2, -0.2, 0.2]),
        np.array([0.4, -0.4, 0.4]),
        gain,
        controller_state,
        control_dt=0.02,
        config=CascadedLQRConfig(path_progress_governor_enabled=True),
    )
    np.testing.assert_allclose(diagnostics["path_progress_scale"], [0.75, 1.0, 1.0])
    np.testing.assert_allclose(diagnostics["governed_vx_ref"], [0.15, -0.2, 0.2])
    np.testing.assert_allclose(diagnostics["governed_wz_ref"], [0.3, -0.4, 0.4])
    np.testing.assert_allclose(diagnostics["requested_vx_ref"], [0.2, -0.2, 0.2])


def test_cascaded_lqr_governor_is_disabled_by_default() -> None:
    controller_state = np.zeros((1, 6))
    controller_state[0, 2:4] = [np.radians(4.0), 1.0]
    _, _, diagnostics = cascaded_lqr_action(
        np.zeros((1, len(LQR_STATE_NAMES))),
        np.array([0.2]),
        np.array([0.4]),
        np.zeros((2, len(LQR_STATE_NAMES))),
        controller_state,
        control_dt=0.02,
        config=CascadedLQRConfig(),
    )
    np.testing.assert_allclose(diagnostics["path_progress_scale"], [1.0])
    np.testing.assert_allclose(diagnostics["governed_vx_ref"], [0.2])
    np.testing.assert_allclose(diagnostics["governed_wz_ref"], [0.4])


def test_cascaded_lqr_extended_governor_includes_opposing_bias() -> None:
    controller_state = np.zeros((2, 6))
    controller_state[:, 2] = np.radians(2.0)
    controller_state[:, 3] = 1.0
    _, _, diagnostics = cascaded_lqr_action(
        np.zeros((2, len(LQR_STATE_NAMES))),
        np.array([-0.2, 0.2]),
        np.full(2, -0.4),
        np.zeros((2, len(LQR_STATE_NAMES))),
        controller_state,
        control_dt=0.02,
        config=CascadedLQRConfig(
            path_progress_governor_enabled=True,
            governor_include_opposing_bias=True,
        ),
    )
    expected_bias_scale = 1.0 - 0.75 * 0.25
    np.testing.assert_allclose(
        diagnostics["path_progress_scale"], [expected_bias_scale, expected_bias_scale]
    )


def test_cascaded_lqr_estimates_equilibrium_pitch_only_at_zero_command() -> None:
    gain = np.zeros((2, len(LQR_STATE_NAMES)))
    states = np.zeros((1, len(LQR_STATE_NAMES)))
    states[0, 0] = 0.05
    _, controller_state, diagnostics = cascaded_lqr_action(
        states,
        np.zeros(1),
        np.zeros(1),
        gain,
        np.zeros((1, 4)),
        control_dt=0.02,
        config=CascadedLQRConfig(pitch_bias_adaptation_rate=5.0),
    )
    np.testing.assert_allclose(controller_state[0, 2], 0.005)
    assert diagnostics["pitch_bias_adapting"][0]
    np.testing.assert_allclose(diagnostics["applied_pitch_bias"], np.zeros(1))

    _, frozen_state, frozen_diagnostics = cascaded_lqr_action(
        states,
        np.ones(1) * 0.2,
        np.zeros(1),
        gain,
        controller_state,
        control_dt=0.02,
        config=CascadedLQRConfig(pitch_bias_adaptation_rate=5.0),
    )
    np.testing.assert_allclose(frozen_state[0, 2], controller_state[0, 2])
    assert frozen_state[0, 3] == 1.0
    assert not frozen_diagnostics["pitch_bias_adapting"][0]
    assert frozen_diagnostics["pitch_bias_calibrated"][0]
    np.testing.assert_allclose(
        frozen_diagnostics["applied_pitch_bias"], controller_state[:, 2]
    )

    _, stopped_state, stopped_diagnostics = cascaded_lqr_action(
        states,
        np.zeros(1),
        np.zeros(1),
        gain,
        frozen_state,
        control_dt=0.02,
        config=CascadedLQRConfig(pitch_bias_adaptation_rate=5.0),
    )
    np.testing.assert_allclose(stopped_state, frozen_state)
    assert not stopped_diagnostics["pitch_bias_adapting"][0]
    assert stopped_diagnostics["pitch_bias_calibrated"][0]
    np.testing.assert_allclose(
        stopped_diagnostics["applied_pitch_bias"], controller_state[:, 2]
    )


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
            vx_ki=0.0,
            wz_kp=0.0,
            wz_ki=0.0,
            wheel_difference_kp=0.0,
            wz_feedforward=0.5,
        ),
    )
    np.testing.assert_allclose(actions[:, 1], [0.2, -0.2])


def test_case74_bounded_yaw_rate_gain_adds_only_corrective_yaw_action() -> None:
    gain = np.array(
        [
            [-3.9542270864, -1.4129892407, 0.0, 0.0002045618, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0064842532, 0.0],
        ]
    )
    state = np.zeros((1, len(LQR_STATE_NAMES)))
    state[0, 5] = -0.1914618313
    requested_wz = np.array([-0.3684858926])
    controller_state = np.zeros((1, 6))
    baseline, _, _ = cascaded_lqr_action(
        state,
        np.zeros(1),
        requested_wz,
        gain,
        controller_state,
        control_dt=0.02,
        config=cascaded_lqr_config("structural_robust_v1"),
    )
    candidate, _, _ = cascaded_lqr_action(
        state,
        np.zeros(1),
        requested_wz,
        gain,
        controller_state,
        control_dt=0.02,
        config=cascaded_lqr_config("structural_robust_v1", wz_kp=0.4),
    )
    np.testing.assert_allclose(candidate[:, 0], baseline[:, 0])
    assert candidate[0, 1] < baseline[0, 1] < 0.0
    assert abs(candidate[0, 1]) < 0.8
    expected_delta = (0.4 - 0.25) * (requested_wz[0] - state[0, 5])
    np.testing.assert_allclose(candidate[0, 1] - baseline[0, 1], expected_delta)


def test_case74_second_yaw_rate_gain_candidate_remains_bounded() -> None:
    gain = np.array(
        [
            [-3.9542270864, -1.4129892407, 0.0, 0.0002045618, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0064842532, 0.0],
        ]
    )
    scenarios = (
        (0.0, 0.3324418843),
        (-0.1914618313, -0.3684858926),
        (0.127, 0.362),
    )
    for actual_wz, requested_wz in scenarios:
        state = np.zeros((1, len(LQR_STATE_NAMES)))
        state[0, 5] = actual_wz
        controller_state = np.zeros((1, 6))
        previous, _, _ = cascaded_lqr_action(
            state,
            np.zeros(1),
            np.array([requested_wz]),
            gain,
            controller_state,
            control_dt=0.02,
            config=cascaded_lqr_config("structural_robust_v1", wz_kp=0.4),
        )
        candidate, _, _ = cascaded_lqr_action(
            state,
            np.zeros(1),
            np.array([requested_wz]),
            gain,
            controller_state,
            control_dt=0.02,
            config=cascaded_lqr_config("structural_robust_v1", wz_kp=0.9),
        )
        np.testing.assert_allclose(candidate[:, 0], previous[:, 0])
        expected_delta = (0.9 - 0.4) * (requested_wz - actual_wz)
        np.testing.assert_allclose(candidate[0, 1] - previous[0, 1], expected_delta)
        assert np.sign(candidate[0, 1] - previous[0, 1]) == np.sign(
            requested_wz - actual_wz
        )
        assert abs(candidate[0, 1]) < 0.8


def test_case74_final_yaw_rate_gain_step_remains_bounded() -> None:
    gain = np.array(
        [
            [-3.9542270864, -1.4129892407, 0.0, 0.0002045618, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0064842532, 0.0],
        ]
    )
    scenarios = (
        (0.0, 0.3324418843),
        (-0.1914618313, -0.3684858926),
        (0.127, 0.362),
    )
    for actual_wz, requested_wz in scenarios:
        state = np.zeros((1, len(LQR_STATE_NAMES)))
        state[0, 5] = actual_wz
        controller_state = np.zeros((1, 6))
        previous, _, _ = cascaded_lqr_action(
            state,
            np.zeros(1),
            np.array([requested_wz]),
            gain,
            controller_state,
            control_dt=0.02,
            config=cascaded_lqr_config("structural_robust_v1", wz_kp=0.9),
        )
        candidate, _, _ = cascaded_lqr_action(
            state,
            np.zeros(1),
            np.array([requested_wz]),
            gain,
            controller_state,
            control_dt=0.02,
            config=cascaded_lqr_config("structural_robust_v1", wz_kp=1.05),
        )
        np.testing.assert_allclose(candidate[:, 0], previous[:, 0])
        expected_delta = (1.05 - 0.9) * (requested_wz - actual_wz)
        np.testing.assert_allclose(candidate[0, 1] - previous[0, 1], expected_delta)
        assert np.sign(candidate[0, 1] - previous[0, 1]) == np.sign(
            requested_wz - actual_wz
        )
        assert abs(candidate[0, 1]) < 0.8


def test_cascaded_lqr_accepts_measured_pitch_bias_override() -> None:
    _, controller_state, diagnostics = cascaded_lqr_action(
        np.zeros((1, len(LQR_STATE_NAMES))),
        np.zeros(1),
        np.zeros(1),
        np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES))),
        np.zeros((1, 6)),
        control_dt=0.02,
        config=CascadedLQRConfig(),
        pitch_bias_override_rad=np.array([-0.05]),
    )
    np.testing.assert_allclose(diagnostics["applied_pitch_bias"], [-0.05])
    np.testing.assert_allclose(controller_state[:, 2], [-0.05])
    np.testing.assert_allclose(controller_state[:, 3], [1.0])


def test_total_pitch_limit_restores_symmetric_physical_headroom() -> None:
    gain = np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES)))
    states = np.zeros((2, len(LQR_STATE_NAMES)))
    bias = np.full(2, np.radians(1.65))
    _, _, diagnostics = cascaded_lqr_action(
        states,
        np.array([-1.0, 1.0]),
        np.zeros(2),
        gain,
        np.zeros((2, 6)),
        control_dt=0.02,
        config=CascadedLQRConfig(limit_total_pitch_reference=True),
        pitch_bias_override_rad=bias,
    )

    np.testing.assert_allclose(
        diagnostics["total_pitch_reference"],
        np.radians([-6.0, 6.0]),
    )
    np.testing.assert_allclose(
        diagnostics["pitch_reference"],
        np.radians([-7.65, 4.35]),
    )
    assert np.all(diagnostics["total_pitch_reference_limit_enabled"])


def test_total_pitch_limit_is_default_off_and_zero_bias_compatible() -> None:
    gain = np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES)))
    states = np.zeros((2, len(LQR_STATE_NAMES)))
    kwargs = dict(
        states=states,
        vx_ref=np.array([-1.0, 1.0]),
        wz_ref=np.zeros(2),
        gain=gain,
        integrals=np.zeros((2, 6)),
        control_dt=0.02,
        pitch_bias_override_rad=np.zeros(2),
    )
    legacy = cascaded_lqr_action(config=CascadedLQRConfig(), **kwargs)[2]
    candidate = cascaded_lqr_action(
        config=CascadedLQRConfig(limit_total_pitch_reference=True), **kwargs
    )[2]

    np.testing.assert_allclose(
        candidate["pitch_reference"], legacy["pitch_reference"]
    )
    np.testing.assert_allclose(
        candidate["total_pitch_reference"], legacy["total_pitch_reference"]
    )
    assert not np.any(legacy["total_pitch_reference_limit_enabled"])


def test_total_pitch_limit_does_not_change_legacy_nonzero_bias_behavior() -> None:
    gain = np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES)))
    states = np.zeros((2, len(LQR_STATE_NAMES)))
    bias = np.full(2, np.radians(1.65))
    legacy = cascaded_lqr_action(
        states,
        np.array([-1.0, 1.0]),
        np.zeros(2),
        gain,
        np.zeros((2, 6)),
        control_dt=0.02,
        config=CascadedLQRConfig(),
        pitch_bias_override_rad=bias,
    )[2]

    np.testing.assert_allclose(
        legacy["pitch_reference"], np.radians([-6.0, 6.0])
    )
    np.testing.assert_allclose(
        legacy["total_pitch_reference"], np.radians([-4.35, 7.65])
    )
    assert not np.any(legacy["total_pitch_reference_limit_enabled"])


def test_cascaded_lqr_defaults_match_selected_tracking_gate() -> None:
    config = CascadedLQRConfig()
    assert config.wheel_radius_m == 0.1016
    assert config.wheel_track_m == 0.620
    assert config.vx_kp == 0.6
    assert config.wz_kp == 0.25
    assert config.wz_feedforward == 0.6
    assert config.vx_ki == 0.05
    assert config.wz_ki == 0.10
    assert config.wz_integral_limit == 2.0
    assert not config.governor_include_opposing_bias
    assert config.wheel_difference_kp == 0.0
    assert not config.path_progress_governor_enabled
    assert config.governor_minimum_progress_scale == 0.75
    assert np.isclose(np.degrees(config.pitch_reference_limit_rad), 6.0)
    assert config.action_limit == 0.8


def test_structural_robust_profile_is_explicit_and_default_safe() -> None:
    default = cascaded_lqr_config()
    robust = cascaded_lqr_config("structural_robust_v1")
    assert default == CascadedLQRConfig()
    assert robust.vx_ki == 0.075
    assert robust.vx_integral_limit == 0.7
    assert robust.path_progress_governor_enabled
    assert robust.governor_include_opposing_bias


def test_diagnostic_plant_variations_are_bounded_and_unique() -> None:
    variations = diagnostic_plant_variations()
    assert len(variations) == 16
    assert len({item.name for item in variations}) == len(variations)
    assert variations[0] == PlantVariation("nominal")
    assert any(item.mass_scale == 1.25 for item in variations)
    assert any(
        item.name == "corner_heavy_high_com_low_grip_low_torque_delay"
        and item.mass_scale == 1.15
        for item in variations
    )
    assert max(item.action_delay_steps for item in variations) == 4
    assert min(item.torque_scale for item in variations) == 0.8
    assert all(
        item.static_friction is None
        or item.dynamic_friction <= item.static_friction
        for item in variations
    )


def test_provisional_plant_variations_match_guessed_operating_envelope() -> None:
    variations = provisional_plant_variations()
    assert len(variations) == 14
    assert len({item.name for item in variations}) == len(variations)
    assert min(item.mass_scale for item in variations) == 0.95
    assert max(item.mass_scale for item in variations) == 1.05
    assert min(item.com_offset_x_m for item in variations) == -0.02
    assert max(item.com_offset_x_m for item in variations) == 0.02
    assert min(item.com_offset_z_m for item in variations) == -0.03
    assert max(item.com_offset_z_m for item in variations) == 0.03
    assert min(item.inertia_scale for item in variations) == 0.85
    assert max(item.inertia_scale for item in variations) == 1.15
    assert min(item.torque_scale for item in variations) == 0.9
    assert max(item.action_delay_steps for item in variations) == 2


def test_plant_variation_rejects_nonphysical_values() -> None:
    for kwargs in (
        {"mass_scale": 0.0},
        {"target_total_mass_kg": -1.0},
        {"mass_scale": 1.1, "target_total_mass_kg": 40.0},
        {"inertia_scale": -1.0},
        {"static_friction": 0.5, "dynamic_friction": 0.6},
        {"static_friction": 0.5},
        {"torque_scale": 1.1},
        {"action_delay_steps": -1},
    ):
        try:
            PlantVariation("invalid", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid variation: {kwargs}")
