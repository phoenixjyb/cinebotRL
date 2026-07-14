"""Pure tests for the two-wheel action and observation contract."""

import numpy as np

from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    OBSERVATION_NAMES,
    BalanceContract,
    CascadedLQRConfig,
    PlantVariation,
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
    assert config.wheel_difference_kp == 0.0
    assert np.isclose(np.degrees(config.pitch_reference_limit_rad), 6.0)
    assert config.action_limit == 0.8


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
