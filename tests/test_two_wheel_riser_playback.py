from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_playback import (
    PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S,
    RiserPlaybackPlan,
    interpolate_riser_playback_plan,
    load_riser_playback_plan,
    phase_scaled_feedforward,
    save_riser_playback_plan,
)


def test_playback_planning_uses_accepted_all79_yaw_cap() -> None:
    assert PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S == 0.25


def test_fixed_path_planner_respects_bounded_per_case_yaw_override() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/rl_platform/tasks/two_wheel_balance/riser_rs4_reference.py"
    ).read_text(encoding="utf-8")
    call = source.split("fixed = _plan_fixed_path_rs4_riser_reference(", 1)[1].split(
        ")\n", 1
    )[0]
    assert "maximum_base_yaw_rate_rad_s=maximum_base_yaw_rate_rad_s" in call
    assert "min(" not in call


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _plan() -> RiserPlaybackPlan:
    return RiserPlaybackPlan(
        case=1,
        time_s=np.array([0.0, 0.1, 0.2]),
        target_position_world_m=np.array(
            [[0.0, 0.0, 0.6], [0.01, 0.0, 0.7], [0.02, 0.0, 0.8]]
        ),
        target_semantic_dfr_quat_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        base_xy_yaw=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        riser_q=np.array([0.0, 0.1, 0.2]),
        proxy_gimbal_q=np.deg2rad(
            np.array([[0.0, 0.0, 179.0], [0.0, 0.0, 181.0], [0.0, 0.0, 183.0]])
        ),
        feedforward_v_wz=np.array([[0.1, 0.0], [0.1, 0.0]]),
        feedforward_riser_velocity=np.array([1.0, 1.0]),
        feedforward_proxy_velocity=np.deg2rad(
            np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 20.0]])
        ),
        vertical_shift_m=0.0,
        planning_strategy="fixed_path",
    )


def test_playback_plan_roundtrip_and_interpolation(tmp_path: Path) -> None:
    path = tmp_path / "case.npz"
    save_riser_playback_plan(path, _plan())
    loaded = load_riser_playback_plan(path)
    sample = interpolate_riser_playback_plan(loaded, 0.15)
    assert loaded.case == 1
    np.testing.assert_allclose(sample.base_xy_yaw, [0.015, 0.0, 0.0])
    assert sample.riser_q == pytest.approx(0.15)
    assert np.rad2deg(sample.proxy_gimbal_q[2]) == pytest.approx(182.0)
    assert sample.feedforward_v_mps == pytest.approx(0.1)


def test_playback_roundtrip_preserves_unequal_source_and_execution_clocks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.npz"
    plan = RiserPlaybackPlan(
        **{**_plan().__dict__, "source_time_s": np.array([0.0, 0.04, 0.08])}
    )
    save_riser_playback_plan(path, plan)
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["execution_time_s"], [0.0, 0.1, 0.2])
        np.testing.assert_array_equal(arrays["source_time_s"], [0.0, 0.04, 0.08])
    loaded = load_riser_playback_plan(path)
    np.testing.assert_array_equal(loaded.source_time_s, [0.0, 0.04, 0.08])
    np.testing.assert_array_equal(loaded.time_s, [0.0, 0.1, 0.2])
    assert loaded.source_time_s[-1] != loaded.time_s[-1]


def test_phase_governor_scales_every_playback_derivative() -> None:
    sample = interpolate_riser_playback_plan(_plan(), 0.15)
    velocity, yaw_rate, riser_rate, proxy_rate = phase_scaled_feedforward(
        sample, 0.25
    )
    assert velocity == pytest.approx(0.025)
    assert yaw_rate == pytest.approx(0.0)
    assert riser_rate == pytest.approx(0.25)
    np.testing.assert_allclose(proxy_rate, np.deg2rad([0.0, 0.0, 5.0]))

    with pytest.raises(ValueError, match="progress scale"):
        phase_scaled_feedforward(sample, 1.01)


def test_playback_plan_rejects_wrapped_yaw_servo_jump() -> None:
    plan = _plan()
    wrapped = plan.proxy_gimbal_q.copy()
    wrapped[1, 2] = np.deg2rad(-179.0)
    bad = RiserPlaybackPlan(**{**plan.__dict__, "proxy_gimbal_q": wrapped})
    with pytest.raises(ValueError, match="continuous_proxy_yaw"):
        bad.validate()


def test_playback_commands_semantic_proxy_position_without_motor_velocity() -> None:
    source = (
        PROJECT_ROOT
        / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
    ).read_text(encoding="utf-8")

    assert (
        '"hardware_proxy_command_contract": "semantic_attitude_position_only"'
        in source
    )
    assert '"rate_audited_ideal_state_adapter"' in source
    assert "robot.set_joint_position_target(proxy_target, joint_ids=proxy_ids)" in source
    assert "robot.write_joint_state_to_sim(" in source
    assert "nearest_equivalent_angle(" in source
    assert "continuous_joint_error(" in source
    assert "proxy_sim_command" in source
    assert '"proxy_unwrapped_semantic_target_deg"' in source
    assert "np.abs(actual_proxy - proxy_command)" not in source
    assert "np.abs(actual_proxy - sample.proxy_gimbal_q)" not in source
    assert "set_joint_velocity_target(proxy_velocity_target" not in source
    assert 'parser.add_argument("--video-fps", type=int, default=200)' in source
    assert '"tracking_profile": "riser_recovery_direction_v4"' in source
    assert '"tracking_direction_blend_speed_mps"' in source
    assert '"tracking_direction_recovery_error_range_m"' in source
    assert '"motion_direction": base_tracking_diagnostics[' in source
    assert '"direction_recovery_blend": base_tracking_diagnostics[' in source
    assert '"feedforward_direction": base_tracking_diagnostics[' in source
    assert '"cross_track_error_m": base_tracking_diagnostics[' in source
    assert '"residual_teacher_unclipped"' in source
    assert "float(np.max(np.abs(teacher_residual_values))) < 1.0 - 1e-6" in source
    assert '"source_duration_s": source_duration_s' in source
    assert '"execution_duration_s": execution_duration_s' in source
    assert "phase_time_s >= execution_duration_s" in source
    assert "phase_time_s >= source_duration_s" not in source
