import json
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_playback import (
    PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S,
    RiserPlaybackPlan,
    interpolate_riser_initialization,
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


def test_playback_plan_accepts_geometry_preserving_explicit_preview() -> None:
    plan = RiserPlaybackPlan(
        **{**_plan().__dict__, "planning_strategy": "smoothed_explicit_preview_v1"}
    )
    plan.validate()


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


def test_playback_roundtrip_and_interpolate_explicit_initialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.npz"
    initialization_time = np.array([0.0, 0.5, 1.0])
    initialization_state = np.array(
        [
            [-0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 178.0],
            [-0.05, 0.0, 0.0, -0.05, 0.0, 0.0, 178.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 179.0],
        ],
        dtype=np.float64,
    )
    initialization_state[:, 6] = np.deg2rad(initialization_state[:, 6])
    plan = RiserPlaybackPlan(
        **{
            **_plan().__dict__,
            "initialization_time_s": initialization_time,
            "initialization_state": initialization_state,
        }
    )
    save_riser_playback_plan(path, plan)
    loaded = load_riser_playback_plan(path)
    sample = interpolate_riser_initialization(loaded, 0.25)
    np.testing.assert_allclose(sample.base_xy_yaw, [-0.075, 0.0, 0.0])
    assert sample.riser_q == pytest.approx(-0.075)
    assert sample.feedforward_v_mps == pytest.approx(0.1)
    assert np.rad2deg(sample.proxy_gimbal_q[2]) == pytest.approx(178.25)
    np.testing.assert_array_equal(loaded.source_time_s, _plan().time_s)
    np.testing.assert_array_equal(loaded.time_s, _plan().time_s)


def test_playback_rejects_initialization_that_does_not_join_execution() -> None:
    plan = _plan()
    bad = RiserPlaybackPlan(
        **{
            **plan.__dict__,
            "initialization_time_s": np.array([0.0, 1.0]),
            "initialization_state": np.zeros((2, 7)),
        }
    )
    with pytest.raises(ValueError, match="initialization"):
        bad.validate()


def test_playback_rejects_half_specified_initialization() -> None:
    plan = RiserPlaybackPlan(
        **{
            **_plan().__dict__,
            "initialization_time_s": np.array([0.0, 1.0]),
        }
    )
    with pytest.raises(ValueError, match="initialization"):
        plan.validate()


def _rewrite_npz_metadata(
    path: Path, *, remove: tuple[str, ...] = (), **updates: object
) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    for name in remove:
        metadata.pop(name, None)
    metadata.update(updates)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def test_playback_loader_accepts_explicit_smoothed_plan_schema(tmp_path: Path) -> None:
    path = tmp_path / "case.npz"
    save_riser_playback_plan(path, _plan())
    _rewrite_npz_metadata(
        path,
        remove=("vertical_shift_m", "planning_strategy"),
        schema="cinebotrl_two_wheel_riser_smoothed_plan_v1",
        smoothed_target={
            "schema": "derived_smoothed_target_v1",
            "vertical_shift_m": 0.0,
            "planning_strategy": "smoothed_preview_0.05m_g2.75",
        },
    )
    loaded = load_riser_playback_plan(path)
    assert loaded.case == 1
    assert loaded.vertical_shift_m == 0.0
    assert loaded.planning_strategy == "smoothed_preview_0.05m_g2.75"
    np.testing.assert_array_equal(loaded.time_s, [0.0, 0.1, 0.2])


def test_playback_loader_rejects_malformed_smoothed_metadata(tmp_path: Path) -> None:
    path = tmp_path / "case.npz"
    save_riser_playback_plan(path, _plan())
    _rewrite_npz_metadata(
        path,
        schema="cinebotrl_two_wheel_riser_smoothed_plan_v1",
        smoothed_target={"schema": "unreviewed_target_v1"},
    )
    with pytest.raises(ValueError, match="invalid smoothed target metadata"):
        load_riser_playback_plan(path)


def test_playback_loader_rejects_unknown_schema_or_ambiguous_clock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.npz"
    save_riser_playback_plan(path, _plan())
    _rewrite_npz_metadata(path, schema="unreviewed_plan_v1")
    with pytest.raises(ValueError, match="unexpected playback schema"):
        load_riser_playback_plan(path)

    save_riser_playback_plan(path, _plan())
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    arrays["execution_time_s"] = np.array([0.0, 0.11, 0.22])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="ambiguous execution time aliases"):
        load_riser_playback_plan(path)


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


def test_runtime_keeps_initialization_separate_from_source_evidence() -> None:
    source = (
        PROJECT_ROOT
        / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
    ).read_text(encoding="utf-8")
    initialization = source.split(
        "if not initialization_completed:", 1
    )[1].split("initialization_source_metrics_clean", 1)[0]
    assert "interpolate_riser_initialization" in initialization
    assert "cascaded_lqr_action" in initialization
    assert "controller_state" in initialization
    assert "position_errors.append" not in initialization
    assert "raw_residual_commands.append" not in initialization
    assert "dataset_observations.append" not in initialization
    assert '"initialization_scored_as_source_tracking": False' in source
    assert '"initialization_source_metric_samples": 0' in source
    assert '"initialization_residual_label_samples": 0' in source
    assert '"initialization_riser_thermal_force_observed"' in source
    assert "initialization_steps + completed_steps" in source
    assert "(initialization_step + 1) / POLICY_HZ" in source


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
    pre_app_source = source.split("app = AppLauncher(args).app", 1)[0]

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
    assert '"riser_recovery_direction_v4"' in source
    assert '"--enable-camera-lever-arm-compensation"' in source
    assert '"--camera-lever-arm-compensation-gain"' in source
    assert '"--maximum-camera-lever-arm-correction-m"' in source
    assert '"--enable-camera-error-recovery-governor"' in source
    assert '"--camera-recovery-error-start-m"' in source
    assert '"--camera-recovery-error-full-m"' in source
    assert '"--minimum-camera-recovery-scale"' in source
    assert '"--tracking-minimum-progress-scale"' in source
    assert '"--tracking-maximum-linear-velocity-mps"' in source
    assert '"--limit-total-pitch-reference"' in source
    assert '"limit_total_pitch_reference"' in source
    assert '"total_pitch_reference_rad"' in source
    assert "zero_progress_hold_velocity_cap_total_pitch_limit_v1" in source
    assert '"total_pitch_reference_limit_enabled"' in source
    assert '"total_pitch_reference_limit_rad"' in source
    assert '"maximum_linear_velocity_mps"' in source
    assert "0.0 < args.tracking_maximum_linear_velocity_mps <= 0.4" in pre_app_source
    assert '"tracking_recovery_velocity_cap_enabled"' in source
    assert "zero_progress_hold_velocity_cap_v1" in source
    assert "math.isfinite(args.camera_lever_arm_compensation_gain)" in pre_app_source
    assert "math.isfinite(args.maximum_camera_lever_arm_correction_m)" in pre_app_source
    assert "bounded_camera_lever_arm_base_target(" in source
    assert "bounded_camera_recovery_progress_scale(" in source
    assert '"camera_recovery_governor_contract"' in source
    assert '"camera_recovery_telemetry_observed"' in source
    assert '"camera_recovery_activation_ratio"' in source
    assert '"phase_governor_contract"' in source
    assert '"minimum_progress_scale"' in source
    assert "summarize_progress_hold(" in source
    assert '"camera_lever_arm_compensation_contract"' in source
    assert '"camera_lever_arm_compensation_enabled"' in source
    assert '"camera_lever_arm_telemetry_observed"' in source
    assert "controller_evidence_checks" in source
    assert '"controller_evidence_passed"' in source
    assert '"camera_lever_arm_correction_max_m"' in source
    assert '"camera_lever_arm_raw_correction_max_m"' in source
    assert '"camera_lever_arm_correction_saturation_ratio"' in source
    assert '"commanded_base_xy_yaw"' in source
    assert '"camera_lever_arm_compensation_enabled"' in source.split(
        "def write_runtime_failure", 1
    )[1]
    assert '"tracking_direction_blend_speed_mps"' in source
    assert '"tracking_direction_recovery_error_range_m"' in source
    assert "RiserMotorThermalMonitor" in source
    assert "riser_thermal_monitor.step(riser_effort, 1.0 / POLICY_HZ)" in source
    assert '"riser_thermal_force_observed"' in source
    assert '"riser_thermal_load_bounded"' in source
    assert '"riser_peak_force_bounded"' in source
    assert '"riser_thermal_force_contract": RISER_THERMAL_FORCE_CONTRACT' in source
    assert '"motion_direction": base_tracking_diagnostics[' in source
    assert '"direction_recovery_blend": base_tracking_diagnostics[' in source
    assert '"feedforward_direction": base_tracking_diagnostics[' in source
    assert '"cross_track_error_m": base_tracking_diagnostics[' in source
    assert '"residual_teacher_unclipped"' in source
    assert "float(np.max(np.abs(teacher_residual_values))) < 1.0 - 1e-6" in source
    assert '"source_duration_s": source_duration_s' in source
    assert "RecoveryTelemetryAccumulator" in source
    assert "recovery_telemetry.step(" in source
    assert '"recovery_telemetry": recovery_telemetry_summary' in source
    assert '"recovery_telemetry_observed"' in source
    assert '"execution_duration_s": execution_duration_s' in source
    assert '"maximum_duration_scale": args.maximum_duration_scale' in source
    assert (
        '"maximum_runtime_s": execution_duration_s * args.maximum_duration_scale'
        in source
    )
    assert (
        '"completion_horizon_contract": "bounded_execution_duration_scale_v1"'
        in source
    )
    assert "LOOKAHEAD_HORIZONS_S" in source
    assert "phase_time_s + horizon_s" in source
    assert "lookahead_base_xy_yaw=np.asarray(" in source
    assert "lookahead_camera_position_world_m=np.asarray(" in source
    assert "lookahead_camera_quat_wxyz=np.asarray(" in source
    assert "lookahead_feedforward_v_wz_riser=lookahead_feedforward" in source
    assert "phase_time_s >= execution_duration_s" in source
    assert "phase_time_s >= source_duration_s" not in source
