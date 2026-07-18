from dataclasses import replace
import importlib.util
from pathlib import Path
import string

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_playback import RiserPlaybackPlan
from rl_platform.tasks.two_wheel_balance.riser_smoothed_plan import (
    MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD,
    PREVIEW_CONFIGURATIONS,
    RECOVERY_CONFIGURATIONS,
    SMOOTHING_SIGMA_CANDIDATES,
    SmoothedPlanResult,
    batch_unicycle_recovery_seed_eligible,
    derived_reset_yaw_rad,
    retime_smoothed_plan_from_demands,
    smooth_source_positions,
    smoothed_path_metrics,
    transition_metrics,
)


def _plan(time_s: np.ndarray) -> RiserPlaybackPlan:
    count = len(time_s)
    yaw = np.linspace(0.0, 0.24, count)
    base = np.column_stack((np.linspace(0.0, 0.3, count), np.zeros(count), yaw))
    proxy = np.column_stack(
        (np.linspace(0.0, 0.18, count), np.zeros(count), np.zeros(count))
    )
    dt = np.diff(time_s)
    midpoint = 0.5 * (yaw[:-1] + yaw[1:])
    delta = np.diff(base[:, :2], axis=0)
    forward = (
        np.cos(midpoint) * delta[:, 0] + np.sin(midpoint) * delta[:, 1]
    ) / dt
    return RiserPlaybackPlan(
        case=1,
        time_s=time_s,
        target_position_world_m=np.column_stack(
            (np.linspace(0.0, 0.3, count), np.zeros(count), np.ones(count))
        ),
        target_semantic_dfr_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        base_xy_yaw=base,
        riser_q=np.linspace(0.2, 0.3, count),
        proxy_gimbal_q=proxy,
        feedforward_v_wz=np.column_stack((forward, np.diff(yaw) / dt)),
        feedforward_riser_velocity=np.diff(np.linspace(0.2, 0.3, count)) / dt,
        feedforward_proxy_velocity=np.diff(proxy, axis=0) / dt[:, None],
        vertical_shift_m=0.0,
        planning_strategy="smoothed_preview_0.05m_g2.75",
        source_time_s=np.array([0.0, 0.5, 1.0]),
    )


def test_smoothing_preserves_endpoints_and_separate_source_array() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.8],
            [0.1, 0.08, 0.9],
            [0.2, -0.08, 1.0],
            [0.3, 0.0, 1.1],
        ]
    )
    original = source.copy()
    smoothed = smooth_source_positions(source, 1.5)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(smoothed[0], source[0])
    np.testing.assert_array_equal(smoothed[-1], source[-1])
    np.testing.assert_array_equal(smoothed[:, 2], source[:, 2])
    assert not np.array_equal(smoothed[1:-1, :2], source[1:-1, :2])

    blended = smooth_source_positions(source, 1.5, 0.45)
    np.testing.assert_allclose(blended, source + 0.45 * (smoothed - source))


def test_reset_yaw_is_derived_from_immutable_source_direction() -> None:
    source = np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 1.0]])
    assert derived_reset_yaw_rad(source, 0.3, "source") == pytest.approx(0.3)
    assert derived_reset_yaw_rad(source, 0.3, "forward_path") == pytest.approx(0.0)
    assert derived_reset_yaw_rad(source, 0.3, "reverse_path") == pytest.approx(
        np.pi
    )
    with pytest.raises(ValueError, match="invalid reset yaw mode"):
        derived_reset_yaw_rad(source, 0.3, "sideways")


def test_path_metrics_measure_length_drift_and_polyline_deviation() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    smoothed = source.copy()
    smoothed[1, 1] = 0.1
    metrics = smoothed_path_metrics(source, smoothed)
    assert metrics["source_path_length_m"] == pytest.approx(2.0)
    assert metrics["smoothed_path_length_m"] > 2.0
    assert metrics["source_polyline_deviation_max_m"] == pytest.approx(0.1)
    assert metrics["opposed_segment_direction_count"] == 0
    assert metrics["start_position_error_m"] == 0.0
    assert metrics["final_position_error_m"] == 0.0


def test_path_metrics_detect_opposed_local_motion_direction() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    opposed = np.array(
        [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    metrics = smoothed_path_metrics(source, opposed)
    assert metrics["opposed_segment_direction_count"] == 1
    assert metrics["minimum_segment_direction_cosine"] == pytest.approx(-1.0)


def test_demand_retimer_discards_provisional_dwell_and_respects_source_duration() -> None:
    provisional = _plan(np.array([0.0, 10.0, 20.0]))
    retimed = retime_smoothed_plan_from_demands(provisional, source_duration_s=1.0)
    assert retimed.time_s[-1] < provisional.time_s[-1]
    assert retimed.time_s[-1] >= 1.0
    assert np.max(np.abs(retimed.feedforward_v_wz[:, 0])) <= 0.4 + 1e-12
    assert np.max(np.abs(retimed.feedforward_v_wz[:, 1])) <= 0.4 + 1e-12
    assert np.max(np.abs(retimed.feedforward_proxy_velocity)) <= np.deg2rad(24.0) + 1e-12


def test_transition_metrics_reject_pre_densification_branch_jump() -> None:
    plan = _plan(np.array([0.0, 0.5, 1.0]))
    metrics = transition_metrics(plan)
    assert metrics["maximum_pre_densification_base_branch_step_rad"] < 0.25
    bad_proxy = plan.proxy_gimbal_q.copy()
    bad_proxy[1, 0] = MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD + 0.01
    bad = RiserPlaybackPlan(**{**plan.__dict__, "proxy_gimbal_q": bad_proxy})
    assert (
        transition_metrics(bad)[
            "maximum_pre_densification_proxy_branch_step_rad"
        ]
        > MAXIMUM_PRE_DENSIFICATION_BRANCH_STEP_RAD
    )


def test_recovery_candidates_are_appended_without_reordering_baseline() -> None:
    assert SMOOTHING_SIGMA_CANDIDATES[:4] == (0.0, 4.0, 8.0, 12.0)
    assert SMOOTHING_SIGMA_CANDIDATES[-1] == 16.0
    assert PREVIEW_CONFIGURATIONS[:4] == (
        (0.05, 2.75),
        (0.10, 2.75),
        (0.15, 2.75),
        (0.25, 2.75),
    )
    assert PREVIEW_CONFIGURATIONS[-2:] == ((0.40, 1.00), (0.50, 1.00))
    assert RECOVERY_CONFIGURATIONS[-2] == (
        16.0,
        0.45,
        0.65,
        1.00,
        "forward_path",
    )
    assert RECOVERY_CONFIGURATIONS[-1] == (
        64.0,
        0.1276273593606172,
        0.90,
        1.00,
        "forward_path",
    )


@pytest.mark.parametrize(
    "strategy",
    (
        "smoothed_preview_0.40m_g1.00",
        "smoothed_preview_0.50m_g1.00",
        "smoothed_preview_0.65m_g1.00",
        "smoothed_preview_0.90m_g1.00",
        "smoothed_batch_unicycle_v1",
    ),
)
def test_recovery_strategy_names_validate(strategy: str) -> None:
    plan = _plan(np.array([0.0, 0.5, 1.0]))
    RiserPlaybackPlan(**{**plan.__dict__, "planning_strategy": strategy}).validate()


def test_batch_recovery_seed_is_fail_closed_to_position_p95_only() -> None:
    plan = _plan(np.array([0.0, 0.5, 1.0]))
    result = SmoothedPlanResult(
        plan=plan,
        smoothed_position_source_frame_m=plan.target_position_world_m,
        smoothing_sigma_samples=64.0,
        smoothing_blend_factor=0.1276273593606172,
        lookahead_distance_m=0.9,
        heading_gain=1.0,
        reset_yaw_mode="forward_path",
        reset_yaw_rad=0.0,
        path_metrics={},
        transition_metrics={},
        kinematic_metrics={"position_error_p95_m": 0.18},
        kinematic_checks={
            "position_p95_bounded": False,
            "position_max_bounded": True,
        },
        checks={"source_integrity": True},
        attempts=(),
    )
    assert batch_unicycle_recovery_seed_eligible(result, source_duration_s=0.5)
    assert not batch_unicycle_recovery_seed_eligible(
        replace(result, checks={"source_integrity": False}),
        source_duration_s=0.5,
    )
    assert not batch_unicycle_recovery_seed_eligible(
        replace(
            result,
            kinematic_checks={
                "position_p95_bounded": False,
                "position_max_bounded": False,
            },
        ),
        source_duration_s=0.5,
    )
    assert not batch_unicycle_recovery_seed_eligible(
        replace(result, kinematic_metrics={"position_error_p95_m": 0.21}),
        source_duration_s=0.5,
    )
    assert not batch_unicycle_recovery_seed_eligible(
        result, source_duration_s=0.49
    )
    with pytest.raises(ValueError, match="source duration"):
        batch_unicycle_recovery_seed_eligible(result, source_duration_s=0.0)


def test_exporter_defaults_to_bounded_case_order_and_training_closed() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/export_riser_smoothed_plans.py"
    ).read_text(encoding="utf-8")
    assert 'default=parse_cases("74,77,52")' in source
    assert 'if value.strip().lower() == "all"' in source
    assert '"code_commit": code_commit' in source
    assert '"minimum_pass_count_met": minimum_pass_count_met' in source
    assert '"selected_smoothing_blend_factor"' in source
    assert '"selected_reset_yaw_mode"' in source
    assert '"isaac_started": False' in source
    assert '"residual_capture_started": False' in source
    assert '"bc_started": False' in source
    assert '"ppo_started": False' in source
    assert '"differential_session_work_started": False' in source


def test_exporter_resolves_linked_worktree_git_identity() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/export_riser_smoothed_plans.py"
    )
    spec = importlib.util.spec_from_file_location("riser_smoothed_export", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module._windows_path_from_gitdir("/mnt/g/wSpace/repo/.git/worktrees/x")
        == "G:/wSpace/repo/.git/worktrees/x"
    )
    head = module._git_output("rev-parse", "HEAD")
    assert len(head) == 40
    assert set(head) <= set(string.hexdigits)
    if (module.PROJECT_ROOT / ".git").is_file():
        assert "--git-dir" in module._git_command()
