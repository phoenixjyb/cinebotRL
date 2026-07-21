import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_playback import RiserPlaybackPlan
from scripts.two_wheel_balance.rank_riser_unused_plan_coverage import (
    build_plan_command_signature,
    select_unused_candidate,
)


def _ranked(case: int, score: float) -> dict[str, object]:
    return {"case": case, "score": score}


def test_unused_candidate_requires_material_improvement() -> None:
    selected, ratio = select_unused_candidate(
        [_ranked(18, 1.0)],
        [_ranked(50, 0.75)],
    )
    assert selected == [50]
    assert ratio == 0.75


def test_unused_candidate_fails_closed_at_insufficient_improvement() -> None:
    selected, ratio = select_unused_candidate(
        [_ranked(18, 1.0)],
        [_ranked(50, 0.81)],
    )
    assert selected == []
    assert ratio == 0.81


def test_unused_candidate_selection_is_single_case_only() -> None:
    selected, _ = select_unused_candidate(
        [_ranked(18, 1.0), _ranked(30, 1.1)],
        [_ranked(50, 0.7), _ranked(49, 0.71)],
    )
    assert selected == [50]


def test_unused_candidate_requires_both_pools() -> None:
    try:
        select_unused_candidate([], [_ranked(50, 0.7)])
    except ValueError as error:
        assert "both comparison pools" in str(error)
    else:
        raise AssertionError("empty existing pool was accepted")


def test_plan_command_signature_contains_current_and_three_lookaheads() -> None:
    plan = RiserPlaybackPlan(
        case=1,
        time_s=np.array([0.0, 1.0]),
        target_position_world_m=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.5]]),
        target_semantic_dfr_quat_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        base_xy_yaw=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        riser_q=np.array([0.4, 0.9]),
        proxy_gimbal_q=np.zeros((2, 3)),
        feedforward_v_wz=np.array([[1.0, 0.0]]),
        feedforward_riser_velocity=np.array([0.5]),
        feedforward_proxy_velocity=np.zeros((1, 3)),
        vertical_shift_m=0.0,
        planning_strategy="fixed_path",
    )
    plan.validate()
    signature = build_plan_command_signature(plan, np.array([0.0, 0.5]))
    assert signature.shape == (2, 42)
    assert np.isfinite(signature).all()
    assert signature[0, :3].tolist() == [1.0, 0.0, 0.5]
    assert signature[0, 3] == 0.25
    assert signature[0, 8] == 0.125
