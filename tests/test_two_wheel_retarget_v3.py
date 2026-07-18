from pathlib import Path
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/two_wheel_balance"))

from retarget_corrected_teacher_v3_nonholonomic import (  # noqa: E402
    BALANCE_PITCH_OUTPUT_TOLERANCE_DEG,
    BALANCE_PITCH_SOLVER_TOLERANCE_DEG,
    HOME_ARM,
    SEMANTIC_BRANCH_RANK_VARIANT_COUNT,
    SemanticBranchAlternative,
    append_gravity_aware_base_arm_recovery_seed,
    balance_pitch_optimization_margin_deg,
    bounded_gimbal_recovery_deltas,
    build_gravity_aware_arm_acquisition,
    gravity_aware_base_arm_recovery_seed,
    physical_gimbal_interpolation_error,
    physical_camera_rotation,
    select_semantic_branch_beam,
    semantic_branch_rank_key,
    select_acquisition_base_route,
    semantic_gimbal_reserve_margin_ratio,
    semantic_gimbal_reserve_search_max_scale,
    should_stop_semantic_retime_search,
    solve_full_pose_anchor,
    wrap_angle,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (  # noqa: E402
    UrdfPhysicalCameraKinematics,
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (  # noqa: E402
    UrdfPositionKinematics,
)


def _branch(
    marker: float,
    *,
    feasible: bool = True,
    local: float = 0.0,
    future: float = 0.0,
    lineage: tuple[int, ...] = (),
) -> SemanticBranchAlternative:
    return SemanticBranchAlternative(
        state=np.full(9, marker),
        previous_control=np.full(8, marker),
        hard_feasible=feasible,
        local_score=local,
        future_score=future,
        lineage=lineage,
        payload_index=int(marker),
    )


def test_semantic_branch_beam_keeps_future_reachable_alternative() -> None:
    greedy = _branch(0.0, local=0.1, future=10.0, lineage=(0,))
    reachable = _branch(1.0, local=0.2, future=0.1, lineage=(1,))

    selected = select_semantic_branch_beam([greedy, reachable], beam_width=2)

    assert [item.payload_index for item in selected] == [1, 0]
    second_step = select_semantic_branch_beam(
        [
            _branch(2.0, feasible=False, lineage=(0, 2)),
            _branch(3.0, local=0.3, future=0.0, lineage=(1, 3)),
        ],
        beam_width=2,
    )
    assert [item.payload_index for item in second_step] == [3]


def test_semantic_branch_beam_is_deterministic_and_deduplicates() -> None:
    first = _branch(1.0, local=0.2, future=0.1, lineage=(1,))
    duplicate = SemanticBranchAlternative(
        state=first.state + 1e-12,
        previous_control=first.previous_control.copy(),
        hard_feasible=True,
        local_score=0.1,
        future_score=0.1,
        lineage=(0,),
        payload_index=9,
    )
    other = _branch(2.0, local=0.4, future=0.2, lineage=(2,))

    selected = select_semantic_branch_beam(
        [other, first, duplicate], beam_width=3
    )

    assert len(selected) == 2
    assert selected[0].payload_index == 9
    assert selected[1].payload_index == 2


def test_semantic_branch_beam_rejects_bad_contract() -> None:
    with pytest.raises(ValueError, match="beam width"):
        select_semantic_branch_beam([], beam_width=0)
    with pytest.raises(ValueError, match="shape"):
        select_semantic_branch_beam(
            [
                SemanticBranchAlternative(
                    state=np.zeros(8),
                    previous_control=np.zeros(8),
                    hard_feasible=True,
                    local_score=0.0,
                    future_score=0.0,
                    lineage=(),
                    payload_index=0,
                )
            ],
            beam_width=1,
        )


def test_semantic_branch_rank_variants_preserve_feasibility_and_ordering() -> None:
    common = {
        "feasible": True,
        "attitude_error_deg": 0.01,
        "source_arm_error_rad": 0.2,
    }
    low_score = semantic_branch_rank_key(
        variant=0,
        baseline_score=1.0,
        position_error_m=0.04,
        gravity_effort_nm=29.0,
        stable_index=0,
        **common,
    )
    low_gravity = semantic_branch_rank_key(
        variant=1,
        baseline_score=2.0,
        position_error_m=0.045,
        gravity_effort_nm=28.0,
        stable_index=1,
        **common,
    )
    assert low_score < semantic_branch_rank_key(
        variant=0,
        baseline_score=2.0,
        position_error_m=0.01,
        gravity_effort_nm=20.0,
        stable_index=1,
        **common,
    )
    assert low_gravity < semantic_branch_rank_key(
        variant=1,
        baseline_score=1.0,
        position_error_m=0.01,
        gravity_effort_nm=29.0,
        stable_index=0,
        **common,
    )
    infeasible = semantic_branch_rank_key(
        variant=2,
        feasible=False,
        baseline_score=0.0,
        position_error_m=0.0,
        attitude_error_deg=0.0,
        gravity_effort_nm=0.0,
        source_arm_error_rad=0.0,
        stable_index=0,
    )
    feasible = semantic_branch_rank_key(
        variant=2,
        feasible=True,
        baseline_score=100.0,
        position_error_m=100.0,
        attitude_error_deg=100.0,
        gravity_effort_nm=100.0,
        source_arm_error_rad=100.0,
        stable_index=1,
    )
    assert feasible < infeasible
    with pytest.raises(ValueError, match="rank variant"):
        semantic_branch_rank_key(
            variant=SEMANTIC_BRANCH_RANK_VARIANT_COUNT,
            feasible=True,
            baseline_score=0.0,
            position_error_m=0.0,
            attitude_error_deg=0.0,
            gravity_effort_nm=0.0,
            source_arm_error_rad=0.0,
            stable_index=0,
        )


def test_balance_pitch_solver_and_output_tolerances_are_separate() -> None:
    assert BALANCE_PITCH_SOLVER_TOLERANCE_DEG == 1e-6
    assert BALANCE_PITCH_OUTPUT_TOLERANCE_DEG == 0.001
    assert balance_pitch_optimization_margin_deg(
        SimpleNamespace(camera_solve_root_model="balanced")
    ) == 0.01
    assert balance_pitch_optimization_margin_deg(
        SimpleNamespace(camera_solve_root_model="upright")
    ) == 0.0


def test_semantic_gimbal_reserve_does_not_relax_admission_margin() -> None:
    args = SimpleNamespace(
        minimum_semantic_gimbal_limit_margin_ratio=0.005,
        minimum_semantic_gimbal_reserve_margin_ratio=0.01,
    )
    assert semantic_gimbal_reserve_margin_ratio(args) == 0.01
    args.minimum_semantic_gimbal_reserve_margin_ratio = 0.004
    with pytest.raises(ValueError, match="between the admission margin"):
        semantic_gimbal_reserve_margin_ratio(args)


def test_semantic_gimbal_reserve_search_scale_is_bounded_separately() -> None:
    args = SimpleNamespace(maximum_semantic_gimbal_reserve_search_scale=24)
    assert semantic_gimbal_reserve_search_max_scale(args) == 24
    assert semantic_gimbal_reserve_search_max_scale(SimpleNamespace()) is None
    args.maximum_semantic_gimbal_reserve_search_scale = 0
    with pytest.raises(ValueError, match="must be positive"):
        semantic_gimbal_reserve_search_max_scale(args)


def test_hard_infeasible_search_continues_beyond_soft_reserve_cap() -> None:
    assert not should_stop_semantic_retime_search(
        hard_feasible=False,
        achieved_gimbal_margin_ratio=0.0,
        reserve_margin_ratio=0.01,
        time_scale=96,
        reserve_search_max_scale=24,
    )


def test_hard_feasible_reserve_low_search_stops_at_soft_cap() -> None:
    common = {
        "hard_feasible": True,
        "achieved_gimbal_margin_ratio": 0.006,
        "reserve_margin_ratio": 0.01,
        "reserve_search_max_scale": 24,
    }
    assert not should_stop_semantic_retime_search(time_scale=16, **common)
    assert should_stop_semantic_retime_search(time_scale=24, **common)


def test_gimbal_recovery_seeds_include_bounded_center_step() -> None:
    current = np.array([0.9, -0.4, 0.2])
    previous = np.array([0.02, -0.01, 0.0])
    lower_delta = np.full(3, -0.05)
    upper_delta = np.full(3, 0.05)
    lower = np.full(3, -1.0)
    upper = np.full(3, 1.0)

    deltas = bounded_gimbal_recovery_deltas(
        current,
        previous,
        lower_delta,
        upper_delta,
        lower,
        upper,
    )

    assert len(deltas) == 6
    np.testing.assert_allclose(deltas[0], previous)
    np.testing.assert_allclose(deltas[1], 0.0)
    np.testing.assert_allclose(deltas[2], [-0.05, 0.05, -0.05])
    np.testing.assert_allclose(deltas[3], [-0.05, 0.0, 0.0])
    np.testing.assert_allclose(deltas[4], [0.0, 0.05, 0.0])
    np.testing.assert_allclose(deltas[5], [0.0, 0.0, -0.05])
    assert all(np.all(delta >= lower_delta) for delta in deltas)
    assert all(np.all(delta <= upper_delta) for delta in deltas)


def test_gimbal_recovery_seeds_isolate_ep557_limiting_axis() -> None:
    current = np.array([2.5615494352, 1.5417994518, -1.7156941216])
    previous = np.array([1.973999e-5, 2.525e-5, -2.525e-5])
    lower_delta = np.full(3, -0.005)
    upper_delta = np.full(3, 0.005)
    lower = np.array([-3.1416, -3.2, -3.2])
    upper = np.array([3.1416, 1.57, 1.57])

    deltas = bounded_gimbal_recovery_deltas(
        current,
        previous,
        lower_delta,
        upper_delta,
        lower,
        upper,
    )

    assert any(np.allclose(delta, [0.0, -0.005, 0.0]) for delta in deltas)


def test_gimbal_recovery_seeds_deduplicate_identical_branches() -> None:
    deltas = bounded_gimbal_recovery_deltas(
        np.zeros(3),
        np.zeros(3),
        np.full(3, -0.05),
        np.full(3, 0.05),
        np.full(3, -1.0),
        np.full(3, 1.0),
    )

    assert len(deltas) == 1
    np.testing.assert_allclose(deltas[0], 0.0)


def _gravity_seed_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
    object,
]:
    baseline = np.zeros(8)
    lower = np.full(8, -0.4)
    upper = np.full(8, 0.4)
    position_jacobian = np.zeros((3, 5))
    position_jacobian[:, :3] = np.eye(3)

    def position(control: np.ndarray) -> np.ndarray:
        return position_jacobian @ control[:5]

    def gravity(control: np.ndarray) -> np.ndarray:
        return np.array([29.503752 + control[3], 2.0, -1.0])

    return baseline, lower, upper, position_jacobian, position, gravity


def test_gravity_recovery_disabled_preserves_existing_seed_order() -> None:
    existing = [np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64) + 10.0]
    expected = [seed.copy() for seed in existing]

    def unexpected(_: np.ndarray) -> np.ndarray:
        pytest.fail("disabled gravity-aware recovery evaluated callbacks")

    status = append_gravity_aware_base_arm_recovery_seed(
        existing,
        enabled=False,
        baseline_control=np.zeros(8),
        lower_control=np.full(8, -1.0),
        upper_control=np.full(8, 1.0),
        control_to_position=unexpected,
        control_to_gravity_effort=unexpected,
        finite_difference_step=1e-5,
        maximum_step_fraction=0.25,
    )

    assert status == "disabled"
    assert len(existing) == len(expected)
    for actual, wanted in zip(existing, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_gravity_recovery_appends_deterministic_nullspace_boundary_seed() -> None:
    baseline, lower, upper, jacobian, position, gravity = _gravity_seed_fixture()
    target = np.array([0.051691, 0.0, 0.0])
    assert np.linalg.norm(position(baseline) - target) == pytest.approx(0.051691)
    assert np.max(np.abs(gravity(baseline))) == pytest.approx(29.503752)

    first = [np.full(8, 0.123)]
    second = [np.full(8, 0.123)]
    common = {
        "enabled": True,
        "baseline_control": baseline,
        "lower_control": lower,
        "upper_control": upper,
        "control_to_position": position,
        "control_to_gravity_effort": gravity,
        "finite_difference_step": 1e-5,
        "maximum_step_fraction": 0.25,
    }
    assert append_gravity_aware_base_arm_recovery_seed(first, **common) == "generated"
    assert append_gravity_aware_base_arm_recovery_seed(second, **common) == "generated"

    np.testing.assert_array_equal(first[0], np.full(8, 0.123))
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all(np.isfinite(first[1]))
    np.testing.assert_allclose(jacobian @ (first[1][:5] - baseline[:5]), 0.0, atol=1e-10)
    assert np.max(np.abs(gravity(first[1]))) < np.max(np.abs(gravity(baseline)))
    assert np.linalg.norm(position(first[1]) - target) == pytest.approx(0.051691)


def test_gravity_recovery_reprojects_around_saturated_axis_before_clipping() -> None:
    baseline = np.zeros(8)
    lower = np.full(8, -0.4)
    upper = np.full(8, 0.4)
    upper[3] = 0.0
    jacobian = np.zeros((3, 5))
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    jacobian[2, 2:5] = 1.0

    def position(control: np.ndarray) -> np.ndarray:
        return jacobian @ control[:5]

    def gravity(control: np.ndarray) -> np.ndarray:
        return np.array([20.0 - control[3] + control[4], 1.0, -1.0])

    seed = gravity_aware_base_arm_recovery_seed(
        baseline,
        lower,
        upper,
        position,
        gravity,
        finite_difference_step=1e-5,
        maximum_step_fraction=1.0,
    )

    assert seed is not None
    assert np.all(seed >= lower) and np.all(seed <= upper)
    np.testing.assert_allclose(jacobian @ (seed[:5] - baseline[:5]), 0.0, atol=1e-10)
    assert seed[3] <= upper[3]


def test_gravity_recovery_deduplicates_existing_seed() -> None:
    baseline, lower, upper, _, position, gravity = _gravity_seed_fixture()
    seed = gravity_aware_base_arm_recovery_seed(
        baseline,
        lower,
        upper,
        position,
        gravity,
        finite_difference_step=1e-5,
        maximum_step_fraction=0.25,
    )
    assert seed is not None
    seeds = [np.ones(8), seed.copy()]

    status = append_gravity_aware_base_arm_recovery_seed(
        seeds,
        enabled=True,
        baseline_control=baseline,
        lower_control=lower,
        upper_control=upper,
        control_to_position=position,
        control_to_gravity_effort=gravity,
        finite_difference_step=1e-5,
        maximum_step_fraction=0.25,
    )

    assert status == "deduplicated"
    assert len(seeds) == 2


def test_gravity_recovery_rejects_nonfinite_local_model() -> None:
    baseline, lower, upper, _, _, gravity = _gravity_seed_fixture()
    with pytest.raises(ValueError, match="EE positions must be finite"):
        gravity_aware_base_arm_recovery_seed(
            baseline,
            lower,
            upper,
            lambda _: np.full(3, np.nan),
            gravity,
            finite_difference_step=1e-5,
            maximum_step_fraction=0.25,
        )


def test_gravity_recovery_is_not_attempted_after_existing_feasible_branch() -> None:
    seeds = [np.zeros(8)]

    def unexpected(_: np.ndarray) -> np.ndarray:
        pytest.fail("feasible existing recovery evaluated gravity-aware callbacks")

    status = append_gravity_aware_base_arm_recovery_seed(
        seeds,
        enabled=True,
        existing_solution_feasible=True,
        baseline_control=np.zeros(8),
        lower_control=np.full(8, -1.0),
        upper_control=np.full(8, 1.0),
        control_to_position=unexpected,
        control_to_gravity_effort=unexpected,
        finite_difference_step=1e-5,
        maximum_step_fraction=0.25,
    )

    assert status == "skipped_existing_feasible"
    assert len(seeds) == 1


def test_acquisition_route_chooses_reverse_when_it_reduces_yaw_travel() -> None:
    target = np.array([0.34391828, -0.39179693, 2.33601960])
    first_turn, drive_distance, final_turn, route = (
        select_acquisition_base_route(target)
    )

    assert route == "reverse"
    assert drive_distance < 0.0
    np.testing.assert_allclose(
        abs(first_turn) + abs(final_turn), math.radians(133.844), atol=1e-3
    )


def test_acquisition_route_preserves_target_pose_for_forward_and_reverse() -> None:
    for target in (
        np.array([1.0, 0.2, 0.3]),
        np.array([-0.4, 0.3, -2.0]),
    ):
        first_turn, drive_distance, final_turn, _ = (
            select_acquisition_base_route(target)
        )
        state = np.zeros(3)
        state[2] = first_turn
        state[:2] = drive_distance * np.array(
            [math.cos(first_turn), math.sin(first_turn)]
        )
        state[2] = wrap_angle(first_turn + final_turn)
        np.testing.assert_allclose(state[:2], target[:2], atol=1e-12)
        np.testing.assert_allclose(state[2], wrap_angle(target[2]), atol=1e-12)


def test_physical_gimbal_interpolation_rejects_equivalent_branch_jump() -> None:
    urdf = (
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
        / "recomoProto2_two_wheel_whole_body_attitude.urdf"
    )
    position_kinematics = UrdfPositionKinematics(urdf)
    camera_kinematics = UrdfPhysicalCameraKinematics(urdf)
    states = np.array(
        [
            [
                0.3381004080507735,
                -0.23059601128545915,
                -0.9768655253307925,
                0.16432255310411242,
                -0.1283785282691526,
                -1.731140586419033,
                3.1400471469325355,
                -3.0216091963656546,
                0.23872870100388868,
            ],
            [
                0.32388400192431144,
                -0.20860717394875136,
                -1.0168655253307926,
                0.203492842067913,
                -0.17837852826915265,
                -1.7697833166181511,
                0.13552841161858523,
                -0.10491205258953261,
                -1.6428041890370524,
            ],
        ]
    )
    attitudes = np.array(
        [
            [
                0.6182402293384809,
                0.2952324714745884,
                0.6532700042933122,
                -0.32226558628734314,
            ],
            [
                0.6199273707680406,
                0.29155735414977213,
                0.6551952951960943,
                -0.31843914546850477,
            ],
        ]
    )

    error_deg, interval = physical_gimbal_interpolation_error(
        states,
        attitudes,
        position_kinematics,
        camera_kinematics,
        SimpleNamespace(wheel_axle_height_m=0.1016),
    )

    assert interval == 0
    assert error_deg > 40.0


def test_gravity_aware_acquisition_avoids_unsafe_direct_path() -> None:
    urdf = (
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
        / "recomoProto2_two_wheel_whole_body_attitude.urdf"
    )
    kinematics = UrdfPositionKinematics(urdf)
    args = SimpleNamespace(
        maximum_acquisition_arm_rate=0.2,
        acquisition_dt_s=0.1,
        maximum_arm_gravity_effort_nm=29.5,
        gravity_effort_tolerance_nm=0.01,
        maximum_equilibrium_pitch_deg=180.0,
        wheel_axle_height_m=0.1016,
    )
    anchor_arm = np.array(
        [-0.30821131351426295, 0.6192383220173465, -0.8758694939937766]
    )

    path, gravity_max, plan = build_gravity_aware_arm_acquisition(
        anchor_arm, kinematics, args
    )

    assert plan.startswith("staged_")
    assert gravity_max <= args.maximum_arm_gravity_effort_nm
    np.testing.assert_allclose(path[0], HOME_ARM)
    np.testing.assert_allclose(path[-1], anchor_arm)
    maximum_rate = float(
        np.max(np.abs(np.diff(path, axis=0))) / args.acquisition_dt_s
    )
    assert maximum_rate <= args.maximum_acquisition_arm_rate + 1e-9


def test_full_pose_anchor_jointly_satisfies_case38_gravity_and_gimbal_margin() -> None:
    urdf = (
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
        / "recomoProto2_two_wheel_whole_body_attitude.urdf"
    )
    position_kinematics = UrdfPositionKinematics(urdf)
    camera_kinematics = UrdfPhysicalCameraKinematics(urdf)
    args = SimpleNamespace(
        maximum_acquisition_arm_rate=0.2,
        acquisition_dt_s=0.1,
        maximum_arm_gravity_effort_nm=29.5,
        gravity_effort_tolerance_nm=0.01,
        minimum_anchor_gimbal_limit_margin_ratio=0.10,
        maximum_equilibrium_pitch_deg=180.0,
        wheel_axle_height_m=0.1016,
    )
    source_base_arm_q = np.array(
        [
            0.3583312,
            0.75625449,
            0.56068647,
            0.57370007,
            0.66025621,
            -0.6767742,
        ]
    )
    feasible_full_state = np.array(
        [
            0.68565355,
            -0.50122637,
            -0.32407767,
            -0.31106408,
            0.62622937,
            -0.86622609,
            1.40637726,
            -1.47981939,
            -1.47745958,
        ]
    )
    target_position = position_kinematics.position(feasible_full_state[:6])
    target_physical_quaternion = matrix_quaternion_wxyz(
        physical_camera_rotation(feasible_full_state, camera_kinematics)
    )
    target_semantic_attitude = physical_cam_to_semantic_dfr_quat_wxyz(
        target_physical_quaternion
    )

    anchor, position_error, attitude_error = solve_full_pose_anchor(
        source_base_arm_q,
        target_position,
        target_semantic_attitude,
        position_kinematics,
        camera_kinematics,
        args,
    )

    gravity_max = float(
        np.max(
            np.abs(position_kinematics.gravitational_effort_nm(anchor[:6]))
        )
    )
    gimbal_range = camera_kinematics.gimbal_upper - camera_kinematics.gimbal_lower
    gimbal_margin = float(
        np.min(
            np.minimum(
                anchor[6:9] - camera_kinematics.gimbal_lower,
                camera_kinematics.gimbal_upper - anchor[6:9],
            )
            / gimbal_range
        )
    )
    assert position_error <= 1e-4
    assert attitude_error <= 0.01
    assert gravity_max <= 29.51
    assert gimbal_margin >= 0.10


def test_gravity_aware_acquisition_routes_case4_through_safe_graph() -> None:
    urdf = (
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
        / "recomoProto2_two_wheel_whole_body_attitude.urdf"
    )
    kinematics = UrdfPositionKinematics(urdf)
    args = SimpleNamespace(
        maximum_acquisition_arm_rate=0.2,
        acquisition_dt_s=0.1,
        maximum_arm_gravity_effort_nm=29.5,
        gravity_effort_tolerance_nm=0.01,
        maximum_equilibrium_pitch_deg=180.0,
        wheel_axle_height_m=0.1016,
    )
    anchor_arm = np.array(
        [-0.061872423001616765, 1.5563690195169733, -1.025919604013786]
    )

    path, gravity_max, plan = build_gravity_aware_arm_acquisition(
        anchor_arm, kinematics, args
    )

    assert plan.startswith("astar_pitch_elbow_")
    assert gravity_max <= 29.51
    np.testing.assert_allclose(path[0], HOME_ARM)
    np.testing.assert_allclose(path[-1], anchor_arm)
    maximum_rate = float(
        np.max(np.abs(np.diff(path, axis=0))) / args.acquisition_dt_s
    )
    assert maximum_rate <= args.maximum_acquisition_arm_rate + 1e-9


def test_acquisition_rejects_endpoint_above_balance_pitch_budget() -> None:
    urdf = (
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
        / "recomoProto2_two_wheel_whole_body_attitude.urdf"
    )
    kinematics = UrdfPositionKinematics(urdf)
    args = SimpleNamespace(
        maximum_acquisition_arm_rate=0.2,
        acquisition_dt_s=0.1,
        maximum_arm_gravity_effort_nm=29.5,
        gravity_effort_tolerance_nm=0.01,
        maximum_equilibrium_pitch_deg=8.5,
        wheel_axle_height_m=0.1016,
    )
    unsafe_anchor_arm = np.radians([-17.8, 21.3, -68.7])

    with pytest.raises(ValueError, match="gravity/COM-safe"):
        build_gravity_aware_arm_acquisition(
            unsafe_anchor_arm, kinematics, args, allow_astar=False
        )


def test_case28_acquisition_uses_yaw_first_com_safe_graph() -> None:
    urdf = (
        PROJECT_ROOT
        / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
        / "recomoProto2_two_wheel_whole_body_attitude.urdf"
    )
    kinematics = UrdfPositionKinematics(urdf)
    args = SimpleNamespace(
        maximum_acquisition_arm_rate=0.2,
        acquisition_dt_s=0.1,
        maximum_arm_gravity_effort_nm=29.5,
        gravity_effort_tolerance_nm=0.01,
        maximum_equilibrium_pitch_deg=10.0,
        wheel_axle_height_m=0.1016,
    )
    anchor_arm = np.array(
        [0.17624625627262436, 0.36142375712669467, -2.0143416473789166]
    )

    path, gravity_max, plan = build_gravity_aware_arm_acquisition(
        anchor_arm, kinematics, args
    )
    pitch_max = max(
        np.degrees(
            abs(
                kinematics.equilibrium_pitch_rad(
                    np.concatenate((np.zeros(3), arm)), 0.1016
                )
            )
        )
        for arm in path
    )

    assert plan.startswith("astar_pitch_elbow_yaw_first_")
    assert gravity_max <= 29.51
    assert pitch_max <= 10.000001
    np.testing.assert_allclose(path[0], HOME_ARM)
    np.testing.assert_allclose(path[-1], anchor_arm)
