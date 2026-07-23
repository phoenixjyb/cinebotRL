import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
GOAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/riser_recursive_improvement_goal_v1.json"
)


def _goal() -> dict:
    return json.loads(GOAL.read_text(encoding="utf-8"))


def test_goal_preserves_robot_and_completion_contract() -> None:
    goal = _goal()
    robot = goal["robot_contract"]
    completion = goal["completion_gates"]
    assert goal["status"] == "active"
    assert robot["arm_joint_count"] == 0
    assert robot["camera_height_m"] == [0.6, 1.8]
    assert robot["riser_speed_mps"] == 1.0
    assert completion["corrected_reference_pass_count"] == 79
    assert completion["position_error_p95_m_max"] == 0.15


def test_current_status_distinguishes_candidates_from_training_corpus() -> None:
    stage = _goal()["current_stage"]
    assert stage["status_as_of"] == "2026-07-23"
    assert stage["quality_qualified_exact_source_cases_available"] == 42
    assert stage["quality_qualified_cases_are_candidates_not_training_corpus"]
    assert stage["model_based_corrective_case_datasets_available"] == 1
    assert stage["model_based_corrective_training_corpus_cases_available"] == 0
    assert stage["corrected_reference_cases_available"] == 79
    assert "corrected_teacher_cases_available" not in stage
    assert "historical_quarantined_corrected_all79_stage" in stage
    assert "historical_quarantined_executed_residual_dataset_smoke" in stage


def test_planner_imitation_failure_and_residual_layer_are_explicit() -> None:
    refresh = _goal()["current_stage"]["status_refresh_20260723"]
    imitation = refresh["planner_imitation_bc"]
    assert imitation["classification"] == (
        "encoder_initialization_only_not_final_residual_policy"
    )
    assert imitation["case78_position_p95_m"] > imitation["case78_position_p95_gate_m"]
    assert imitation["case78_dynamic_admission_passed"] is False
    assert refresh["policy_action_contract"] == (
        "model_based_planner_plus_bounded_policy_residual_v1"
    )
    assert refresh["policy_action_names"] == [
        "delta_vx",
        "delta_wz",
        "delta_riser_target",
    ]
    assert refresh["policy_action_scales"] == [0.05, 0.05, 0.02]


def test_case23_is_the_only_next_runtime_gate_and_learning_stays_closed() -> None:
    goal = _goal()
    stage = goal["current_stage"]
    corrective = stage["status_refresh_20260723"]["model_based_corrective_teacher"]
    assert corrective["diverse_pair_tranche"] == [30, 23, 6, 2, 7]
    assert corrective["next_case"] == 23
    assert corrective["case23_pair_cpu_ready"] is True
    assert corrective["case23_pair_runtime_authorized"] is False
    assert corrective["case23_pair_executed"] is True
    assert corrective["case23_pair_passed"] is True
    assert corrective["case23_position_p95_improvement_m"] > 0.003
    assert corrective["case23_position_p95_improvement_fraction"] > 0.02
    assert corrective["case23_capture_review_ready"] is True
    assert corrective["case23_capture_authorized"] is False
    assert corrective["case23_conversion_authorized"] is False
    assert corrective["case30_valid_for_training"] is False
    assert corrective["conversion_route"]["case30_default_preserved"] is True
    assert corrective["conversion_route"]["allowed_splits"] == [
        "train",
        "validation",
    ]
    assert corrective["conversion_route"]["holdout_allowed"] is False
    corpus = corrective["multi_case_corpus_contract"]
    assert corpus["implemented"] is True
    assert corpus["implementation_commit"] == (
        "872d3e7a6430785ac6b06b45ad51c7b8e0a54523"
    )
    assert corpus["authoritative_cpu_suite"] == (
        "863_passed_11_skipped_2_warnings"
    )
    assert corpus["minimum_train_cases"] == 4
    assert corpus["minimum_validation_cases"] == 2
    assert corpus["reserved_holdout_cases_unopened"] == [3, 5, 13, 19, 24]
    assert corpus["effective_post_supervisor_labels_only"] is True
    assert corpus["bc_entrypoint_refuses_review_only_schema"] is True
    assert corrective["multi_case_corpus_created"] is False
    assert stage["runtime_authorized"] is False
    assert stage["bc_authorized"] is False
    assert stage["training_authorized"] is False
    assert stage["ppo_authorized"] is False
    assert "case23_capture_only_contract" in goal["next_iteration"]["required_change"]


def test_hardware_status_remains_measurement_blocked() -> None:
    hardware = _goal()["current_stage"]["status_refresh_20260723"][
        "hardware_readiness"
    ]
    assert hardware["bench_measurement_missing_fields"] == 34
    assert hardware["ready_for_production_design_review"] is False
    assert hardware["valid_for_production_procurement"] is False
    assert hardware["valid_for_hardware_transfer"] is False
