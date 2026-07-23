import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]
GOAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/riser_recursive_improvement_goal_v1.json"
)
GOAL_COMPLETION_AUDIT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_riser_goal_completion_audit_v1/summary.json"
)
CASE23_CAPTURE_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/model_based_corrective_teacher_case23_capture_contract_v1.json"
)
CASE23_CAPTURE_RESULT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v1_rejected/final_status.json"
)
CASE23_CAPTURE_V2_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case23_capture_contract_v2.json"
)
CASE23_CAPTURE_V2_CPU_REVIEW = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v2_cpu/summary.json"
)
CASE23_CAPTURE_V3_REJECTION = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v3_rejected_save_route/"
    "manifest.json"
)
CASE23_CAPTURE_V4_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case23_capture_contract_v4.json"
)
CASE23_CAPTURE_V4_CPU_REVIEW = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v4_cpu/summary.json"
)
DRIVE_PROFILE = (
    ROOT
    / "docs/03_training/two_wheel_balance/evidence_20260723_riser_drive_profile_selection_v1/summary.json"
)
TEMPORAL_PROJECTION_AUDIT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_temporal_projection_v1/summary.json"
)
PROJECTED_BC_LOSS_MODULE = (
    ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_bc_loss.py"
)
PROJECTED_BC_LOSS_AUDIT_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_bc_loss.py"
)
PROJECTED_BC_LOSS_AUDIT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_bc_loss_v1/summary.json"
)
PROJECTED_TRAINING_PROMOTION_MODULE = (
    ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_corrective_training_dataset.py"
)
PROJECTED_TRAINING_PROMOTION_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "promote_model_based_corrective_training_dataset.py"
)
PROJECTED_TRAINING_ADMISSION_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_CORRECTIVE_TRAINING_ADMISSION_TEMPLATE_20260723.json"
)
PROJECTED_TRAINING_BC_ADAPTER = (
    ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_corrective_bc_adapter.py"
)
PROJECTED_TRAINING_BC_TRAINER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "train_riser_residual_bc.py"
)
PROJECTED_TRAINING_BC_EXECUTION_CONTRACT = (
    ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_corrective_bc_contract.py"
)
PROJECTED_TRAINING_BC_EXECUTION_ADMISSION_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_CORRECTIVE_BC_EXECUTION_ADMISSION_TEMPLATE_20260723.json"
)
BENCH_RAW_LOG_REDUCER = (
    ROOT / "scripts/two_wheel_balance/reduce_riser_bench_log.py"
)
BENCH_RAW_LOG_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_BENCH_RAW_LOG_TEMPLATE_20260723.csv"
)
SUPPLIER_RESPONSE_AUDIT = (
    ROOT / "scripts/two_wheel_balance/audit_riser_supplier_response.py"
)
SUPPLIER_RESPONSE_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_SUPPLIER_RESPONSE_TEMPLATE_20260723.json"
)
BENCH_CANDIDATE_ROUTE_AUDIT = (
    ROOT / "scripts/two_wheel_balance/audit_riser_bench_measurements.py"
)
BENCH_400W_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
BENCH_750W_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
BENCH_750W_TEMPLATE_AUDIT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_riser_bench_750w_template_v1/summary.json"
)
BENCH_750W_ASSEMBLER = (
    ROOT / "scripts/two_wheel_balance/assemble_riser_750w_bench_evidence.py"
)
BENCH_750W_ASSEMBLY_CONTRACT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_BENCH_ASSEMBLY_CONTRACT_CN_20260723.md"
)
EXTERNAL_EVIDENCE_CHECKLIST_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_riser_750w_external_evidence_checklist.py"
)
EXTERNAL_EVIDENCE_CHECKLIST = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_riser_750w_external_checklist_v1/summary.json"
)
EXTERNAL_EVIDENCE_CHECKLIST_CN = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_EXTERNAL_EVIDENCE_CHECKLIST_CN_20260723.md"
)


def _goal() -> dict:
    return json.loads(GOAL.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert corrective["case23_capture_review_ready"] is False
    assert corrective["case23_capture_attempted"] is True
    assert corrective["case23_capture_passed"] is False
    assert corrective["case23_capture_failure_stage"] == (
        "runtime_argument_validation_before_isaac_initialization"
    )
    assert corrective["case23_capture_result_sha256"] == _sha256(
        CASE23_CAPTURE_RESULT
    )
    assert corrective["case23_pair_contract_sha256"] == (
        "e5b5b360efdb0334412fb156d77dba7e0a6eb605651c16bffc280a8076caa043"
    )
    assert corrective["case23_capture_contract_sha256"] == _sha256(
        CASE23_CAPTURE_CONTRACT
    )
    assert corrective["case23_capture_drive_profile"] == (
        "leadshine_400w_engineering_sample_v1"
    )
    assert corrective["case23_capture_drive_profile_sha256"] == _sha256(
        DRIVE_PROFILE
    )
    assert corrective["case23_capture_cpu_preflight_passed"] is True
    assert corrective["case23_capture_authorized"] is False
    assert corrective["case23_conversion_authorized"] is False
    assert corrective["case23_capture_v2_cpu_review_ready"] is True
    assert corrective["case23_capture_v2_cpu_review_sha256"] == _sha256(
        CASE23_CAPTURE_V2_CPU_REVIEW
    )
    assert corrective["case23_capture_v2_contract_sha256"] == _sha256(
        CASE23_CAPTURE_V2_CONTRACT
    )
    assert corrective["case23_capture_v2_runtime_authorized"] is False
    assert corrective["case23_capture_v2_label_capture_authorized"] is False
    assert corrective["case23_capture_v2_conversion_authorized"] is False
    assert corrective["case23_capture_v2_authorization_consumed"] is True
    assert corrective["case23_capture_v2_retry_authorized"] is False
    assert corrective["case23_capture_v3_pre_runtime_cpu_ready"] is True
    assert corrective["case23_capture_v3_cpu_ready"] is False
    assert corrective["case23_capture_v3_runtime_authorized"] is False
    assert corrective["case23_capture_v3_capture_case_propagated"] is True
    assert corrective["case23_capture_v3_capture_split_propagated"] is True
    assert corrective["case23_capture_v3_authorization_consumed"] is True
    assert corrective["case23_capture_v3_full_execution_phase_reached"] is True
    assert corrective["case23_capture_v3_dynamic_gate_result_written"] is False
    assert corrective["case23_capture_v3_passed"] is False
    assert corrective["case23_capture_v3_capture_files"] == 0
    assert corrective["case23_capture_v3_rejection_manifest_sha256"] == _sha256(
        CASE23_CAPTURE_V3_REJECTION
    )
    assert corrective["case23_capture_v3_retry_authorized"] is False
    assert corrective["case23_capture_v4_contract_sha256"] == _sha256(
        CASE23_CAPTURE_V4_CONTRACT
    )
    assert corrective["case23_capture_v4_save_case_propagated"] is True
    assert corrective["case23_capture_v4_save_split_propagated"] is True
    assert corrective["case23_capture_v4_finalizer_namespace_pinned"] is True
    assert corrective["case23_capture_v4_real_archive_to_finalizer_passed"] is True
    assert (
        corrective["case23_capture_v4_all_archive_gate_and_contract_checks_required"]
        is True
    )
    assert corrective["case23_capture_v4_authoritative_cpu_suite"] == (
        "960_passed_12_skipped_2_warnings_in_82.03s"
    )
    assert corrective["case23_capture_v4_cpu_ready"] is True
    assert corrective["case23_capture_v4_no_token_preflight_passed"] is True
    assert corrective["case23_capture_v4_runtime_authorized"] is False
    assert corrective["case23_capture_v4_label_capture_authorized"] is False
    assert corrective["case23_capture_v4_conversion_authorized"] is False
    assert json.loads(CASE23_CAPTURE_V4_CPU_REVIEW.read_text())["passed"] is True
    assert corrective["temporal_projection_audit_sha256"] == _sha256(
        TEMPORAL_PROJECTION_AUDIT
    )
    assert corrective["temporal_projection_contract"] == (
        "model_based_residual_safety_projection_v1"
    )
    assert corrective["requested_teacher_slew_violation_count"] == [0, 0, 0]
    assert corrective["effective_label_slew_violation_count"] == [30, 49, 8]
    assert corrective["effective_label_slew_violation_transition_count"] == 87
    assert corrective["effective_slew_violations_all_supervisor_clipped"] is True
    assert corrective["projection_aware_effective_label_loss_required"] is True
    assert corrective["requested_output_slew_regularization_required"] is True
    assert corrective["temporal_projection_valid_for_training"] is False
    assert corrective["temporal_projection_authoritative_cpu_suite"] == (
        "897_passed_12_skipped_2_warnings"
    )
    assert corrective["projected_bc_loss_contract"] == (
        "model_based_projected_effective_action_bc_loss_v1"
    )
    assert corrective["requested_output_slew_regularization_contract"] == (
        "requested_physical_residual_slew_hinge_v1"
    )
    assert corrective["projected_bc_loss_module_sha256"] == _sha256(
        PROJECTED_BC_LOSS_MODULE
    )
    assert corrective["projected_bc_loss_audit_script_sha256"] == _sha256(
        PROJECTED_BC_LOSS_AUDIT_SCRIPT
    )
    assert corrective["projected_bc_loss_audit_summary_sha256"] == _sha256(
        PROJECTED_BC_LOSS_AUDIT
    )
    projected_audit = json.loads(PROJECTED_BC_LOSS_AUDIT.read_text())
    assert projected_audit["passed"] is True
    assert projected_audit["sample_count"] == 11411
    assert corrective["projected_bc_loss_case30_sample_count"] == 11411
    assert corrective["projected_bc_loss_case30_pointwise_loss"] <= 1e-9
    assert corrective["naive_requested_to_effective_case30_mse"] > 0.005
    assert corrective["projected_bc_loss_requested_slew_violations"] == [
        0,
        0,
        0,
    ]
    assert corrective["projected_bc_loss_effective_projection_violations"] == [
        30,
        49,
        8,
    ]
    assert corrective["projected_bc_loss_effective_unclipped_violations"] == [
        0,
        0,
        0,
    ]
    assert corrective["projected_bc_loss_differentiable"] is True
    assert corrective["projected_bc_loss_torchscript_compatible"] is True
    assert corrective["projected_bc_loss_focused_cpu_suite"] == (
        "66_passed_13_warnings"
    )
    assert corrective["projected_bc_loss_authoritative_cpu_commit"] == (
        "da0653d39509839bdd48c5eb81de36ca8e391838"
    )
    assert corrective["projected_bc_loss_authoritative_cpu_suite"] == (
        "976_passed_12_skipped_2_warnings_in_89.44s"
    )
    assert corrective["projected_bc_loss_contract_review_passed"] is True
    assert corrective["projected_bc_loss_valid_for_training"] is False
    assert corrective["review_only_corpus_still_rejected_by_bc_entrypoint"] is True
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
    assert corpus["projection_training_dataset_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_training_v1"
    )
    assert corpus["projection_training_admission_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_training_admission_v1"
    )
    assert corpus["projection_training_promotion_module_sha256"] == _sha256(
        PROJECTED_TRAINING_PROMOTION_MODULE
    )
    assert corpus["projection_training_promotion_script_sha256"] == _sha256(
        PROJECTED_TRAINING_PROMOTION_SCRIPT
    )
    assert corpus["projection_training_admission_template_sha256"] == _sha256(
        PROJECTED_TRAINING_ADMISSION_TEMPLATE
    )
    assert corpus["projection_training_transition_contract"] == (
        "same_case_previous_row_elapsed_delta_v1"
    )
    assert corpus["projection_training_case_balancing_contract"] == (
        "unit_total_weight_per_case_v1"
    )
    assert corpus["projection_training_promotion_implemented"] is True
    assert corpus["projection_training_implementation_commit"] == (
        "6dd9027568ab7afdca68615ad08a9934191ec874"
    )
    assert corpus["projection_training_admission_template_approved"] is False
    assert corpus["projection_training_real_corpus_available"] is False
    assert corpus["projection_training_dataset_created"] is False
    assert corpus["projection_training_bc_entrypoint_integrated"] is True
    assert corpus["projection_training_bc_preflight_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_bc_preflight_v1"
    )
    assert corpus["projection_training_bc_adapter_contract"] == (
        "projection_aware_effective_label_bc_adapter_v1"
    )
    assert corpus["projection_training_bc_adapter_sha256"] == _sha256(
        PROJECTED_TRAINING_BC_ADAPTER
    )
    assert corpus["projection_training_bc_trainer_sha256"] == _sha256(
        PROJECTED_TRAINING_BC_TRAINER
    )
    assert corpus["projection_training_bc_preflight_integrated"] is True
    assert corpus["projection_training_bc_preflight_cpu_only"] is True
    assert corpus["projection_training_bc_preflight_creates_output"] is False
    assert corpus["projection_training_bc_optimizer_contract"] == (
        "exact_case_balanced_projection_aware_gradient_accumulation_v1"
    )
    assert corpus["projection_training_bc_validation_contract"] == (
        "projected_effective_action_case_balanced_validation_v1"
    )
    assert corpus["projection_training_bc_optimizer_kernel_implemented"] is True
    assert corpus["projection_training_bc_optimizer_kernel_synthetic_only"] is True
    assert corpus["projection_training_bc_optimizer_kernel_creates_artifacts"] is False
    assert corpus[
        "projection_training_bc_optimizer_kernel_implementation_commit"
    ] == ("9dad263fb14ec767cabe912276e80461c9bf4b77")
    assert corpus["projection_training_bc_optimizer_kernel_focused_cpu_suite"] == (
        "67_passed_5_warnings_in_21.22s"
    )
    assert corpus[
        "projection_training_bc_optimizer_kernel_authoritative_cpu_suite"
    ] == ("993_passed_12_skipped_2_warnings_in_102.76s")
    assert corpus["projection_training_bc_optimizer_path_integrated"] is True
    assert corpus["projection_training_bc_execution_admission_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_admission_v1"
    )
    assert corpus["projection_training_bc_execution_report_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_report_v1"
    )
    assert corpus[
        "projection_training_bc_execution_contract_module_sha256"
    ] == _sha256(PROJECTED_TRAINING_BC_EXECUTION_CONTRACT)
    assert corpus[
        "projection_training_bc_execution_admission_template_sha256"
    ] == _sha256(PROJECTED_TRAINING_BC_EXECUTION_ADMISSION_TEMPLATE)
    assert corpus[
        "projection_training_bc_execution_contract_implementation_commit"
    ] == ("ebb89fc6ae911329537b238427aa0c104fbe0f4d")
    assert corpus[
        "projection_training_bc_execution_contract_focused_cpu_suite"
    ] == ("82_passed_5_warnings_in_21.55s")
    assert corpus[
        "projection_training_bc_execution_contract_authoritative_cpu_suite"
    ] == ("1008_passed_12_skipped_2_warnings_in_96.17s")
    assert corpus["projection_training_bc_execution_admission_template_usable"] is False
    assert corpus["projection_training_bc_execution_trainer_integrated"] is True
    assert corpus["projection_training_bc_execution_synthetic_end_to_end_passed"] is True
    assert corpus["projection_training_bc_execution_real_dataset_available"] is False
    assert corpus["projection_training_bc_execution_real_admission_authorized"] is False
    assert corpus[
        "projection_training_bc_execution_trainer_integration_commit"
    ] == ("02dbff8bca1b8ed2fee3eb2598a3382c0adce0af")
    assert corpus["projection_training_bc_execution_trainer_focused_cpu_suite"] == (
        "84_passed_10_warnings_in_21.41s"
    )
    assert corpus[
        "projection_training_bc_execution_trainer_authoritative_cpu_suite"
    ] == ("1010_passed_12_skipped_2_warnings_in_93.62s")
    assert corpus["projection_training_bc_execution_real_training_started"] is False
    assert corpus["projection_training_bc_execution_real_checkpoint_created"] is False
    assert corpus["projection_training_bc_authorized"] is False
    assert corpus["projection_training_focused_cpu_suite"] == (
        "161_passed_3_warnings_in_17.59s"
    )
    assert corpus["projection_training_authoritative_cpu_commit"] == (
        "6dd9027568ab7afdca68615ad08a9934191ec874"
    )
    assert corpus["projection_training_authoritative_cpu_suite"] == (
        "989_passed_12_skipped_2_warnings_in_95.37s"
    )
    assert corpus["projection_training_bc_preflight_focused_cpu_suite"] == (
        "64_passed_5_warnings_in_20.97s"
    )
    assert corpus["projection_training_bc_preflight_authoritative_cpu_commit"] == (
        "5f46832e706356a7587cd78d67822af0679ed51d"
    )
    assert corpus["projection_training_bc_preflight_authoritative_cpu_suite"] == (
        "990_passed_12_skipped_2_warnings_in_100.07s"
    )
    assert corrective["multi_case_corpus_created"] is False
    assert stage["runtime_authorized"] is False
    assert stage["bc_authorized"] is False
    assert stage["training_authorized"] is False
    assert stage["ppo_authorized"] is False
    assert "exactly_one_case23_v4_capture" in (
        goal["next_iteration"]["required_change"]
    )


def test_hardware_status_remains_measurement_blocked() -> None:
    hardware = _goal()["current_stage"]["status_refresh_20260723"][
        "hardware_readiness"
    ]
    assert hardware["production_design_review_candidate"] == (
        "leadshine_elvm8075v48eh_m17_hd_plus_eld2_can7020b"
    )
    assert hardware["production_candidate_emergency_8kg_margin_ratio"] > 1.15
    assert hardware["production_candidate_ready_for_supplier_and_bench_review"]
    assert hardware["bench_measurement_missing_fields"] == 34
    assert hardware["bench_raw_log_reducer_schema"] == (
        "cinebotrl_two_wheel_riser_bench_log_reduction_v2"
    )
    assert hardware["bench_raw_log_reducer_legacy_schema"] == (
        "cinebotrl_two_wheel_riser_bench_log_reduction_v1"
    )
    assert hardware["bench_raw_log_numeric_reduction_ready"] is True
    assert hardware["bench_raw_log_candidate_profile_required_for_assembly"]
    assert hardware["bench_raw_log_legacy_valid_for_candidate_bound_merge"] is False
    assert hardware["bench_raw_log_measurements_collected"] is False
    assert hardware["bench_raw_log_reducer_sha256"] == _sha256(
        BENCH_RAW_LOG_REDUCER
    )
    assert hardware["bench_raw_log_template_sha256"] == _sha256(
        BENCH_RAW_LOG_TEMPLATE
    )
    assert hardware["bench_raw_log_authoritative_cpu_commit"] == (
        "e189e0732e4a650c01f6dbd7edcf1b0b8f25e69f"
    )
    assert hardware["bench_raw_log_authoritative_cpu_suite"] == (
        "932_passed_12_skipped_2_warnings_in_81.86s"
    )
    assert hardware["supplier_response_contract_schema"] == (
        "cinebotrl_two_wheel_riser_supplier_response_v1"
    )
    assert hardware["supplier_response_audit_sha256"] == _sha256(
        SUPPLIER_RESPONSE_AUDIT
    )
    assert hardware["supplier_response_template_sha256"] == _sha256(
        SUPPLIER_RESPONSE_TEMPLATE
    )
    assert hardware["supplier_response_missing_fields"] == 52
    assert hardware["supplier_response_collected"] is False
    assert hardware["supplier_response_valid_for_bench_evidence_merge"] is False
    assert (
        hardware["supplier_response_valid_for_current_400w_bench_evidence_merge"]
        is False
    )
    assert (
        hardware["supplier_response_valid_for_750w_bench_evidence_merge"] is False
    )
    assert hardware["bench_candidate_route_audit_sha256"] == _sha256(
        BENCH_CANDIDATE_ROUTE_AUDIT
    )
    assert hardware["bench_400w_legacy_template_sha256"] == _sha256(
        BENCH_400W_TEMPLATE
    )
    assert hardware["bench_750w_template_sha256"] == _sha256(
        BENCH_750W_TEMPLATE
    )
    assert hardware["bench_750w_template_audit_summary_sha256"] == _sha256(
        BENCH_750W_TEMPLATE_AUDIT
    )
    assert hardware["bench_750w_measurements_collected"] is False
    assert hardware["bench_candidate_cross_merge_rejected"] is True
    assert hardware["supplier_response_authoritative_cpu_commit"] == (
        "681977133fc8c07e790f8c43832f3e06f0dbde42"
    )
    assert hardware["supplier_response_authoritative_cpu_suite"] == (
        "921_passed_12_skipped_2_warnings_in_85.84s"
    )
    assert hardware["bench_candidate_route_authoritative_cpu_commit"] == (
        "681977133fc8c07e790f8c43832f3e06f0dbde42"
    )
    assert hardware["bench_candidate_route_authoritative_cpu_suite"] == (
        "921_passed_12_skipped_2_warnings_in_85.84s"
    )
    assert hardware["bench_750w_assembly_schema"] == (
        "cinebotrl_two_wheel_riser_750w_bench_assembly_v1"
    )
    assert hardware["bench_750w_assembler_sha256"] == _sha256(
        BENCH_750W_ASSEMBLER
    )
    assert hardware["bench_750w_assembly_contract_sha256"] == _sha256(
        BENCH_750W_ASSEMBLY_CONTRACT
    )
    assert hardware["bench_750w_assembly_cpu_ready"] is True
    assert hardware["bench_750w_real_evidence_assembled"] is False
    assert hardware["bench_750w_assembly_valid_for_production_design_review"] is False
    assert hardware["bench_750w_assembly_valid_for_production_procurement"] is False
    assert hardware["bench_750w_assembly_valid_for_hardware_transfer"] is False
    assert hardware["bench_750w_assembly_authoritative_cpu_commit"] == (
        "e189e0732e4a650c01f6dbd7edcf1b0b8f25e69f"
    )
    assert hardware["bench_750w_assembly_authoritative_cpu_suite"] == (
        "932_passed_12_skipped_2_warnings_in_81.86s"
    )
    assert hardware["external_evidence_checklist_schema"] == (
        "cinebotrl_two_wheel_riser_750w_external_evidence_checklist_v1"
    )
    assert hardware["external_evidence_checklist_builder_sha256"] == _sha256(
        EXTERNAL_EVIDENCE_CHECKLIST_BUILDER
    )
    assert hardware["external_evidence_checklist_summary_sha256"] == _sha256(
        EXTERNAL_EVIDENCE_CHECKLIST
    )
    assert hardware["external_evidence_checklist_cn_sha256"] == _sha256(
        EXTERNAL_EVIDENCE_CHECKLIST_CN
    )
    assert hardware["external_evidence_checklist_authoritative_cpu_commit"] == (
        "8c8f627f6e4a2d51eefee3a8ccccfaa496d51bd3"
    )
    assert hardware["external_evidence_checklist_authoritative_cpu_suite"] == (
        "965_passed_12_skipped_2_warnings_in_81.77s"
    )
    checklist = json.loads(EXTERNAL_EVIDENCE_CHECKLIST.read_text())
    assert checklist["external_collection_package_ready"] is True
    assert checklist["hardware_qualified"] is False
    assert (
        hardware["external_evidence_supplier_missing_fields_preserved"]
        == checklist["supplier_collection"]["missing_or_invalid_field_count"]
        == 52
    )
    assert (
        hardware["external_evidence_bench_missing_fields_preserved"]
        == checklist["bench_collection"]["missing_or_invalid_field_count"]
        == 34
    )
    assert hardware["external_evidence_collection_package_ready"] is True
    assert (
        hardware["external_evidence_manual_or_supplier_approval_synthesized"]
        is False
    )
    assert hardware["external_evidence_real_supplier_response_collected"] is False
    assert hardware["external_evidence_real_bench_measurements_collected"] is False
    assert hardware["ready_for_production_design_review"] is False
    assert hardware["valid_for_production_procurement"] is False
    assert hardware["valid_for_hardware_transfer"] is False


def test_goal_completion_audit_preserves_the_real_end_state() -> None:
    audit = _goal()["current_stage"]["status_refresh_20260723"][
        "goal_completion_audit"
    ]
    assert audit["schema"] == "cinebotrl_two_wheel_riser_goal_completion_audit_v1"
    assert audit["implementation_commit"] == (
        "15a58801ae0fab3fdc5686a1d0a7b0da8b42e6ef"
    )
    assert audit["host_independent_lf_evidence"] is True
    assert audit["mac_and_windows_report_byte_parity_verified"] is True
    assert audit["auditor_code_identity_bound"] is True
    assert audit["all79_full_row_revalidation_required"] is True
    assert audit["learned_all79_admission_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_learned_all79_admission_v1"
    )
    assert audit["learned_all79_admission_template_sha256"] == (
        "955cbe3068aa0bb2f8b601dbeddd283b86aab6ebb23034315f0bf1ef4d1ed438"
    )
    assert audit["learned_all79_admission_implementation_commit"] == (
        "15a58801ae0fab3fdc5686a1d0a7b0da8b42e6ef"
    )
    assert audit["learned_all79_exact_source_and_plan_identity_bound"] is True
    assert audit["learned_all79_runtime_asset_identity_bound"] is True
    assert audit["learned_all79_raw_rollout_hashes_required"] is True
    assert audit["learned_all79_model_based_route_required"] is True
    assert audit["learned_all79_preflight_and_resume_contract_required"] is True
    assert audit["learned_all79_execution_wrapper_sha256"] == _sha256(
        ROOT / "scripts/two_wheel_balance/run_model_based_learned_all79_policy_gate.sh"
    )
    assert audit["learned_all79_preflight_validator_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/"
        "validate_model_based_learned_all79_admission.py"
    )
    assert audit["learned_all79_requires_validation_and_holdout"] is True
    assert audit["learned_split_admission_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_learned_split_admission_v1"
    )
    assert audit["learned_split_admission_template_sha256"] == _sha256(
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "MODEL_BASED_LEARNED_SPLIT_ADMISSION_TEMPLATE_20260723.json"
    )
    assert audit["learned_split_admission_module_sha256"] == _sha256(
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_learned_split_contract.py"
    )
    assert audit["learned_split_execution_wrapper_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/"
        "run_model_based_learned_split_policy_gate.sh"
    )
    assert audit["learned_split_preflight_validator_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/"
        "validate_model_based_learned_split_admission.py"
    )
    assert audit["learned_split_model_based_route_required"] is True
    assert audit["learned_split_validation_before_holdout_required"] is True
    assert audit["learned_split_holdout_cases"] == [3, 5, 13, 19, 24]
    assert (
        audit["learned_split_admission_and_preflight_contents_revalidated_by_all79"]
        is True
    )
    assert audit["learned_split_runtime_authorized"] is False
    assert audit["learned_split_holdout_opened"] is False
    assert audit["learned_all79_runtime_authorized"] is False
    assert audit["render_policy_all79_video_hashes_required"] is True
    assert audit["render_intact_robot_visual_checks_required"] is True
    assert audit["required_gate_pass_count"] == 6
    assert audit["required_gate_count"] == 10
    assert audit["completion_blockers"] == [
        "model_based_corrective_training_corpus",
        "projection_aware_bc_policy",
        "learned_policy_all79_dynamic_gate",
        "learned_policy_render_audit",
    ]
    assert audit["physical_riser_bench_qualification_required_for_goal"] is False
    assert audit["physical_riser_bench_qualification_passed"] is False
    assert audit["goal_achieved"] is False
    assert audit["runtime_started"] is False
    assert audit["bc_started_by_audit"] is False
    assert audit["ppo_started_by_audit"] is False
    assert audit["focused_local_cpu_suite"] == "61_passed_2_warnings"
    assert audit["focused_windows_cpu_suite"] == "61_passed_2_warnings"
    assert audit["authoritative_windows_cpu_suite"] == (
        "1068_passed_12_skipped_2_warnings_in_111.96s"
    )
    assert _sha256(GOAL_COMPLETION_AUDIT) == (
        "387d15eb4a812bff8461bd5480be5b5864bae34e0b93205fbc4d1203eb20d51f"
    )
    report = json.loads(GOAL_COMPLETION_AUDIT.read_text())
    assert report["git"]["head"] == "e6ceab046f9bab5954697fca01fa774c0642d92f"
    assert report["inputs"]["auditor"]["sha256"] == (
        "b895c25183ba1a3451f76377a62a9abbe632babd57db24c1a509947fb97d0510"
    )
    assert report["required_gate_pass_count"] == audit["required_gate_pass_count"]
    assert report["required_gate_count"] == audit["required_gate_count"]
    assert report["completion_blockers"] == audit["completion_blockers"]
    assert report["goal_achieved"] is False
    assert report["runtime_started"] is False
    assert report["bc_started_by_audit"] is False
    assert report["ppo_started_by_audit"] is False
