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
CASE23_CAPTURE_V4_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v4"
)
CASE23_CAPTURE_V4_FINAL = CASE23_CAPTURE_V4_EVIDENCE / "final_status.json"
CASE23_CAPTURE_V4_ARCHIVE = (
    CASE23_CAPTURE_V4_EVIDENCE
    / "capture/case_0023_corrective_teacher_capture_v2.npz"
)
CASE23_CONVERSION_REVIEW = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_conversion_review_v1/summary.json"
)
CASE23_CONVERSION_REVIEW_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_review_contract_v1.json"
)
CASE23_CONVERSION_REVIEWER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case23_conversion_readiness.py"
)
CASE23_CONVERSION_EXECUTION_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case23_corrective_conversion_execution_cpu_v2/summary.json"
)
CASE23_CONVERSION_EXECUTION_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_execution_contract_v1.json"
)
CASE23_CONVERSION_EXECUTION_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_case23_conversion_execution.py"
)
CASE23_CONVERSION_EXECUTION_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_case23_conversion_v1.sh"
)
CASE23_CONVERSION_EXECUTION_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "finalize_model_based_corrective_case23_conversion.py"
)
CORRECTIVE_CORPUS_INTAKE_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_corpus_intake.py"
)
CORRECTIVE_CORPUS_INTAKE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_model_based_corrective_corpus_intake_v1/summary.json"
)
CASE6_PAIR_READINESS_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case6_pair_readiness.py"
)
CASE6_PAIR_READINESS_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_pair_readiness_cpu_v1/summary.json"
)
CASE6_PAIR_PROFILE_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case6_pair_profiles.py"
)
CASE6_PAIR_PROFILE_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_pair_profile_cpu_v1/proposal.json"
)
CASE6_CORRECTIVE_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case6_profile_v1.json"
)
CASE6_WRENCH_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case6_wrench_profile_v1.json"
)
CASE2_PAIR_READINESS_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case2_pair_readiness.py"
)
CASE2_PAIR_READINESS_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_pair_readiness_cpu_v1/summary.json"
)
CASE2_NATURAL_ERROR_PROFILE_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case2_natural_error_profile.py"
)
CASE2_NATURAL_ERROR_PROFILE_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_natural_error_profile_cpu_v1/proposal.json"
)
CASE2_NATURAL_ERROR_CORRECTIVE_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case2_natural_error_profile_v1.json"
)
CASE2_NATURAL_ERROR_PAIR_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case2_natural_error_pair_contract_v1.json"
)
CASE2_NATURAL_ERROR_PAIR_ADAPTER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "smoke_riser_case2_natural_error_pair.py"
)
CASE2_NATURAL_ERROR_PAIR_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_natural_error_pair_route_cpu_v1/summary.json"
)
CASE7_PAIR_READINESS_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case7_pair_readiness.py"
)
CASE7_PAIR_READINESS_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_pair_readiness_cpu_v1/summary.json"
)
CASE7_PAIR_PROFILE_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case7_pair_profiles.py"
)
CASE7_PAIR_PROFILE_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_pair_profile_cpu_v1/proposal.json"
)
CASE7_CORRECTIVE_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_profile_v1.json"
)
CASE7_WRENCH_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_wrench_profile_v1.json"
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
    assert stage["status_as_of"] == "2026-07-24"
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


def test_case23_v4_capture_is_preserved_and_learning_stays_closed() -> None:
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
    assert corrective["case23_capture_v4_pre_runtime_cpu_ready"] is True
    assert corrective["case23_capture_v4_cpu_ready"] is False
    assert corrective["case23_capture_v4_no_token_preflight_passed"] is True
    assert corrective["case23_capture_v4_runtime_authorized"] is False
    assert corrective["case23_capture_v4_label_capture_authorized"] is False
    assert corrective["case23_capture_v4_attempted"] is True
    assert corrective["case23_capture_v4_authorization_consumed"] is True
    assert corrective["case23_capture_v4_completed_steps"] == 3273
    assert corrective["case23_capture_v4_dynamic_quality_passed"] is True
    assert corrective["case23_capture_v4_archive_checks_passed"] is True
    assert corrective["case23_capture_v4_passed"] is True
    assert corrective["case23_capture_v4_capture_files"] == 1
    assert corrective["case23_capture_v4_retry_authorized"] is False
    assert (
        corrective["case23_capture_v4_capture_admitted_for_dataset_conversion"]
        is True
    )
    assert corrective["case23_capture_v4_final_status_sha256"] == _sha256(
        CASE23_CAPTURE_V4_FINAL
    )
    assert corrective["case23_capture_v4_capture_sha256"] == _sha256(
        CASE23_CAPTURE_V4_ARCHIVE
    )
    assert corrective["case23_capture_v4_evidence_commit"] == (
        "8e3ea24482cb03eefcc0d55e3acfc0846148d196"
    )
    assert corrective["case23_capture_v4_normalized_mode_commit"] == (
        "46370ecf03957b6921c9dd93bff86ae1cdf54df1"
    )
    assert corrective["case23_capture_v4_post_runtime_authoritative_cpu_suite"] == (
        "1082_passed_12_skipped_2_warnings_in_114.24s"
    )
    final = json.loads(CASE23_CAPTURE_V4_FINAL.read_text())
    assert final["case"] == 23
    assert final["split"] == "train"
    assert final["passed"] is True
    assert final["dynamic_quality_passed"] is True
    assert final["capture_admitted_for_dataset_conversion"] is True
    assert final["normalized_training_dataset_created"] is False
    assert final["bc_authorized"] is False
    assert final["ppo_authorized"] is False
    assert final["training_started"] is False
    assert final["valid_for_training"] is False
    assert corrective["case23_capture_v4_conversion_review_passed"] is True
    assert (
        corrective["case23_capture_v4_conversion_review_sha256"]
        == _sha256(CASE23_CONVERSION_REVIEW)
    )
    assert (
        corrective["case23_capture_v4_conversion_review_contract_sha256"]
        == _sha256(CASE23_CONVERSION_REVIEW_CONTRACT)
    )
    assert corrective["case23_capture_v4_conversion_reviewer_sha256"] == _sha256(
        CASE23_CONVERSION_REVIEWER
    )
    review = json.loads(CASE23_CONVERSION_REVIEW.read_text())
    assert review["case"] == 23
    assert review["split"] == "train"
    assert review["passed"] is True
    assert review["prospective_case_dataset_valid_for_merge"] is True
    assert review["prospective_dataset_metrics"]["sample_count"] == 3273
    assert review["output_created"] is False
    assert review["conversion_authorized"] is False
    assert review["merged_dataset_created"] is False
    assert review["bc_authorized"] is False
    assert review["ppo_authorized"] is False
    assert review["training_started"] is False
    assert review["valid_for_training"] is False
    assert corrective[
        "case23_capture_v4_conversion_review_authoritative_cpu_commit"
    ] == "34ca577f5c83ce3d3cf229d261ba467a85e9b5e8"
    assert corrective[
        "case23_capture_v4_conversion_review_authoritative_cpu_suite"
    ] == "1087_passed_12_skipped_2_warnings_in_109.31s"
    assert corrective["case23_capture_v4_conversion_execution_schema"] == (
        "cinebotrl_two_wheel_riser_case23_conversion_execution_admission_v1"
    )
    assert corrective[
        "case23_capture_v4_conversion_execution_implementation_commit"
    ] == "02a090e02f03523c0274151202ab7af204585c32"
    assert corrective[
        "case23_capture_v4_conversion_execution_wsl_ext4_fix_commit"
    ] == "298805562202320c72319f7adb0f955fd9568116"
    assert corrective[
        "case23_capture_v4_conversion_execution_contract_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_CONTRACT)
    assert corrective[
        "case23_capture_v4_conversion_execution_contract_git_blob_sha1"
    ] == "6ccbbc5c3153b38d5c33cebac026cb5dfee3c1ea"
    assert corrective[
        "case23_capture_v4_conversion_execution_validator_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_VALIDATOR)
    assert corrective[
        "case23_capture_v4_conversion_execution_wrapper_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_WRAPPER)
    assert corrective[
        "case23_capture_v4_conversion_execution_finalizer_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_FINALIZER)
    assert corrective[
        "case23_capture_v4_conversion_execution_preflight_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_EVIDENCE)
    assert corrective[
        "case23_capture_v4_conversion_execution_cpu_contract_ready"
    ] is True
    assert corrective[
        "case23_capture_v4_conversion_execution_token_requires_wsl_ext4"
    ] is True
    assert corrective[
        "case23_capture_v4_conversion_execution_token_present"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_output_created"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_authoritative_cpu_commit"
    ] == "3040a6db2b70b1fced0fd306ea17e2a008009bd3"
    assert corrective[
        "case23_capture_v4_conversion_execution_authoritative_cpu_suite"
    ] == "1096_passed_12_skipped_2_warnings_in_130.96s"
    assert corrective["corrective_corpus_intake_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_corpus_intake_v1"
    )
    assert corrective["corrective_corpus_intake_implementation_commit"] == (
        "e6a3688de943864f043691f407de90eb0e51f75d"
    )
    assert corrective["corrective_corpus_intake_script_sha256"] == _sha256(
        CORRECTIVE_CORPUS_INTAKE_SCRIPT
    )
    assert corrective["corrective_corpus_intake_summary_sha256"] == _sha256(
        CORRECTIVE_CORPUS_INTAKE_EVIDENCE
    )
    assert corrective["corrective_corpus_intake_mac_windows_byte_parity"] is True
    assert corrective["corrective_corpus_intake_converted_train_cases"] == [30]
    assert corrective["corrective_corpus_intake_converted_validation_cases"] == []
    assert corrective["corrective_corpus_intake_missing_train_case_count"] == 3
    assert corrective["corrective_corpus_intake_missing_validation_case_count"] == 2
    assert corrective["corrective_corpus_intake_pending_minimum_train_cases"] == [
        23,
        6,
        2,
    ]
    assert corrective["corrective_corpus_intake_pending_validation_cases"] == [
        8,
        16,
    ]
    assert corrective["corrective_corpus_intake_manifest_ready"] is False
    assert corrective["corrective_corpus_intake_merge_authorized"] is False
    assert corrective["corrective_corpus_intake_authoritative_cpu_commit"] == (
        "fa4c834ae78ed65c74bd1e369c9e4868ea0c2d44"
    )
    assert corrective["corrective_corpus_intake_authoritative_cpu_suite"] == (
        "1107_passed_12_skipped_2_warnings_in_112.64s"
    )
    assert corrective["case6_pair_readiness_schema"] == (
        "cinebotrl_two_wheel_riser_case6_pair_readiness_cpu_v1"
    )
    assert corrective["case6_pair_readiness_implementation_commit"] == (
        "95666c94930eba4f9726a5d8ff3dbb7dcea83a40"
    )
    assert corrective["case6_pair_readiness_script_sha256"] == _sha256(
        CASE6_PAIR_READINESS_SCRIPT
    )
    assert corrective["case6_pair_readiness_script_git_blob_sha1"] == (
        "a2c10a0ceac6adbd0901d20a6bb4c02c1fd397e1"
    )
    assert corrective["case6_pair_readiness_summary_sha256"] == _sha256(
        CASE6_PAIR_READINESS_EVIDENCE
    )
    assert corrective["case6_pair_readiness_mac_windows_byte_parity"] is True
    assert corrective["case6_pair_readiness_focused_cpu_suite"] == (
        "24_passed_2_warnings"
    )
    assert corrective["case6_pair_readiness_authoritative_cpu_commit"] == (
        "993b81a5bb11e18d5a08d79b667beddb5d9a3b10"
    )
    assert corrective["case6_pair_readiness_authoritative_cpu_suite"] == (
        "1114_passed_12_skipped_2_warnings_in_114.32s"
    )
    assert corrective["case6_pair_readiness_source_states"] == 807
    assert corrective["case6_pair_readiness_transitions"] == 806
    assert corrective["case6_pair_readiness_camera_height_range_m"] == [
        0.6,
        1.528812,
    ]
    assert (
        corrective["case6_pair_readiness_lever_arm_saturation_ratio"] > 0.95
    )
    assert corrective[
        "case6_pair_readiness_case_specific_profile_required"
    ] is True
    assert corrective[
        "case6_pair_readiness_case23_profile_reuse_authorized"
    ] is False
    assert corrective["case6_pair_readiness_pair_profile_cpu_ready"] is False
    assert corrective["case6_pair_readiness_runtime_authorized"] is False
    assert corrective["case6_pair_readiness_label_capture_authorized"] is False
    assert corrective["case6_pair_readiness_training_authorized"] is False
    case6 = json.loads(CASE6_PAIR_READINESS_EVIDENCE.read_text())
    assert case6["passed"] is True
    assert case6["case_specific_profile_required"] is True
    assert case6["case23_profile_reuse_authorized"] is False
    assert case6["pair_profile_cpu_ready"] is False
    assert case6["runtime_authorized"] is False
    assert case6["label_capture_authorized"] is False
    assert case6["training_started"] is False
    assert corrective["case6_pair_profile_schema"] == (
        "cinebotrl_two_wheel_riser_case6_pair_profile_proposal_cpu_v1"
    )
    assert corrective["case6_pair_profile_implementation_commit"] == (
        "8667320d83c3fd3518927bfca3819b061532cb50"
    )
    assert corrective["case6_pair_profile_builder_sha256"] == _sha256(
        CASE6_PAIR_PROFILE_BUILDER
    )
    assert corrective["case6_pair_profile_builder_git_blob_sha1"] == (
        "00d2ba88d8fa8f2c86ed65b82100d8620a6090ed"
    )
    assert corrective["case6_pair_profile_proposal_sha256"] == _sha256(
        CASE6_PAIR_PROFILE_PROPOSAL
    )
    assert corrective["case6_corrective_profile_sha256"] == _sha256(
        CASE6_CORRECTIVE_PROFILE
    )
    assert corrective["case6_wrench_profile_sha256"] == _sha256(
        CASE6_WRENCH_PROFILE
    )
    assert corrective["case6_pair_profile_mac_windows_byte_parity"] is True
    assert corrective["case6_pair_profile_focused_cpu_suite"] == (
        "50_passed_2_warnings"
    )
    assert corrective["case6_pair_profile_authoritative_cpu_commit"] == (
        "3e3a4f070b1384a8a798d908e4bc174060921ba7"
    )
    assert corrective["case6_pair_profile_authoritative_cpu_suite"] == (
        "1124_passed_12_skipped_2_warnings_in_113.57s"
    )
    assert corrective[
        "case6_pair_profile_envelope_retention_fraction"
    ] == 0.75
    assert corrective["case6_pair_profile_maximum_residuals"] == [
        0.028767878925779956,
        0.007952802338471211,
        0.0017865156836203155,
    ]
    assert corrective["case6_pair_profile_slew_horizon_s"] == 0.3
    assert corrective["case6_pair_profile_pulse_duration_steps"] == 20
    assert corrective["case6_pair_profile_pulse_force_body_x_n"] == 20.0
    assert corrective["case6_pair_profile_recovery_tail_s"] > 0.45
    assert corrective["case6_pair_profile_cpu_ready"] is True
    assert corrective["case6_pair_profile_runtime_route_implemented"] is False
    assert corrective["case6_pair_profile_runtime_authorized"] is False
    assert corrective["case6_pair_profile_label_capture_authorized"] is False
    assert corrective["case6_pair_profile_training_authorized"] is False
    case6_profile = json.loads(CASE6_PAIR_PROFILE_PROPOSAL.read_text())
    assert case6_profile["passed"] is True
    assert case6_profile["pair_profile_cpu_ready"] is True
    assert case6_profile["runtime_route_implemented"] is False
    assert case6_profile["runtime_authorized"] is False
    assert case6_profile["label_capture_authorized"] is False
    assert case6_profile["training_started"] is False
    assert corrective["case2_pair_readiness_schema"] == (
        "cinebotrl_two_wheel_riser_case2_pair_readiness_cpu_v1"
    )
    assert corrective["case2_pair_readiness_implementation_commit"] == (
        "9a16e5ca18ab4da96d78269234b674eec88d103c"
    )
    assert corrective["case2_pair_readiness_script_sha256"] == _sha256(
        CASE2_PAIR_READINESS_SCRIPT
    )
    assert corrective["case2_pair_readiness_script_git_blob_sha1"] == (
        "f535e9bfb3de7b41e99e7398e81aff86b7c5e5cc"
    )
    assert corrective["case2_pair_readiness_summary_sha256"] == _sha256(
        CASE2_PAIR_READINESS_EVIDENCE
    )
    assert corrective["case2_pair_readiness_mac_windows_byte_parity"] is True
    assert corrective["case2_pair_readiness_focused_cpu_suite"] == (
        "15_passed_2_warnings"
    )
    assert corrective["case2_pair_readiness_authoritative_cpu_suite"] == (
        "1149_passed_12_skipped_2_warnings_in_124.40s"
    )
    assert corrective["case2_pair_readiness_source_states"] == 480
    assert corrective["case2_pair_readiness_transitions"] == 479
    assert corrective["case2_pair_readiness_camera_height_range_m"] == [
        1.174803,
        1.362249,
    ]
    assert corrective["case2_pair_readiness_low_motion_window_found"] is False
    assert corrective[
        "case2_pair_readiness_case_specific_profile_required"
    ] is True
    assert corrective[
        "case2_pair_readiness_case23_profile_reuse_authorized"
    ] is False
    assert corrective[
        "case2_pair_readiness_case6_profile_reuse_authorized"
    ] is False
    assert corrective["case2_pair_readiness_pair_profile_cpu_ready"] is False
    assert corrective["case2_pair_readiness_runtime_authorized"] is False
    assert corrective["case2_pair_readiness_label_capture_authorized"] is False
    assert corrective["case2_pair_readiness_training_authorized"] is False
    case2_readiness = json.loads(CASE2_PAIR_READINESS_EVIDENCE.read_text())
    assert case2_readiness["passed"] is True
    assert case2_readiness[
        "safe_window_absent_requires_structural_profile"
    ] is True
    assert case2_readiness["case23_profile_reuse_authorized"] is False
    assert case2_readiness["case6_profile_reuse_authorized"] is False
    assert case2_readiness["runtime_authorized"] is False
    assert case2_readiness["label_capture_authorized"] is False
    assert case2_readiness["training_started"] is False
    assert corrective["case2_natural_error_profile_schema"] == (
        "cinebotrl_two_wheel_riser_case2_natural_error_profile_proposal_cpu_v1"
    )
    assert corrective[
        "case2_natural_error_profile_implementation_commit"
    ] == "2911a0bdb45cf4c83d57a490dfdff9d9a9e90a58"
    assert corrective["case2_natural_error_profile_builder_sha256"] == _sha256(
        CASE2_NATURAL_ERROR_PROFILE_BUILDER
    )
    assert corrective[
        "case2_natural_error_profile_builder_git_blob_sha1"
    ] == "3d9ba69980b6bb8c1a4cb7c0ec592ca1d0317851"
    assert corrective[
        "case2_natural_error_profile_proposal_sha256"
    ] == _sha256(CASE2_NATURAL_ERROR_PROFILE_PROPOSAL)
    assert corrective[
        "case2_natural_error_corrective_profile_sha256"
    ] == _sha256(CASE2_NATURAL_ERROR_CORRECTIVE_PROFILE)
    assert corrective[
        "case2_natural_error_profile_mac_windows_byte_parity"
    ] is True
    assert corrective[
        "case2_natural_error_profile_focused_cpu_suite"
    ] == "50_passed_2_warnings"
    assert corrective[
        "case2_natural_error_profile_authoritative_cpu_suite"
    ] == "1161_passed_12_skipped_2_warnings_in_122.95s"
    assert corrective[
        "case2_natural_error_profile_envelope_retention_fraction"
    ] == 0.25
    assert corrective["case2_natural_error_profile_maximum_residuals"] == [
        0.010247377951381045,
        0.004541053111745168,
        0.000614431056640985,
    ]
    assert corrective["case2_natural_error_profile_slew_horizon_s"] == 0.4
    assert corrective["case2_natural_error_trace_samples"] == 47
    assert corrective["case2_natural_error_samples_above_003m"] == 42
    assert corrective["case2_natural_error_external_wrench_required"] is False
    assert corrective[
        "case2_natural_error_outward_linear_projection_transitions"
    ] == 430
    assert corrective[
        "case2_natural_error_outward_yaw_projection_transitions"
    ] == 103
    assert corrective["case2_natural_error_pair_profile_cpu_ready"] is True
    assert corrective["case2_natural_error_pair_contract_sha256"] == _sha256(
        CASE2_NATURAL_ERROR_PAIR_CONTRACT
    )
    assert corrective["case2_natural_error_pair_adapter_sha256"] == _sha256(
        CASE2_NATURAL_ERROR_PAIR_ADAPTER
    )
    assert corrective["case2_natural_error_pair_identity_count"] == 19
    assert corrective["case2_natural_error_pair_reset_seed"] == (
        corrective["case2_natural_error_pair_configuration_seed"] + 2
    )
    assert corrective["case2_natural_error_pair_external_wrench_used"] is False
    assert (
        corrective[
            "case2_natural_error_pair_projection_observer_modifies_commands"
        ]
        is False
    )
    assert corrective[
        "case2_natural_error_pair_historical_case23_v4_regression_passed"
    ] is True
    assert corrective["case2_natural_error_runtime_route_implemented"] is True
    assert corrective["case2_natural_error_execution_route_complete"] is True
    assert corrective["case2_natural_error_cpu_preflight_passed"] is True
    assert (
        corrective["case2_natural_error_authorization_token_issued"] is False
    )
    assert corrective["case2_natural_error_runtime_authorized"] is False
    assert corrective["case2_natural_error_label_capture_authorized"] is False
    assert corrective["case2_natural_error_training_authorized"] is False
    case2_route = json.loads(CASE2_NATURAL_ERROR_PAIR_EVIDENCE.read_text())
    assert case2_route["passed"] is True
    assert case2_route["execution_route_complete"] is True
    assert case2_route["runtime_authorized"] is False
    assert case2_route["gpu_launch_authorized"] is False
    assert case2_route["authorization_token_issued"] is False
    assert case2_route["label_capture_authorized"] is False
    assert case2_route["dataset_creation_authorized"] is False
    assert case2_route["training_started"] is False
    assert case2_route["valid_for_training"] is False
    case2_profile = json.loads(
        CASE2_NATURAL_ERROR_PROFILE_PROPOSAL.read_text()
    )
    assert case2_profile["passed"] is True
    assert case2_profile["pair_profile_cpu_ready"] is True
    assert case2_profile["runtime_route_implemented"] is False
    assert case2_profile["runtime_authorized"] is False
    assert case2_profile["label_capture_authorized"] is False
    assert case2_profile["training_started"] is False
    assert corrective["case7_pair_readiness_schema"] == (
        "cinebotrl_two_wheel_riser_case7_pair_readiness_cpu_v1"
    )
    assert corrective["case7_pair_readiness_script_sha256"] == _sha256(
        CASE7_PAIR_READINESS_SCRIPT
    )
    assert corrective["case7_pair_readiness_script_git_blob_sha1"] == (
        "267afc56c52f749d87b02ac43134fe73532c5a27"
    )
    assert corrective["case7_pair_readiness_summary_sha256"] == _sha256(
        CASE7_PAIR_READINESS_EVIDENCE
    )
    assert corrective["case7_pair_readiness_mac_windows_byte_parity"] is True
    assert corrective["case7_pair_readiness_source_states"] == 663
    assert corrective["case7_pair_readiness_transitions"] == 662
    assert corrective["case7_pair_readiness_camera_height_range_m"] == [
        0.6,
        1.605452,
    ]
    assert corrective["case7_pair_readiness_low_motion_window_count"] == 4
    assert (
        corrective["case7_pair_readiness_longest_low_motion_window_s"] > 3.4
    )
    assert corrective[
        "case7_pair_readiness_case_specific_profile_required"
    ] is True
    assert corrective[
        "case7_pair_readiness_case23_profile_reuse_authorized"
    ] is False
    assert corrective[
        "case7_pair_readiness_case6_profile_reuse_authorized"
    ] is False
    assert corrective[
        "case7_pair_readiness_case2_profile_reuse_authorized"
    ] is False
    assert corrective["case7_pair_readiness_pair_profile_cpu_ready"] is False
    assert corrective["case7_pair_readiness_runtime_authorized"] is False
    assert corrective["case7_pair_readiness_label_capture_authorized"] is False
    assert corrective["case7_pair_readiness_training_authorized"] is False
    case7 = json.loads(CASE7_PAIR_READINESS_EVIDENCE.read_text())
    assert case7["passed"] is True
    assert case7["case_specific_profile_required"] is True
    assert case7["profile_window_contract"]["bounded_window_found"] is True
    assert len(case7["profile_window_contract"]["windows"]) == 4
    assert case7["case23_profile_reuse_authorized"] is False
    assert case7["case6_profile_reuse_authorized"] is False
    assert case7["case2_profile_reuse_authorized"] is False
    assert case7["runtime_authorized"] is False
    assert case7["label_capture_authorized"] is False
    assert case7["training_started"] is False
    assert corrective["case7_pair_profile_schema"] == (
        "cinebotrl_two_wheel_riser_case7_pair_profile_proposal_cpu_v1"
    )
    assert corrective["case7_pair_profile_builder_sha256"] == _sha256(
        CASE7_PAIR_PROFILE_BUILDER
    )
    assert corrective["case7_pair_profile_builder_git_blob_sha1"] == (
        "afba80e774fbf86512f6b0e2efeff4ba800b180b"
    )
    assert corrective["case7_pair_corrective_profile_sha256"] == _sha256(
        CASE7_CORRECTIVE_PROFILE
    )
    assert corrective["case7_pair_wrench_profile_sha256"] == _sha256(
        CASE7_WRENCH_PROFILE
    )
    assert corrective["case7_pair_profile_proposal_sha256"] == _sha256(
        CASE7_PAIR_PROFILE_PROPOSAL
    )
    assert corrective["case7_pair_profile_maximum_residuals"] == [
        0.019165321461451848,
        0.010077209250079967,
        0.0012628383956008627,
    ]
    assert corrective["case7_pair_profile_slew_horizon_s"] == 0.35
    assert corrective["case7_pair_profile_pulse_force_n"] == 20.0
    assert corrective["case7_pair_profile_pulse_duration_s"] == 0.1
    assert corrective["case7_pair_profile_pulse_source_sample_count"] == 4
    assert corrective["case7_pair_profile_recovery_tail_s"] > 15.0
    assert corrective[
        "case7_pair_profile_full_plan_projection_clipped_transitions"
    ] == [0, 0, 4]
    assert corrective["case7_pair_profile_pulse_window_fully_unclipped"] is True
    assert corrective["case7_pair_profile_mac_windows_hash_parity"] is True
    assert corrective["case7_pair_profile_cpu_ready"] is True
    assert corrective["case7_pair_profile_runtime_route_implemented"] is False
    assert corrective["case7_pair_profile_authorization_token_issued"] is False
    assert corrective["case7_pair_profile_runtime_authorized"] is False
    assert corrective["case7_pair_profile_label_capture_authorized"] is False
    assert corrective["case7_pair_profile_training_authorized"] is False
    case7_profile = json.loads(CASE7_PAIR_PROFILE_PROPOSAL.read_text())
    assert case7_profile["passed"] is True
    assert case7_profile["pair_profile_cpu_ready"] is True
    assert case7_profile["runtime_route_implemented"] is False
    assert case7_profile["authorization_token_issued"] is False
    assert case7_profile["runtime_authorized"] is False
    assert case7_profile["label_capture_authorized"] is False
    assert case7_profile["training_started"] is False
    intake = json.loads(CORRECTIVE_CORPUS_INTAKE_EVIDENCE.read_text())
    assert intake["passed"] is True
    assert intake["corpus_manifest_ready"] is False
    assert intake["dataset_conversion_authorized"] is False
    assert intake["dataset_merge_authorized"] is False
    assert intake["bc_authorized"] is False
    assert intake["ppo_authorized"] is False
    assert intake["training_started"] is False
    assert intake["valid_for_training"] is False
    execution_preflight = json.loads(
        CASE23_CONVERSION_EXECUTION_EVIDENCE.read_text()
    )
    assert execution_preflight["passed"] is True
    assert execution_preflight["cpu_contract_ready"] is True
    assert all(execution_preflight["repository_checks"].values())
    assert all(execution_preflight["contract_checks"].values())
    assert all(
        identity["passed"]
        for identity in execution_preflight["identities"].values()
    )
    assert execution_preflight["conversion_authorized"] is False
    assert (
        execution_preflight["authorization_consumed_before_conversion"]
        is False
    )
    assert execution_preflight["output_created"] is False
    assert execution_preflight["merged_dataset_created"] is False
    assert execution_preflight["bc_authorized"] is False
    assert execution_preflight["ppo_authorized"] is False
    assert execution_preflight["training_started"] is False
    assert execution_preflight["valid_for_training"] is False
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
    assert "exactly_one_case23_v4_cpu_conversion" in (
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
        "172d4efa8d43418b7eba656117b5004a7df7e708"
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
    assert audit["learned_render_admission_schema"].endswith(
        "model_based_learned_render_admission_v1"
    )
    assert audit["learned_render_admission_template_sha256"] == _sha256(
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "MODEL_BASED_LEARNED_RENDER_ADMISSION_TEMPLATE_20260723.json"
    )
    assert audit["learned_render_visual_review_template_sha256"] == _sha256(
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "MODEL_BASED_LEARNED_RENDER_VISUAL_REVIEW_TEMPLATE_20260723.json"
    )
    assert audit["learned_render_contract_module_sha256"] == _sha256(
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_learned_render_contract.py"
    )
    assert audit["learned_render_execution_wrapper_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/run_model_based_learned_render_gate.sh"
    )
    assert audit["learned_render_preflight_validator_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/"
        "validate_model_based_learned_render_admission.py"
    )
    assert audit["learned_render_media_auditor_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/"
        "audit_model_based_learned_render_media.py"
    )
    assert audit["learned_render_finalizer_sha256"] == _sha256(
        ROOT
        / "scripts/two_wheel_balance/finalize_model_based_learned_render.py"
    )
    assert audit["learned_render_representative_cases"] == [1, 15, 31, 50, 73, 79]
    assert (
        audit["learned_render_machine_media_and_manual_visual_review_required"]
        is True
    )
    assert audit["learned_render_runtime_authorized"] is False
    assert audit["learned_render_recording_started"] is False
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
    assert audit["focused_local_cpu_suite"] == "75_passed_2_warnings"
    assert audit["focused_windows_cpu_suite"] == "75_passed_2_warnings"
    assert audit["authoritative_windows_cpu_suite"] == (
        "1082_passed_12_skipped_2_warnings_in_113.60s"
    )
    assert _sha256(GOAL_COMPLETION_AUDIT) == (
        "b9804473371407657a1206ba18806b508e5745fc879a3d9071c0af6a8bb8c0ce"
    )
    report = json.loads(GOAL_COMPLETION_AUDIT.read_text())
    assert report["git"]["head"] == "89d74defd0783cb492a03388592cd908d09d3050"
    assert report["inputs"]["auditor"]["sha256"] == (
        "1d2903d0f7f7c9e0a76bfdc220072b89180a5f44b64cb7840925a0cddb1c8d63"
    )
    assert report["required_gate_pass_count"] == audit["required_gate_pass_count"]
    assert report["required_gate_count"] == audit["required_gate_count"]
    assert report["completion_blockers"] == audit["completion_blockers"]
    assert report["goal_achieved"] is False
    assert report["runtime_started"] is False
    assert report["bc_started_by_audit"] is False
    assert report["ppo_started_by_audit"] is False
