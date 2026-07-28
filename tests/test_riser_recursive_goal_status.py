import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
GOAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/riser_recursive_improvement_goal_v1.json"
)
GOAL_COMPLETION_AUDIT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_riser_goal_completion_audit_v7/summary.json"
)
GENERIC_CAPTURE_FINALIZER_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_generic_corrective_capture_finalizer_cpu_v1/"
    "summary.json"
)
CAPTURE_COMMAND_AUDIT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260725_corrective_capture_command_equivalence_cpu_v1/"
    "summary.json"
)
CAPTURE_COMMAND_AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_capture_command_equivalence.py"
)
CAPTURE_COMMAND_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_capture_command_contract_v1.json"
)
SHARED_RESOURCE_GUARD = (
    ROOT
    / "scripts/two_wheel_balance/check_windows_shared_resource_admission.py"
)
SHARED_RESOURCE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case7_corrective_capture_v2/"
    "resource_admission.json"
)
SHARED_RESOURCE_PREFLIGHT = (
    SHARED_RESOURCE_EVIDENCE.parent / "tokenless_preflight.json"
)
CASE7_RESOURCE_FINALIZER_SEAL_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260727_case7_resource_finalizer_seal_cpu_v1"
)
CASE7_RESOURCE_FINALIZER_SEAL_SUMMARY = (
    CASE7_RESOURCE_FINALIZER_SEAL_EVIDENCE / "summary.json"
)
CASE7_RESOURCE_FINALIZER_SEAL_PREFLIGHT = (
    CASE7_RESOURCE_FINALIZER_SEAL_EVIDENCE / "tokenless_preflight.json"
)
CASE7_RESOURCE_FINALIZER_SEAL_LIVE = (
    CASE7_RESOURCE_FINALIZER_SEAL_EVIDENCE / "live_resource_admission.json"
)
CASE7_RESOURCE_FINALIZER_SEAL_COMMANDS = (
    CASE7_RESOURCE_FINALIZER_SEAL_EVIDENCE / "command_equivalence.json"
)
CORRECTIVE_ROUTE_CATALOG = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_route_catalog_v1.json"
)
CORRECTIVE_ROUTE_PREPARER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "prepare_model_based_corrective_routes.py"
)
CORRECTIVE_ROUTE_PREFLIGHT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_corrective_projection_evidence_repair_cpu_v2/"
    "route_preflight.json"
)
PROJECTION_EVIDENCE_ENGINE = (
    ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_projection_evidence.py"
)
PROJECTION_REPAIR_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_corrective_projection_evidence_repair_cpu_v2"
)
PROJECTION_REPAIR_CASE2 = (
    PROJECTION_REPAIR_EVIDENCE / "case2_reclassification.json"
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
CASE23_CAPTURE_V4_ARCHIVED_CONTRACT = CASE23_CAPTURE_V4_EVIDENCE / "contract.json"
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
CASE23_CONVERSION_EXECUTION_V5 = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case23_corrective_conversion_execution_cpu_v5"
)
CASE23_CONVERSION_EXECUTION_V5_ADMISSION = (
    CASE23_CONVERSION_EXECUTION_V5 / "admission.json"
)
CASE23_CONVERSION_EXECUTION_V5_RESULT = (
    CASE23_CONVERSION_EXECUTION_V5 / "conversion_result.json"
)
CASE23_CONVERSION_EXECUTION_V5_DATASET = (
    CASE23_CONVERSION_EXECUTION_V5
    / "case_0023_model_based_corrective_case_dataset_v1.npz"
)
CASE23_CONVERSION_EXECUTION_V5_FINAL = (
    CASE23_CONVERSION_EXECUTION_V5 / "final_status.json"
)
CASE23_CONVERSION_EXECUTION_V5_RECOVERY = (
    CASE23_CONVERSION_EXECUTION_V5 / "recovery_audit.json"
)
CORRECTIVE_CORPUS_INTAKE_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_corpus_intake.py"
)
CORRECTIVE_CORPUS_INTAKE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_model_based_corrective_corpus_intake_v5/summary.json"
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
CASE2_NATURAL_ERROR_PAIR_EXECUTION = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_natural_error_pair_execution_v1/"
    "post_run_audit.json"
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
CASE7_PAIR_ROUTE_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_pair_contract_v1.json"
)
CASE7_PAIR_ROUTE_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case7_pair.py"
)
CASE7_PAIR_ROUTE_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case7_pair.sh"
)
CASE7_PAIR_ROUTE_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case7_pair.py"
)
CASE7_PAIR_ROUTE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_pair_route_cpu_v1/summary.json"
)
CASE7_PAIR_EXECUTION_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_corrective_pair_execution_v1"
)
CASE7_PAIR_EXECUTION_FINAL = CASE7_PAIR_EXECUTION_EVIDENCE / "final_status.json"
CASE7_PAIR_EXECUTION_PROJECTION = (
    CASE7_PAIR_EXECUTION_EVIDENCE / "projection_audit.json"
)
CASE7_CAPTURE_ROUTE_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_capture_contract_v1.json"
)
CASE7_CAPTURE_ROUTE_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case7_capture.py"
)
CASE7_CAPTURE_ROUTE_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case7_capture.sh"
)
CASE7_CAPTURE_ROUTE_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case7_capture.py"
)
CASE7_CAPTURE_RESOURCE_MONITOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "monitor_windows_shared_resource_pressure.py"
)
CASE7_CAPTURE_ROUTE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_corrective_capture_route_cpu_v2/summary.json"
)
CASE7_CAPTURE_ROUTE_PREFLIGHT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_corrective_capture_route_cpu_v2/"
    "preflight_windows.json"
)
CASE7_CAPTURE_V2_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case7_corrective_capture_v2"
)
CASE7_CAPTURE_V2_SUMMARY = CASE7_CAPTURE_V2_EVIDENCE / "summary.json"
CASE7_CAPTURE_V2_FINAL = CASE7_CAPTURE_V2_EVIDENCE / "final_status.json"
CASE7_CAPTURE_V2_GATE = CASE7_CAPTURE_V2_EVIDENCE / "case_0007.json"
CASE7_CAPTURE_V2_ARCHIVE = (
    CASE7_CAPTURE_V2_EVIDENCE
    / "capture/case_0007_corrective_teacher_capture_v2.npz"
)
CASE7_CAPTURE_V2_PREFLIGHT = (
    CASE7_CAPTURE_V2_EVIDENCE / "tokenless_preflight.json"
)
CASE7_CAPTURE_V2_RESOURCE_MONITOR = (
    CASE7_CAPTURE_V2_EVIDENCE / "resource_monitor.json"
)
GENERIC_CORRECTIVE_CONVERSION_PREPARER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "prepare_model_based_corrective_conversion_route.py"
)
GENERIC_CORRECTIVE_CONVERSION_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_generic_corrective_conversion_proposals_v1"
)
GENERIC_CORRECTIVE_CONVERSION_SUMMARY = (
    GENERIC_CORRECTIVE_CONVERSION_EVIDENCE / "summary.json"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_conversion_execution_contract_v2.json"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_conversion_execution.py"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_conversion_v2.sh"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "finalize_model_based_corrective_conversion.py"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_conversion_execution_contract.py"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_generic_corrective_conversion_execution_route_cpu_v2"
)
GENERIC_CORRECTIVE_CONVERSION_EXECUTION_SUMMARY = (
    GENERIC_CORRECTIVE_CONVERSION_EXECUTION_EVIDENCE / "summary.json"
)
CASE8_VALIDATION_PAIR_READINESS_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case8_validation_pair_readiness.py"
)
CASE8_VALIDATION_PAIR_READINESS_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case8_validation_pair_readiness_cpu_v1/summary.json"
)
CASE8_VALIDATION_PAIR_PROFILE_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case8_validation_pair_profiles.py"
)
CASE8_VALIDATION_PAIR_PROFILE_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case8_validation_pair_profile_cpu_v1/proposal.json"
)
CASE8_VALIDATION_CORRECTIVE_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_profile_v1.json"
)
CASE8_VALIDATION_WRENCH_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_wrench_profile_v1.json"
)
CASE8_VALIDATION_PAIR_ROUTE_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_pair_contract_v1.json"
)
CASE8_VALIDATION_PAIR_ROUTE_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case8_validation_pair.py"
)
CASE8_VALIDATION_PAIR_ROUTE_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case8_validation_pair.sh"
)
CASE8_VALIDATION_PAIR_ROUTE_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case8_validation_pair.py"
)
CASE8_VALIDATION_ASSESSMENT = (
    ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_corrective_validation.py"
)
CASE8_VALIDATION_PAIR_ROUTE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case8_validation_pair_v2/tokenless_preflight.json"
)
CASE8_VALIDATION_PAIR_EXECUTION_SUMMARY = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case8_validation_pair_v2/summary.json"
)
CASE8_VALIDATION_CAPTURE_SUMMARY = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case8_validation_capture_v1/summary.json"
)
CASE8_VALIDATION_CONVERSION_SUMMARY = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case8_validation_conversion_execution_cpu_v1/"
    "summary.json"
)
CASE16_VALIDATION_PAIR_READINESS_SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case16_validation_pair_readiness.py"
)
CASE16_VALIDATION_PAIR_READINESS_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case16_validation_pair_readiness_cpu_v1/summary.json"
)
CASE16_VALIDATION_NATURAL_ERROR_PROFILE_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case16_validation_natural_error_profile.py"
)
CASE16_VALIDATION_NATURAL_ERROR_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case16_validation_natural_error_"
    "profile_v1.json"
)
CASE16_VALIDATION_NATURAL_ERROR_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case16_validation_natural_error_profile_cpu_v1/"
    "proposal.json"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case16_validation_natural_error_"
    "pair_contract_v1.json"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_teacher_case16_validation_natural_error_"
    "pair_contract.py"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case16_validation_natural_error_"
    "pair.py"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case16_validation_natural_error_"
    "pair.sh"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_ADAPTER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "smoke_riser_case16_validation_natural_error_pair.py"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case16_validation_natural_error_"
    "pair.py"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case16_validation_natural_error_pair_route_cpu_v1/"
    "summary.json"
)
CASE16_VALIDATION_NATURAL_ERROR_PAIR_EXECUTION_SUMMARY = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case16_validation_pair_v2_rejected/summary.json"
)
CASE16_VALIDATION_DISPOSITION_AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/audit_case16_validation_disposition.py"
)
CASE16_VALIDATION_DISPOSITION_SUMMARY = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case16_validation_disposition_cpu_v1/summary.json"
)
CASE32_VALIDATION_SELECTION_BUILDER = (
    ROOT / "scripts/two_wheel_balance/prepare_case32_validation_selection.py"
)
CASE32_VALIDATION_SELECTION = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_selection_cpu_v1/selection.json"
)
CASE32_VALIDATION_READINESS_AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case32_validation_pair_readiness.py"
)
CASE32_VALIDATION_READINESS = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_pair_readiness_cpu_v1/summary.json"
)
CASE32_VALIDATION_PROFILE_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case32_validation_natural_error_profile.py"
)
CASE32_VALIDATION_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case32_validation_natural_error_"
    "profile_v1.json"
)
CASE32_VALIDATION_PROFILE_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_natural_error_profile_cpu_v1/"
    "proposal.json"
)
CASE32_VALIDATION_PAIR_CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case32_validation_natural_error_"
    "pair_contract_v1.json"
)
CASE32_VALIDATION_PAIR_BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_teacher_case32_validation_natural_error_"
    "pair_contract.py"
)
CASE32_VALIDATION_PAIR_VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case32_validation_natural_error_"
    "pair.py"
)
CASE32_VALIDATION_PAIR_WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case32_validation_natural_error_"
    "pair.sh"
)
CASE32_VALIDATION_PAIR_ADAPTER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "smoke_riser_case32_validation_natural_error_pair.py"
)
CASE32_VALIDATION_PAIR_FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case32_validation_natural_error_"
    "pair.py"
)
CASE32_VALIDATION_PAIR_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_natural_error_pair_route_cpu_v1/"
    "summary.json"
)
PENDING_CORRECTIVE_ROUTE_QUEUE_AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_pending_route_queue.py"
)
PENDING_CORRECTIVE_ROUTE_QUEUE_TEST = (
    ROOT
    / "tests/test_audit_model_based_corrective_pending_route_queue.py"
)
PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_pending_corrective_route_queue_cpu_v1/summary.json"
)
PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V2 = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_pending_corrective_route_queue_cpu_v2/summary.json"
)
PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V3 = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_pending_corrective_route_queue_cpu_v3/summary.json"
)
PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V4 = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_pending_corrective_route_queue_cpu_v4/summary.json"
)
CASE23_CONVERSION_EXECUTION_EVIDENCE_V3 = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case23_corrective_conversion_execution_cpu_v3/"
    "summary.json"
)
CASE23_CONVERSION_EXECUTION_EVIDENCE_V4 = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case23_corrective_conversion_execution_cpu_v4/"
    "summary.json"
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
VENDOR_SOURCE_RECONCILIATION = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_VENDOR_SOURCE_RECONCILIATION_20260724.json"
)
VENDOR_SOURCE_RECONCILIATION_AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_riser_vendor_source_reconciliation.py"
)
VENDOR_SOURCE_RECONCILIATION_EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_riser_vendor_source_reconciliation_v1/summary.json"
)


def _goal() -> dict:
    return json.loads(GOAL.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


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
    assert stage["status_as_of"] == "2026-07-28"
    assert stage["quality_qualified_exact_source_cases_available"] == 42
    assert stage["quality_qualified_cases_are_candidates_not_training_corpus"]
    assert stage["model_based_corrective_case_datasets_available"] == 4
    assert stage["model_based_corrective_training_corpus_cases_available"] == 0
    assert stage["corrected_reference_cases_available"] == 79
    assert "corrected_teacher_cases_available" not in stage
    assert "historical_quarantined_corrected_all79_stage" in stage
    assert "historical_quarantined_executed_residual_dataset_smoke" in stage


def test_planner_imitation_failure_and_residual_layer_are_explicit() -> None:
    refresh = _goal()["current_stage"]["status_refresh_20260723"]
    imitation = refresh["planner_imitation_bc"]
    residual = refresh["residual_dnn_admission_contract"]
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
    assert residual["architecture"] == (
        "model_based_shared_encoder_zero_initialized_residual_v1"
    )
    assert residual["observation_dimension"] == 65
    assert residual["base_observation_dimension"] == 26
    assert residual["lookahead_horizon_count"] == 3
    assert residual["lookahead_channel_count_per_horizon"] == 13
    assert residual["action_dimension"] == 3
    assert residual["parameter_count"] == 142019
    assert residual["zero_initialize_action_head"] is True
    assert residual["real_case30_observation_shape"] == [11411, 65]
    assert residual["real_case23_v4_observation_shape"] == [3273, 65]
    assert residual["architecture_contract_authoritative_cpu_commit"] == (
        "546fa7dfb54150832e6d81b0ad562b7ecdc0ef85"
    )
    assert residual["architecture_contract_authoritative_cpu_suite"] == (
        "1300_passed_12_skipped_2_warnings_in_170.69s"
    )


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
        CASE23_CAPTURE_V4_ARCHIVED_CONTRACT
    )
    assert corrective["case23_capture_v4_contract_git_blob_sha1"] == (
        _git_blob_sha1(CASE23_CAPTURE_V4_ARCHIVED_CONTRACT)
    )
    assert corrective["case23_capture_v4_contract_identity"] == (
        "immutable_archived_executed_contract"
    )
    assert corrective["case23_capture_v4_active_contract_sha256"] == _sha256(
        CASE23_CAPTURE_V4_CONTRACT
    )
    assert corrective["case23_capture_v4_active_contract_git_blob_sha1"] == (
        _git_blob_sha1(CASE23_CAPTURE_V4_CONTRACT)
    )
    assert corrective["case23_capture_v4_active_contract_identity"] == (
        "current_resealed_tokenless_contract"
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
    assert corrective[
        "case23_capture_v4_conversion_review_contract_sha256"
    ] == "58d358b546094c547a995c1fd336d9b750ce94bb734e481e28d8c65022d5f4a5"
    assert corrective[
        "case23_capture_v4_conversion_review_contract_identity"
    ] == "historical_contract_used_for_review_evidence"
    assert corrective[
        "case23_capture_v4_conversion_review_active_contract_sha256"
    ] == _sha256(CASE23_CONVERSION_REVIEW_CONTRACT)
    assert corrective[
        "case23_capture_v4_conversion_review_active_contract_git_blob_sha1"
    ] == _git_blob_sha1(CASE23_CONVERSION_REVIEW_CONTRACT)
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
    ] == "02657099559b3ffa544adff02cbb7a727d4753b7df70fa5c5a5749056326fb9a"
    assert corrective[
        "case23_capture_v4_conversion_execution_contract_git_blob_sha1"
    ] == "6ccbbc5c3153b38d5c33cebac026cb5dfee3c1ea"
    assert corrective[
        "case23_capture_v4_conversion_execution_contract_identity"
    ] == "historical_contract_used_for_cpu_preflight_evidence"
    assert corrective[
        "case23_capture_v4_conversion_execution_active_contract_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_CONTRACT)
    assert corrective[
        "case23_capture_v4_conversion_execution_active_contract_git_blob_sha1"
    ] == _git_blob_sha1(CASE23_CONVERSION_EXECUTION_CONTRACT)
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
    ] is True
    assert corrective[
        "case23_capture_v4_conversion_execution_runtime_commit"
    ] == "11fd27698955d277f4b926151bcca0cda2f4b27c"
    assert corrective[
        "case23_capture_v4_conversion_execution_evidence_commit"
    ] == "3052ca41a56a6cc4b5014787c8480aaded531224"
    assert corrective[
        "case23_capture_v4_conversion_execution_admission_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_V5_ADMISSION)
    assert corrective[
        "case23_capture_v4_conversion_execution_result_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_V5_RESULT)
    assert corrective[
        "case23_capture_v4_conversion_execution_dataset_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_V5_DATASET)
    assert corrective[
        "case23_capture_v4_conversion_execution_final_status_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_V5_FINAL)
    assert corrective[
        "case23_capture_v4_conversion_execution_recovery_audit_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_V5_RECOVERY)
    assert corrective[
        "case23_capture_v4_conversion_execution_sample_count"
    ] == 3273
    assert corrective[
        "case23_capture_v4_conversion_execution_observation_shape"
    ] == [3273, 65]
    assert corrective[
        "case23_capture_v4_conversion_execution_clipped_rows"
    ] == [0, 0, 0]
    assert corrective[
        "case23_capture_v4_conversion_execution_single_converter_invocation"
    ] is True
    assert corrective[
        "case23_capture_v4_conversion_execution_path_recovered_without_retry"
    ] is True
    assert corrective[
        "case23_capture_v4_conversion_execution_valid_for_case_merge"
    ] is True
    assert corrective[
        "case23_capture_v4_conversion_execution_merged_dataset_created"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_bc_authorized"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_ppo_authorized"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_training_started"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_valid_for_training"
    ] is False
    assert corrective[
        "case23_capture_v4_conversion_execution_authoritative_cpu_commit"
    ] == "224af37e3c6d870b0997a5ed67fec9e0096024cd"
    assert corrective[
        "case23_capture_v4_conversion_execution_authoritative_cpu_suite"
    ] == "1337_passed_12_skipped_2_warnings_in_175.33s"
    assert corrective["corrective_corpus_intake_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_corpus_intake_v5"
    )
    assert corrective["corrective_corpus_intake_implementation_commit"] == (
        "7a52a05d9cadb92b19190119ed93fabaee7af55e"
    )
    assert corrective["corrective_corpus_intake_script_sha256"] == _sha256(
        CORRECTIVE_CORPUS_INTAKE_SCRIPT
    )
    assert corrective["corrective_corpus_intake_summary_sha256"] == _sha256(
        CORRECTIVE_CORPUS_INTAKE_EVIDENCE
    )
    assert corrective["corrective_corpus_intake_mac_windows_byte_parity"] is True
    assert corrective["corrective_corpus_intake_converted_train_cases"] == [
        6,
        7,
        23,
        30,
    ]
    assert corrective["corrective_corpus_intake_converted_validation_cases"] == [8]
    assert corrective["corrective_corpus_intake_missing_train_case_count"] == 0
    assert corrective["corrective_corpus_intake_missing_validation_case_count"] == 1
    assert corrective["corrective_corpus_intake_pending_minimum_train_cases"] == []
    assert corrective["corrective_corpus_intake_pending_validation_cases"] == [16]
    assert corrective["corrective_corpus_intake_manifest_ready"] is False
    assert corrective["corrective_corpus_intake_merge_authorized"] is False
    assert corrective["corrective_corpus_intake_authoritative_cpu_commit"] == (
        "92f4b413fbb5c36534d2066644def0a177fb800c"
    )
    assert corrective["corrective_corpus_intake_authoritative_cpu_suite"] == (
        "51_passed_2_warnings_in_35.82s"
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
    assert corrective[
        "case2_natural_error_pair_contract_sha256"
    ] == "1a5b9190bf656cd52b973193efd90bd60aa316dda2b67c41f93f21376626872c"
    assert corrective[
        "case2_natural_error_pair_contract_identity"
    ] == "historical_cpu_route_contract"
    assert corrective[
        "case2_natural_error_pair_active_contract_sha256"
    ] == _sha256(
        CASE2_NATURAL_ERROR_PAIR_CONTRACT
    )
    assert corrective[
        "case2_natural_error_pair_active_contract_git_blob_sha1"
    ] == _git_blob_sha1(CASE2_NATURAL_ERROR_PAIR_CONTRACT)
    assert corrective["case2_natural_error_pair_adapter_sha256"] == _sha256(
        CASE2_NATURAL_ERROR_PAIR_ADAPTER
    )
    assert corrective["case2_natural_error_pair_identity_count"] == 19
    assert corrective["case2_natural_error_pair_active_identity_count"] == 20
    assert corrective["corrective_projection_evidence_engine_sha256"] == (
        _sha256(PROJECTION_EVIDENCE_ENGINE)
    )
    assert corrective["case2_projection_reclassification_sha256"] == (
        _sha256(PROJECTION_REPAIR_CASE2)
    )
    assert corrective[
        "case2_projection_reclassification_evidence_passed"
    ] is True
    assert corrective[
        "case2_projection_reclassification_dynamic_pair_completed"
    ] is True
    assert corrective[
        "case2_projection_reclassification_corrective_target_admitted"
    ] is False
    assert corrective["corrective_projection_evidence_runtime_started"] is False
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
    assert corrective["case2_natural_error_authorization_token_issued"] is True
    assert (
        corrective["case2_natural_error_authorization_token_consumed"] is True
    )
    assert corrective["case2_natural_error_runtime_authorized"] is False
    assert corrective["case2_natural_error_label_capture_authorized"] is False
    assert corrective["case2_natural_error_training_authorized"] is False
    case2_execution = json.loads(
        CASE2_NATURAL_ERROR_PAIR_EXECUTION.read_text()
    )
    assert case2_execution["baseline_dynamic_quality_passed"] is True
    assert case2_execution["candidate_dynamic_quality_passed"] is True
    assert case2_execution["absolute_improvement_gate_passed"] is False
    assert case2_execution["relative_improvement_gate_passed"] is False
    assert (
        corrective["case2_pair_execution_final_status_sha256"]
        == _sha256(CASE2_NATURAL_ERROR_PAIR_EXECUTION.parent / "final_status.json")
    )
    assert corrective["case2_pair_execution_capture_eligible"] is False
    assert corrective["case2_pair_execution_dataset_created"] is False
    assert corrective["case2_pair_execution_training_started"] is False
    assert corrective[
        "case2_pair_execution_authoritative_windows_cpu_suite"
    ] == "1362_passed_12_skipped_2_warnings_in_188.10s"
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
    assert corrective[
        "case7_pair_profile_proposal_sha256"
    ] == "c279d585600b7a1c2d80430dd1618785444335d12edf535c2903c268c38061cb"
    assert corrective[
        "case7_pair_profile_proposal_identity"
    ] == "historical_cpu_profile_proposal"
    assert corrective["case7_pair_profile_active_proposal_sha256"] == _sha256(
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
    assert corrective["case7_pair_route_schema"] == (
        "cinebotrl_two_wheel_riser_corrective_teacher_case7_pair_contract_v1"
    )
    assert corrective[
        "case7_pair_route_contract_sha256"
    ] == "406b38800ca58b7a5a0e579d8a9c7bb1031bed46e5ef4863b6facf69a046ef40"
    assert corrective[
        "case7_pair_route_contract_identity"
    ] == "historical_cpu_route_contract"
    assert corrective["case7_pair_route_active_contract_sha256"] == _sha256(
        CASE7_PAIR_ROUTE_CONTRACT
    )
    assert corrective[
        "case7_pair_route_active_contract_git_blob_sha1"
    ] == _git_blob_sha1(CASE7_PAIR_ROUTE_CONTRACT)
    assert corrective["case7_pair_route_validator_sha256"] == _sha256(
        CASE7_PAIR_ROUTE_VALIDATOR
    )
    assert corrective["case7_pair_route_wrapper_sha256"] == _sha256(
        CASE7_PAIR_ROUTE_WRAPPER
    )
    assert corrective["case7_pair_route_finalizer_sha256"] == _sha256(
        CASE7_PAIR_ROUTE_FINALIZER
    )
    assert corrective["case7_pair_route_preflight_sha256"] == _sha256(
        CASE7_PAIR_ROUTE_EVIDENCE
    )
    assert corrective["case7_pair_route_identity_count"] == 18
    assert corrective["case7_pair_route_reset_seed"] == (
        corrective["case7_pair_route_configuration_seed"] + 7
    )
    assert corrective[
        "case7_pair_route_same_plan_seed_physics_and_perturbation"
    ] is True
    assert corrective["case7_pair_route_cpu_preflight_passed"] is True
    assert corrective["case7_pair_route_runtime_route_contract_ready"] is True
    assert corrective["case7_pair_route_execution_route_complete"] is True
    assert corrective["case7_pair_route_authorization_token_issued"] is False
    assert corrective["case7_pair_route_runtime_authorized"] is False
    assert corrective["case7_pair_route_gpu_launch_authorized"] is False
    assert corrective["case7_pair_route_label_capture_authorized"] is False
    assert corrective["case7_pair_route_dataset_creation_authorized"] is False
    assert corrective["case7_pair_route_training_authorized"] is False
    case7_route = json.loads(CASE7_PAIR_ROUTE_EVIDENCE.read_text())
    assert case7_route["passed"] is True
    assert case7_route["cpu_contract_ready"] is True
    assert case7_route["execution_route_complete"] is True
    assert len(case7_route["identities"]) == 18
    assert all(case7_route["checks"].values())
    assert case7_route["authorization_token_issued"] is False
    assert case7_route["runtime_authorized"] is False
    assert case7_route["gpu_launch_authorized"] is False
    assert case7_route["label_capture_authorized"] is False
    assert case7_route["dataset_creation_authorized"] is False
    assert case7_route["training_started"] is False
    case7_execution = json.loads(CASE7_PAIR_EXECUTION_FINAL.read_text())
    case7_projection = json.loads(CASE7_PAIR_EXECUTION_PROJECTION.read_text())
    assert (
        corrective["case7_pair_execution_final_status_sha256"]
        == _sha256(CASE7_PAIR_EXECUTION_FINAL)
    )
    assert corrective["case7_pair_execution_evidence_commit"] == (
        "69516b4273ec9b363f328eade80835b2422f748d"
    )
    assert (
        corrective["case7_pair_execution_projection_audit_sha256"]
        == _sha256(CASE7_PAIR_EXECUTION_PROJECTION)
    )
    assert case7_execution["dynamic_pair_completed"] is True
    assert case7_execution["corrective_target_admission_passed"] is True
    assert case7_execution["label_capture_authorized"] is False
    assert case7_execution["dataset_created"] is False
    assert case7_execution["training_started"] is False
    assert case7_projection["passed"] is True
    assert case7_projection["candidate"][
        "projection_affected_sample_count"
    ] == 9
    assert corrective["case7_pair_execution_capture_eligible"] is True
    assert corrective["case7_pair_execution_label_capture_authorized"] is False
    assert corrective["case7_pair_execution_training_started"] is False
    assert corrective[
        "case7_pair_execution_authoritative_windows_cpu_suite"
    ] == "1375_passed_12_skipped_2_warnings_in_222.39s"
    case7_capture_route = json.loads(
        CASE7_CAPTURE_ROUTE_EVIDENCE.read_text()
    )
    case7_capture_preflight = json.loads(
        CASE7_CAPTURE_ROUTE_PREFLIGHT.read_text()
    )
    assert corrective["case7_corrective_capture_route_schema"] == (
        "cinebotrl_two_wheel_riser_corrective_teacher_capture_contract_v2"
    )
    assert corrective[
        "case7_corrective_capture_route_implementation_commit"
    ] == "93d9b60eea2a4fa4bdd9748e5e4864d52f456514"
    assert corrective["case7_corrective_capture_route_reseal_commit"] == (
        "2743c64f8c118dd3a549968fb603ae64ec9c59d1"
    )
    assert corrective[
        "case7_corrective_capture_route_shared_resource_guard_commit"
    ] == "2b4bd88791a2e183b2a9bd4b9d4a0e8374b49fe1"
    assert corrective[
        "case7_corrective_capture_route_resource_finalizer_seal_commit"
    ] == "0f2de2b8175e59395cd61b45d37a49a071ad81e5"
    assert corrective[
        "case7_corrective_capture_route_resource_finalizer_seal_summary_sha256"
    ] == _sha256(CASE7_RESOURCE_FINALIZER_SEAL_SUMMARY)
    assert corrective["case7_corrective_capture_route_evidence_schema"] == (
        "cinebotrl_two_wheel_riser_case7_corrective_capture_route_cpu_evidence_v2"
    )
    assert corrective[
        "case7_corrective_capture_route_summary_sha256"
    ] == _sha256(CASE7_CAPTURE_ROUTE_EVIDENCE)
    assert corrective[
        "case7_corrective_capture_route_contract_sha256"
    ] == _sha256(CASE7_CAPTURE_ROUTE_CONTRACT)
    assert corrective[
        "case7_corrective_capture_route_validator_sha256"
    ] == _sha256(CASE7_CAPTURE_ROUTE_VALIDATOR)
    assert corrective[
        "case7_corrective_capture_route_wrapper_sha256"
    ] == _sha256(CASE7_CAPTURE_ROUTE_WRAPPER)
    assert corrective[
        "case7_corrective_capture_route_shared_resource_guard_sha256"
    ] == _sha256(SHARED_RESOURCE_GUARD)
    assert corrective[
        "case7_corrective_capture_route_shared_resource_monitor_sha256"
    ] == _sha256(CASE7_CAPTURE_RESOURCE_MONITOR)
    assert corrective[
        "case7_corrective_capture_route_finalizer_sha256"
    ] == _sha256(CASE7_CAPTURE_ROUTE_FINALIZER)
    assert corrective[
        "case7_corrective_capture_route_preflight_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_PREFLIGHT)
    assert corrective[
        "case7_corrective_capture_route_historical_preflight_sha256"
    ] == _sha256(CASE7_CAPTURE_ROUTE_PREFLIGHT)
    assert corrective["case7_corrective_capture_route_identity_count"] == 21
    assert corrective[
        "case7_corrective_capture_route_projection_evidence_passed"
    ] is True
    assert corrective[
        "case7_corrective_capture_route_cpu_preflight_passed"
    ] is True
    assert corrective[
        "case7_corrective_capture_route_broad_local_cpu_suite"
    ] == "1368_passed_16_failed_27_warnings_in_89.74s"
    assert corrective[
        "case7_corrective_capture_route_full_windows_cpu_suite_deferred"
    ] is True
    assert corrective[
        "case7_corrective_capture_route_full_windows_cpu_suite_deferred_reason"
    ] == "active_shared_windows_cad_and_memory_pressure"
    assert corrective[
        "case7_corrective_capture_route_authorization_token_issued"
    ] is True
    assert corrective[
        "case7_corrective_capture_route_authorization_token_consumed"
    ] is True
    assert corrective[
        "case7_corrective_capture_route_runtime_authorized"
    ] is False
    assert corrective[
        "case7_corrective_capture_route_label_capture_authorized"
    ] is False
    assert corrective["case7_corrective_capture_route_training_started"] is False
    assert case7_capture_route["passed"] is True
    assert case7_capture_route["runtime_namespace_created"] is False
    assert case7_capture_preflight["passed"] is True
    assert case7_capture_preflight["cpu_contract_ready"] is True
    assert len(case7_capture_preflight["identities"]) == 19
    assert case7_capture_preflight["runtime_authorized"] is False
    assert case7_capture_preflight["label_capture_authorized"] is False
    assert case7_capture_preflight["dataset_creation_authorized"] is False
    assert case7_capture_preflight["training_started"] is False
    active_preflight = json.loads(SHARED_RESOURCE_PREFLIGHT.read_text())
    assert active_preflight["passed"] is True
    assert active_preflight["cpu_contract_ready"] is True
    assert len(active_preflight["identities"]) == 21
    assert active_preflight["runtime_authorized"] is False
    assert active_preflight["label_capture_authorized"] is False
    assert active_preflight["training_started"] is False
    sealed_preflight = json.loads(
        CASE7_RESOURCE_FINALIZER_SEAL_PREFLIGHT.read_text()
    )
    sealed_summary = json.loads(
        CASE7_RESOURCE_FINALIZER_SEAL_SUMMARY.read_text()
    )
    sealed_live = json.loads(CASE7_RESOURCE_FINALIZER_SEAL_LIVE.read_text())
    sealed_commands = json.loads(
        CASE7_RESOURCE_FINALIZER_SEAL_COMMANDS.read_text()
    )
    assert sealed_summary["passed"] is True
    assert sealed_summary["implementation_commit"] == (
        "0f2de2b8175e59395cd61b45d37a49a071ad81e5"
    )
    assert all(sealed_summary["checks"].values())
    assert sealed_preflight["passed"] is True
    assert len(sealed_preflight["identities"]) == 20
    assert sealed_preflight["runtime_authorized"] is False
    assert sealed_live["passed"] is False
    assert sealed_live["runtime_started"] is False
    assert sealed_live["authorization_consumed"] is False
    assert sealed_commands["passed"] is True
    case7_command = next(
        route for route in sealed_commands["routes"] if route["case"] == 7
    )
    assert case7_command["current_command_compatible"] is True
    assert case7_command["mismatches"] == []
    capture_v2 = json.loads(CASE7_CAPTURE_V2_SUMMARY.read_text())
    capture_v2_final = json.loads(CASE7_CAPTURE_V2_FINAL.read_text())
    assert corrective["case7_corrective_capture_v2_runtime_commit"] == (
        "d0365653571d50523584e80e2ec1943febdfe6d4"
    )
    assert corrective[
        "case7_corrective_capture_v2_evidence_summary_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_SUMMARY)
    assert corrective[
        "case7_corrective_capture_v2_gate_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_GATE)
    assert corrective[
        "case7_corrective_capture_v2_final_status_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_FINAL)
    assert corrective[
        "case7_corrective_capture_v2_capture_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_ARCHIVE)
    assert corrective[
        "case7_corrective_capture_v2_resource_monitor_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_RESOURCE_MONITOR)
    assert capture_v2["passed"] is True
    assert capture_v2["dynamic_quality_passed"] is True
    assert capture_v2["capture_sample_count"] == 6597
    assert capture_v2["resource_monitor_sample_count"] == 46
    assert capture_v2["capture_admitted_for_dataset_conversion"] is True
    assert capture_v2["conversion_authorized"] is False
    assert capture_v2["training_started"] is False
    assert capture_v2_final["passed"] is True
    assert capture_v2_final["shared_windows_resource_monitor_passed"] is True
    assert (
        capture_v2_final["capture_admitted_for_dataset_conversion"] is True
    )
    assert capture_v2_final["training_started"] is False
    assert capture_v2_final["valid_for_training"] is False
    generic_conversion = json.loads(
        GENERIC_CORRECTIVE_CONVERSION_SUMMARY.read_text()
    )
    assert corrective["generic_corrective_conversion_proposal_schema"] == (
        "cinebotrl_two_wheel_riser_corrective_conversion_proposal_v1"
    )
    assert corrective[
        "generic_corrective_conversion_proposal_implementation_commit"
    ] == "8d394ed13cd4724868f167dda8b58f613b2109f8"
    assert corrective[
        "generic_corrective_conversion_proposal_preparer_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_PREPARER)
    assert corrective[
        "generic_corrective_conversion_proposal_preparer_git_blob_sha1"
    ] == _git_blob_sha1(GENERIC_CORRECTIVE_CONVERSION_PREPARER)
    assert corrective[
        "generic_corrective_conversion_proposal_summary_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_SUMMARY)
    assert corrective[
        "generic_corrective_conversion_proposal_cases"
    ] == [6, 23, 30]
    for case, expected_sha in zip(
        [6, 23, 30],
        [
            corrective[
                "generic_corrective_conversion_proposal_case6_sha256"
            ],
            corrective[
                "generic_corrective_conversion_proposal_case23_sha256"
            ],
            corrective[
                "generic_corrective_conversion_proposal_case30_sha256"
            ],
        ],
        strict=True,
    ):
        report = (
            GENERIC_CORRECTIVE_CONVERSION_EVIDENCE
            / f"case_{case:04d}.json"
        )
        assert expected_sha == _sha256(report)
        payload = json.loads(report.read_text())
        assert payload["passed"] is True
        assert payload["case"] == case
        assert payload["conversion_execution_implemented"] is False
        assert payload["conversion_authorized"] is False
        assert payload["output_created"] is False
        assert payload["merged_dataset_created"] is False
        assert payload["bc_authorized"] is False
        assert payload["ppo_authorized"] is False
        assert payload["training_started"] is False
    assert generic_conversion["passed"] is True
    assert generic_conversion["case_count"] == 3
    assert generic_conversion["total_sample_count"] == 22617
    assert generic_conversion["observation_dimension"] == 65
    assert generic_conversion["action_dimension"] == 3
    assert generic_conversion["mac_windows_byte_parity"] is True
    assert generic_conversion["conversion_execution_implemented"] is False
    assert generic_conversion["conversion_authorized"] is False
    assert generic_conversion["output_created"] is False
    assert generic_conversion["training_started"] is False
    assert corrective[
        "generic_corrective_conversion_proposal_authoritative_windows_cpu_suite"
    ] == "1396_passed_12_skipped_2_warnings_in_214.66s"
    assert corrective[
        "generic_corrective_conversion_proposal_authoritative_windows_cpu_commit"
    ] == "34f83bcac0a78460b5cc8409624a9495deb08b5e"
    generic_execution = json.loads(
        GENERIC_CORRECTIVE_CONVERSION_EXECUTION_SUMMARY.read_text()
    )
    assert corrective["generic_corrective_conversion_execution_schema"] == (
        "cinebotrl_two_wheel_riser_generic_corrective_conversion_execution_"
        "contract_v2"
    )
    assert corrective[
        "generic_corrective_conversion_execution_contract_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_EXECUTION_CONTRACT)
    assert corrective[
        "generic_corrective_conversion_execution_contract_git_blob_sha1"
    ] == _git_blob_sha1(
        GENERIC_CORRECTIVE_CONVERSION_EXECUTION_CONTRACT
    )
    assert corrective[
        "generic_corrective_conversion_execution_validator_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_EXECUTION_VALIDATOR)
    assert corrective[
        "generic_corrective_conversion_execution_wrapper_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_EXECUTION_WRAPPER)
    assert corrective[
        "generic_corrective_conversion_execution_finalizer_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_EXECUTION_FINALIZER)
    assert corrective[
        "generic_corrective_conversion_execution_builder_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_EXECUTION_BUILDER)
    assert corrective[
        "generic_corrective_conversion_execution_summary_sha256"
    ] == _sha256(GENERIC_CORRECTIVE_CONVERSION_EXECUTION_SUMMARY)
    assert corrective[
        "generic_corrective_conversion_execution_preflight_cases"
    ] == [6, 23, 30]
    for case, field in [
        (6, "generic_corrective_conversion_execution_case6_preflight_sha256"),
        (
            23,
            "generic_corrective_conversion_execution_case23_preflight_sha256",
        ),
        (
            30,
            "generic_corrective_conversion_execution_case30_preflight_sha256",
        ),
    ]:
        report = (
            GENERIC_CORRECTIVE_CONVERSION_EXECUTION_EVIDENCE
            / f"case_{case:04d}_preflight.json"
        )
        assert corrective[field] == _sha256(report)
        payload = json.loads(report.read_text())
        assert payload["passed"] is True
        assert payload["cpu_contract_ready"] is True
        assert payload["conversion_authorized"] is False
        assert payload["output_created"] is False
        assert payload["training_started"] is False
    assert generic_execution["passed"] is True
    assert generic_execution["preflight_case_count"] == 3
    assert generic_execution["preflight_total_samples"] == 22617
    assert generic_execution["mac_windows_byte_parity"] is True
    assert generic_execution["wrapper_execute_without_token_exit_code"] == 4
    assert generic_execution[
        "wrapper_execute_without_token_namespace_created"
    ] is False
    assert generic_execution["conversion_execution_implemented"] is True
    assert generic_execution["authorization_token_issued"] is False
    assert generic_execution["conversion_authorized"] is False
    assert generic_execution["output_created"] is False
    assert generic_execution["training_started"] is False
    assert corrective["generic_corrective_conversion_execution_implemented"] is True
    assert corrective["generic_corrective_conversion_authorized"] is False
    assert corrective["generic_corrective_conversion_output_created"] is False
    assert corrective["generic_corrective_conversion_training_started"] is False
    assert corrective["case8_validation_pair_readiness_schema"] == (
        "cinebotrl_two_wheel_riser_case8_validation_pair_readiness_cpu_v1"
    )
    assert corrective["case8_validation_pair_readiness_script_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_READINESS_SCRIPT)
    )
    assert corrective[
        "case8_validation_pair_readiness_script_git_blob_sha1"
    ] == "cf44aa738f042a7f15256f999f70f7669c4f4d04"
    assert corrective["case8_validation_pair_readiness_summary_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_READINESS_EVIDENCE)
    )
    assert corrective[
        "case8_validation_pair_readiness_mac_windows_byte_parity"
    ] is True
    assert corrective[
        "case8_validation_pair_readiness_authoritative_cpu_suite"
    ] == "1232_passed_12_skipped_2_warnings_in_151.02s"
    assert corrective["case8_validation_pair_readiness_split"] == "validation"
    assert corrective["case8_validation_pair_readiness_source_states"] == 663
    assert corrective["case8_validation_pair_readiness_transitions"] == 662
    assert corrective[
        "case8_validation_pair_readiness_source_duration_s"
    ] == 12.940941
    assert corrective[
        "case8_validation_pair_readiness_execution_duration_s"
    ] == 18.1173174
    assert corrective[
        "case8_validation_pair_readiness_camera_height_range_m"
    ] == [0.6, 1.605452]
    assert corrective[
        "case8_validation_pair_readiness_low_motion_window_count"
    ] == 4
    assert (
        corrective[
            "case8_validation_pair_readiness_longest_low_motion_window_s"
        ]
        > 3.4
    )
    assert corrective[
        "case8_validation_pair_readiness_case_specific_profile_required"
    ] is True
    assert corrective[
        "case8_validation_pair_readiness_train_profile_reuse_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_readiness_pair_profile_cpu_ready"
    ] is False
    assert corrective[
        "case8_validation_pair_readiness_runtime_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_readiness_label_capture_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_readiness_training_authorized"
    ] is False
    case8 = json.loads(CASE8_VALIDATION_PAIR_READINESS_EVIDENCE.read_text())
    assert case8["passed"] is True
    assert case8["split"] == "validation"
    assert case8["case_specific_profile_required"] is True
    assert case8["profile_window_contract"]["bounded_window_found"] is True
    assert len(case8["profile_window_contract"]["windows"]) == 4
    assert case8["case30_profile_reuse_authorized"] is False
    assert case8["case23_profile_reuse_authorized"] is False
    assert case8["case6_profile_reuse_authorized"] is False
    assert case8["case2_profile_reuse_authorized"] is False
    assert case8["case7_profile_reuse_authorized"] is False
    assert case8["runtime_authorized"] is False
    assert case8["gpu_launch_authorized"] is False
    assert case8["label_capture_authorized"] is False
    assert case8["dataset_conversion_authorized"] is False
    assert case8["dataset_merge_authorized"] is False
    assert case8["bc_authorized"] is False
    assert case8["ppo_authorized"] is False
    assert case8["training_started"] is False
    assert case8["valid_for_training"] is False
    assert corrective["case8_validation_pair_profile_schema"] == (
        "cinebotrl_two_wheel_riser_case8_validation_pair_profile_"
        "proposal_cpu_v1"
    )
    assert corrective["case8_validation_pair_profile_builder_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_PROFILE_BUILDER)
    )
    assert corrective[
        "case8_validation_pair_profile_builder_git_blob_sha1"
    ] == "6cb1b96cc1b209b27276c3d8d2b3d892baa3d5e9"
    assert corrective[
        "case8_validation_corrective_profile_sha256"
    ] == _sha256(CASE8_VALIDATION_CORRECTIVE_PROFILE)
    assert corrective["case8_validation_wrench_profile_sha256"] == _sha256(
        CASE8_VALIDATION_WRENCH_PROFILE
    )
    assert corrective[
        "case8_validation_pair_profile_proposal_sha256"
    ] == "1a74b5d362e8c92c76b18e2c2ea0a9e7d67745504ea57fd27099cbc688dfb829"
    assert corrective[
        "case8_validation_pair_profile_proposal_identity"
    ] == "historical_cpu_profile_proposal"
    assert corrective[
        "case8_validation_pair_profile_active_proposal_sha256"
    ] == _sha256(CASE8_VALIDATION_PAIR_PROFILE_PROPOSAL)
    assert corrective[
        "case8_validation_pair_profile_mac_windows_byte_parity"
    ] is True
    assert corrective[
        "case8_validation_pair_profile_authoritative_cpu_suite"
    ] == "1244_passed_12_skipped_2_warnings_in_152.91s"
    assert corrective["case8_validation_pair_profile_split"] == "validation"
    assert corrective[
        "case8_validation_pair_profile_envelope_retention_fraction"
    ] == 0.4
    assert corrective[
        "case8_validation_pair_profile_slew_horizon_s"
    ] == 0.4
    assert corrective[
        "case8_validation_pair_profile_pulse_force_n"
    ] == 18.0
    assert corrective[
        "case8_validation_pair_profile_pulse_duration_s"
    ] == 0.1
    assert corrective[
        "case8_validation_pair_profile_recovery_tail_s"
    ] > 15.0
    assert corrective[
        "case8_validation_pair_profile_effective_normalized_action_abs_max"
    ] < 0.31
    assert corrective[
        "case8_validation_pair_profile_pulse_window_fully_unclipped"
    ] is True
    assert corrective[
        "case8_validation_pair_profile_train_profile_reuse_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_profile_cpu_ready"
    ] is True
    assert corrective[
        "case8_validation_pair_profile_runtime_route_implemented"
    ] is False
    assert corrective[
        "case8_validation_pair_profile_authorization_token_issued"
    ] is False
    assert corrective[
        "case8_validation_pair_profile_runtime_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_profile_label_capture_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_profile_training_authorized"
    ] is False
    case8_profile = json.loads(
        CASE8_VALIDATION_PAIR_PROFILE_PROPOSAL.read_text()
    )
    assert case8_profile["passed"] is True
    assert case8_profile["split"] == "validation"
    assert all(case8_profile["validation_profile_checks"].values())
    assert case8_profile["train_profile_reuse_authorized"] is False
    assert case8_profile["pair_profile_cpu_ready"] is True
    assert case8_profile["runtime_route_implemented"] is False
    assert case8_profile["authorization_token_issued"] is False
    assert case8_profile["runtime_authorized"] is False
    assert case8_profile["gpu_launch_authorized"] is False
    assert case8_profile["label_capture_authorized"] is False
    assert case8_profile["dataset_conversion_authorized"] is False
    assert case8_profile["dataset_merge_authorized"] is False
    assert case8_profile["bc_authorized"] is False
    assert case8_profile["ppo_authorized"] is False
    assert case8_profile["training_started"] is False
    assert case8_profile["valid_for_training"] is False
    assert corrective[
        "case8_validation_pair_route_contract_sha256"
    ] == "ada5b6172fcee053daeb74326e4546246330fe4914c81d590ba3d59d3f750517"
    assert corrective[
        "case8_validation_pair_route_contract_identity"
    ] == "historical_cpu_route_contract"
    assert corrective[
        "case8_validation_pair_route_active_contract_sha256"
    ] == _sha256(CASE8_VALIDATION_PAIR_ROUTE_CONTRACT)
    assert corrective[
        "case8_validation_pair_route_active_contract_git_blob_sha1"
    ] == _git_blob_sha1(CASE8_VALIDATION_PAIR_ROUTE_CONTRACT)
    assert corrective["case8_validation_pair_route_validator_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_ROUTE_VALIDATOR)
    )
    assert corrective["case8_validation_pair_route_wrapper_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_ROUTE_WRAPPER)
    )
    assert corrective["case8_validation_pair_route_finalizer_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_ROUTE_FINALIZER)
    )
    assert corrective["case8_validation_assessment_sha256"] == _sha256(
        CASE8_VALIDATION_ASSESSMENT
    )
    assert corrective["case8_validation_pair_route_preflight_sha256"] == (
        _sha256(CASE8_VALIDATION_PAIR_ROUTE_EVIDENCE)
    )
    assert corrective["case8_validation_pair_route_identity_count"] == 21
    assert corrective["case8_validation_pair_route_reset_seed"] == (
        corrective["case8_validation_pair_route_configuration_seed"] + 8
    )
    assert corrective[
        "case8_validation_pair_route_same_plan_seed_physics_and_perturbation"
    ] is True
    assert corrective[
        "case8_validation_pair_route_cpu_preflight_passed"
    ] is True
    assert corrective[
        "case8_validation_pair_route_contract_ready"
    ] is True
    assert corrective[
        "case8_validation_pair_route_execution_route_complete"
    ] is True
    assert corrective[
        "case8_validation_pair_route_authorization_token_issued"
    ] is False
    assert corrective[
        "case8_validation_pair_route_runtime_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_route_gpu_launch_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_route_teacher_admission_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_route_label_capture_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_route_dataset_creation_authorized"
    ] is False
    assert corrective[
        "case8_validation_pair_route_training_authorized"
    ] is False
    case8_route = json.loads(CASE8_VALIDATION_PAIR_ROUTE_EVIDENCE.read_text())
    assert case8_route["passed"] is True
    assert case8_route["case"] == 8
    assert case8_route["split"] == "validation"
    assert len(case8_route["identities"]) == 21
    assert all(case8_route["document_checks"].values())
    assert all(case8_route["checks"].values())
    assert case8_route["runtime_authorized"] is False
    assert case8_route["gpu_launch_authorized"] is False
    assert case8_route["teacher_admission_authorized"] is False
    assert case8_route["label_capture_authorized"] is False
    assert case8_route["dataset_creation_authorized"] is False
    assert case8_route["training_started"] is False
    assert case8_route["valid_for_training"] is False
    case8_execution = json.loads(
        CASE8_VALIDATION_PAIR_EXECUTION_SUMMARY.read_text()
    )
    assert corrective["case8_validation_pair_executed"] is True
    assert corrective["case8_validation_pair_passed"] is True
    assert corrective[
        "case8_validation_pair_evidence_summary_sha256"
    ] == _sha256(CASE8_VALIDATION_PAIR_EXECUTION_SUMMARY)
    assert corrective[
        "case8_validation_pair_authorization_token_consumed"
    ] is True
    assert corrective[
        "case8_validation_pair_capture_authorized"
    ] is False
    assert corrective["case8_validation_pair_dataset_created"] is False
    assert corrective["case8_validation_pair_valid_for_training"] is False
    assert case8_execution["passed"] is True
    assert case8_execution["validation_pair_passed"] is True
    assert case8_execution["split"] == "validation"
    assert case8_execution["candidate_position_error_p95_m"] < (
        case8_execution["baseline_position_error_p95_m"]
    )
    assert case8_execution["position_p95_relative_improvement"] > 0.04
    assert case8_execution["label_capture_authorized"] is False
    assert case8_execution["dataset_created"] is False
    assert case8_execution["training_started"] is False
    case8_capture = json.loads(CASE8_VALIDATION_CAPTURE_SUMMARY.read_text())
    assert corrective["case8_validation_capture_completed"] is True
    assert corrective["case8_validation_capture_passed"] is True
    assert corrective[
        "case8_validation_capture_evidence_summary_sha256"
    ] == _sha256(CASE8_VALIDATION_CAPTURE_SUMMARY)
    assert corrective["case8_validation_capture_sample_count"] == 6607
    assert corrective[
        "case8_validation_capture_valid_for_conversion"
    ] is True
    assert corrective[
        "case8_validation_capture_conversion_authorized"
    ] is False
    assert corrective["case8_validation_capture_valid_for_training"] is False
    assert case8_capture["passed"] is True
    assert case8_capture["split"] == "validation"
    assert case8_capture["capture_sample_count"] == 6607
    assert case8_capture["capture_admitted_for_dataset_conversion"] is True
    assert case8_capture["conversion_authorized"] is False
    assert case8_capture["normalized_training_dataset_created"] is False
    assert case8_capture["training_started"] is False
    case8_conversion = json.loads(
        CASE8_VALIDATION_CONVERSION_SUMMARY.read_text()
    )
    assert corrective["case8_validation_conversion_completed"] is True
    assert corrective[
        "case8_validation_conversion_evidence_summary_sha256"
    ] == _sha256(CASE8_VALIDATION_CONVERSION_SUMMARY)
    assert corrective["case8_validation_conversion_sample_count"] == 6607
    assert corrective["case8_validation_conversion_token_consumed"] is True
    assert corrective[
        "case8_validation_conversion_valid_for_case_merge"
    ] is True
    assert corrective[
        "case8_validation_conversion_valid_for_training"
    ] is False
    assert case8_conversion["passed"] is True
    assert case8_conversion["split"] == "validation"
    assert case8_conversion["sample_count"] == 6607
    assert case8_conversion["valid_for_case_merge"] is True
    assert case8_conversion["merged_dataset_created"] is False
    assert case8_conversion["training_started"] is False
    assert corrective["case16_validation_pair_readiness_script_sha256"] == (
        _sha256(CASE16_VALIDATION_PAIR_READINESS_SCRIPT)
    )
    assert corrective[
        "case16_validation_pair_readiness_script_git_blob_sha1"
    ] == "8c294399cd5e457801e4aee6d498dcae554cdca2"
    assert corrective["case16_validation_pair_readiness_summary_sha256"] == (
        _sha256(CASE16_VALIDATION_PAIR_READINESS_EVIDENCE)
    )
    assert corrective[
        "case16_validation_pair_readiness_mac_windows_byte_parity"
    ] is True
    assert corrective["case16_validation_pair_readiness_split"] == "validation"
    assert corrective["case16_validation_pair_readiness_source_states"] == 896
    assert corrective["case16_validation_pair_readiness_transitions"] == 895
    assert corrective[
        "case16_validation_pair_readiness_low_motion_window_count"
    ] == 0
    assert corrective[
        "case16_validation_pair_readiness_safe_window_absent_requires_structural_profile"
    ] is True
    assert corrective[
        "case16_validation_pair_readiness_case_specific_profile_required"
    ] is True
    assert corrective[
        "case16_validation_pair_readiness_external_wrench_profile_suitable"
    ] is False
    assert corrective[
        "case16_validation_pair_readiness_pair_profile_cpu_ready"
    ] is False
    assert corrective[
        "case16_validation_pair_readiness_runtime_authorized"
    ] is False
    assert corrective[
        "case16_validation_pair_readiness_label_capture_authorized"
    ] is False
    assert corrective[
        "case16_validation_pair_readiness_training_authorized"
    ] is False
    case16 = json.loads(
        CASE16_VALIDATION_PAIR_READINESS_EVIDENCE.read_text()
    )
    assert case16["passed"] is True
    assert case16["case"] == 16
    assert case16["split"] == "validation"
    assert all(case16["selection_checks"].values())
    assert all(case16["plan_checks"].values())
    assert all(case16["gate_checks"].values())
    assert case16["profile_window_contract"]["windows"] == []
    assert case16[
        "safe_window_absent_requires_structural_profile"
    ] is True
    assert case16["pair_profile_cpu_ready"] is False
    assert case16["runtime_authorized"] is False
    assert case16["gpu_launch_authorized"] is False
    assert case16["label_capture_authorized"] is False
    assert case16["dataset_conversion_authorized"] is False
    assert case16["dataset_merge_authorized"] is False
    assert case16["bc_authorized"] is False
    assert case16["ppo_authorized"] is False
    assert case16["training_started"] is False
    assert case16["valid_for_training"] is False
    assert corrective[
        "case16_validation_natural_error_profile_builder_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PROFILE_BUILDER)
    assert corrective[
        "case16_validation_natural_error_profile_builder_git_blob_sha1"
    ] == "0a1964bc25a4f2d4e10a2f0b7689c36ca7c708e4"
    assert corrective[
        "case16_validation_natural_error_profile_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PROFILE)
    assert corrective[
        "case16_validation_natural_error_profile_proposal_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PROPOSAL)
    assert corrective[
        "case16_validation_natural_error_profile_mac_windows_byte_parity"
    ] is True
    assert corrective[
        "case16_validation_natural_error_profile_envelope_retention_fraction"
    ] == 0.4
    assert corrective[
        "case16_validation_natural_error_profile_negative_projection_counts"
    ] == [0, 20, 0]
    assert corrective[
        "case16_validation_natural_error_profile_positive_projection_counts"
    ] == [607, 174, 0]
    assert corrective[
        "case16_validation_natural_error_profile_external_wrench_created"
    ] is False
    assert corrective[
        "case16_validation_natural_error_profile_cpu_ready"
    ] is True
    assert corrective[
        "case16_validation_natural_error_profile_runtime_route_implemented"
    ] is False
    assert corrective[
        "case16_validation_natural_error_profile_runtime_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_profile_label_capture_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_profile_training_authorized"
    ] is False
    case16_profile = json.loads(
        CASE16_VALIDATION_NATURAL_ERROR_PROPOSAL.read_text()
    )
    assert case16_profile["passed"] is True
    assert case16_profile["case"] == 16
    assert case16_profile["split"] == "validation"
    assert all(case16_profile["input_checks"].values())
    assert all(case16_profile["shape_checks"].values())
    assert all(case16_profile["gate_checks"].values())
    assert all(case16_profile["formula_checks"].values())
    assert all(case16_profile["validation_profile_checks"].values())
    assert case16_profile["validation_pair_profile_cpu_ready"] is True
    assert case16_profile["runtime_route_implemented"] is False
    assert case16_profile["runtime_authorized"] is False
    assert case16_profile["label_capture_authorized"] is False
    assert case16_profile["dataset_creation_authorized"] is False
    assert case16_profile["training_started"] is False
    assert case16_profile["valid_for_training"] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_contract_sha256"
    ] == "c648865218132204f95a8346866aed1c1d30c795bdcc37c5767ecb7a44b4fb9c"
    assert corrective[
        "case16_validation_natural_error_pair_route_contract_identity"
    ] == "historical_cpu_route_contract"
    assert corrective[
        "case16_validation_natural_error_pair_route_active_contract_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_CONTRACT)
    assert corrective[
        "case16_validation_natural_error_pair_route_active_contract_git_blob_sha1"
    ] == _git_blob_sha1(CASE16_VALIDATION_NATURAL_ERROR_PAIR_CONTRACT)
    assert corrective[
        "case16_validation_natural_error_pair_route_builder_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_BUILDER)
    assert corrective[
        "case16_validation_natural_error_pair_route_validator_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_VALIDATOR)
    assert corrective[
        "case16_validation_natural_error_pair_route_wrapper_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_WRAPPER)
    assert corrective[
        "case16_validation_natural_error_pair_route_adapter_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_ADAPTER)
    assert corrective[
        "case16_validation_natural_error_pair_route_finalizer_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_FINALIZER)
    assert corrective[
        "case16_validation_natural_error_pair_route_preflight_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_EVIDENCE)
    assert corrective[
        "case16_validation_natural_error_pair_route_identity_count"
    ] == 24
    assert corrective[
        "case16_validation_natural_error_pair_route_active_identity_count"
    ] == 27
    assert corrective[
        "case16_validation_natural_error_pair_route_reset_seed"
    ] == (
        corrective[
            "case16_validation_natural_error_pair_route_configuration_seed"
        ]
        + 16
    )
    assert corrective[
        "case16_validation_natural_error_pair_route_same_plan_seed_and_physics"
    ] is True
    assert corrective[
        "case16_validation_natural_error_pair_route_external_wrench_forbidden"
    ] is True
    assert corrective[
        "case16_validation_natural_error_pair_route_cpu_preflight_passed"
    ] is True
    assert corrective[
        "case16_validation_natural_error_pair_route_contract_ready"
    ] is True
    assert corrective[
        "case16_validation_natural_error_pair_route_execution_route_complete"
    ] is True
    assert corrective[
        "case16_validation_natural_error_pair_route_authorization_token_issued"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_runtime_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_gpu_launch_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_teacher_admission_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_label_capture_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_dataset_creation_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_route_training_authorized"
    ] is False
    case16_route = json.loads(
        CASE16_VALIDATION_NATURAL_ERROR_PAIR_EVIDENCE.read_text()
    )
    assert case16_route["passed"] is True
    assert case16_route["case"] == 16
    assert case16_route["split"] == "validation"
    assert len(case16_route["identities"]) == 24
    assert all(case16_route["document_checks"].values())
    assert all(case16_route["checks"].values())
    assert case16_route["authorization_token_issued"] is False
    assert case16_route["runtime_authorized"] is False
    assert case16_route["gpu_launch_authorized"] is False
    assert case16_route["teacher_admission_authorized"] is False
    assert case16_route["label_capture_authorized"] is False
    assert case16_route["dataset_creation_authorized"] is False
    assert case16_route["bc_authorized"] is False
    assert case16_route["ppo_authorized"] is False
    assert case16_route["training_started"] is False
    assert case16_route["valid_for_training"] is False
    case16_execution = json.loads(
        CASE16_VALIDATION_NATURAL_ERROR_PAIR_EXECUTION_SUMMARY.read_text()
    )
    assert corrective[
        "case16_validation_natural_error_pair_executed"
    ] is True
    assert corrective["case16_validation_natural_error_pair_passed"] is False
    assert corrective[
        "case16_validation_natural_error_pair_evidence_summary_sha256"
    ] == _sha256(CASE16_VALIDATION_NATURAL_ERROR_PAIR_EXECUTION_SUMMARY)
    assert corrective[
        "case16_validation_natural_error_pair_authorization_token_consumed"
    ] is True
    assert corrective[
        "case16_validation_natural_error_pair_retry_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_capture_authorized"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_dataset_created"
    ] is False
    assert corrective[
        "case16_validation_natural_error_pair_valid_for_training"
    ] is False
    assert case16_execution["passed"] is False
    assert case16_execution["dynamic_pair_completed"] is True
    assert case16_execution["baseline_dynamic_quality_passed"] is True
    assert case16_execution["candidate_dynamic_quality_passed"] is True
    assert case16_execution["validation_pair_passed"] is False
    assert case16_execution["failed_paired_checks"] == [
        "minimum_position_p95_improvement",
        "saturation_not_regressed",
    ]
    assert case16_execution["label_capture_authorized"] is False
    assert case16_execution["dataset_created"] is False
    assert case16_execution["training_started"] is False
    case16_disposition = json.loads(
        CASE16_VALIDATION_DISPOSITION_SUMMARY.read_text()
    )
    assert corrective["case16_validation_disposition_auditor_sha256"] == (
        _sha256(CASE16_VALIDATION_DISPOSITION_AUDITOR)
    )
    assert corrective["case16_validation_disposition_auditor_git_blob_sha1"] == (
        "63184520b5422b32bc396b453784931d0c8b2c69"
    )
    assert corrective["case16_validation_disposition_summary_sha256"] == (
        _sha256(CASE16_VALIDATION_DISPOSITION_SUMMARY)
    )
    assert corrective["case16_validation_disposition_ceiling_limited"] is True
    assert corrective[
        "case16_validation_disposition_intrinsically_hard"
    ] is False
    assert corrective[
        "case16_validation_disposition_further_tuning_recommended"
    ] is False
    assert corrective[
        "case16_validation_disposition_teacher_capture_recommended"
    ] is False
    assert corrective[
        "case16_validation_disposition_selected_replacement_case"
    ] == 32
    assert corrective[
        "case16_validation_disposition_case32_currently_admitted"
    ] is False
    assert corrective[
        "case16_validation_disposition_case32_fresh_readiness_required"
    ] is True
    assert corrective[
        "corrective_corpus_intake_frozen_pending_case16_superseded"
    ] is True
    assert corrective[
        "corrective_corpus_intake_selected_pending_validation_case"
    ] == 32
    assert case16_disposition["passed"] is True
    assert case16_disposition["selected_replacement_case"] == 32
    assert case16_disposition["case16"]["ceiling_limited"] is True
    assert case16_disposition["case16"][
        "intrinsically_hard_in_realized_dynamics"
    ] is False
    assert case16_disposition["runtime_authorized"] is False
    assert case16_disposition["label_capture_authorized"] is False
    assert case16_disposition["training_started"] is False
    case32_selection = json.loads(CASE32_VALIDATION_SELECTION.read_text())
    case32_readiness = json.loads(CASE32_VALIDATION_READINESS.read_text())
    case32_proposal = json.loads(
        CASE32_VALIDATION_PROFILE_PROPOSAL.read_text()
    )
    assert corrective["case32_validation_selection_builder_sha256"] == (
        _sha256(CASE32_VALIDATION_SELECTION_BUILDER)
    )
    assert corrective["case32_validation_selection_sha256"] == _sha256(
        CASE32_VALIDATION_SELECTION
    )
    assert corrective["case32_validation_selection_selected_cases"] == [8, 32]
    assert corrective["case32_validation_selection_retired_cases"] == [16]
    assert corrective["case32_validation_readiness_auditor_sha256"] == (
        _sha256(CASE32_VALIDATION_READINESS_AUDITOR)
    )
    assert corrective["case32_validation_readiness_summary_sha256"] == (
        _sha256(CASE32_VALIDATION_READINESS)
    )
    assert corrective["case32_validation_profile_builder_sha256"] == (
        _sha256(CASE32_VALIDATION_PROFILE_BUILDER)
    )
    assert corrective["case32_validation_profile_sha256"] == _sha256(
        CASE32_VALIDATION_PROFILE
    )
    assert corrective["case32_validation_profile_proposal_sha256"] == (
        _sha256(CASE32_VALIDATION_PROFILE_PROPOSAL)
    )
    assert corrective["case32_validation_profile_cpu_ready"] is True
    assert corrective[
        "case32_validation_profile_runtime_route_implemented"
    ] is True
    assert corrective["case32_validation_pair_route_contract_sha256"] == (
        _sha256(CASE32_VALIDATION_PAIR_CONTRACT)
    )
    assert corrective["case32_validation_pair_route_builder_sha256"] == (
        _sha256(CASE32_VALIDATION_PAIR_BUILDER)
    )
    assert corrective["case32_validation_pair_route_validator_sha256"] == (
        _sha256(CASE32_VALIDATION_PAIR_VALIDATOR)
    )
    assert corrective["case32_validation_pair_route_wrapper_sha256"] == (
        _sha256(CASE32_VALIDATION_PAIR_WRAPPER)
    )
    assert corrective["case32_validation_pair_route_adapter_sha256"] == (
        _sha256(CASE32_VALIDATION_PAIR_ADAPTER)
    )
    assert corrective["case32_validation_pair_route_finalizer_sha256"] == (
        _sha256(CASE32_VALIDATION_PAIR_FINALIZER)
    )
    assert corrective[
        "case32_validation_pair_route_preflight_summary_sha256"
    ] == _sha256(CASE32_VALIDATION_PAIR_EVIDENCE)
    assert corrective["case32_validation_pair_route_identity_count"] == 27
    assert corrective[
        "case32_validation_pair_route_canonical_preflight_passed"
    ] is True
    assert corrective[
        "case32_validation_pair_route_authorization_token_issued"
    ] is False
    assert corrective[
        "case32_validation_pair_route_unauthorized_python_started"
    ] is False
    assert corrective[
        "case32_validation_pair_route_unauthorized_isaac_started"
    ] is False
    assert corrective[
        "case32_validation_pair_route_namespace_created"
    ] is False
    assert corrective["case32_validation_profile_runtime_authorized"] is False
    assert corrective["case32_validation_profile_capture_authorized"] is False
    assert corrective["case32_validation_profile_training_authorized"] is False
    assert case32_selection["selected_cases"] == [8, 32]
    assert case32_selection["runtime_authorized"] is False
    assert case32_readiness["case"] == 32
    assert case32_readiness["passed"] is True
    assert case32_readiness["runtime_authorized"] is False
    assert case32_proposal["case"] == 32
    assert case32_proposal["passed"] is True
    assert case32_proposal["runtime_route_implemented"] is False
    assert case32_proposal["runtime_authorized"] is False
    assert case32_proposal["training_started"] is False
    assert corrective[
        "pending_corrective_route_queue_auditor_sha256"
    ] == _sha256(PENDING_CORRECTIVE_ROUTE_QUEUE_AUDITOR)
    assert corrective[
        "pending_corrective_route_queue_auditor_git_blob_sha1"
    ] == "231c98bd8374958a4877049085ff002abc6f4cea"
    assert corrective[
        "pending_corrective_route_queue_test_sha256"
    ] == "00c392594b94fb64d321f313fcb4b79672162f52d48f998a116484a6aa0c716b"
    assert corrective[
        "pending_corrective_route_queue_test_git_blob_sha1"
    ] == "6d9c0adcdd3badfb56d9aaa790d6c3d7d6372485"
    assert corrective[
        "pending_corrective_route_queue_test_identity"
    ] == "historical_cpu_queue_test"
    assert corrective[
        "pending_corrective_route_queue_active_test_sha256"
    ] == _sha256(PENDING_CORRECTIVE_ROUTE_QUEUE_TEST)
    assert corrective[
        "pending_corrective_route_queue_active_test_git_blob_sha1"
    ] == _git_blob_sha1(PENDING_CORRECTIVE_ROUTE_QUEUE_TEST)
    assert corrective[
        "pending_corrective_route_queue_summary_sha256"
    ] == _sha256(PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE)
    assert corrective[
        "pending_corrective_route_queue_summary_identity"
    ] == "historical_cpu_queue_evidence_v1"
    assert corrective[
        "pending_corrective_route_queue_v2_summary_sha256"
    ] == _sha256(PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V2)
    assert corrective[
        "pending_corrective_route_queue_v2_case23_preflight_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_EVIDENCE_V3)
    assert corrective["pending_corrective_route_queue_v2_ready_count"] == 6
    assert corrective[
        "pending_corrective_route_queue_v2_all_preflights_passed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v2_all_authorization_closed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v3_summary_sha256"
    ] == _sha256(PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V3)
    assert corrective[
        "pending_corrective_route_queue_v3_case23_preflight_sha256"
    ] == _sha256(CASE23_CONVERSION_EXECUTION_EVIDENCE_V4)
    assert corrective["pending_corrective_route_queue_v3_ready_count"] == 6
    assert corrective["pending_corrective_route_queue_v3_identity_count"] == 107
    assert corrective[
        "pending_corrective_route_queue_v3_all_preflights_passed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v3_all_authorization_closed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v3_route_identities_unchanged_from_v2"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v3_goal_binding_commit"
    ] == "c6dd84b9cab94fefb2a215d2256c343b40ff72fa"
    assert corrective[
        "pending_corrective_route_queue_v3_focused_mac_cpu_suite"
    ] == "30_passed_2_warnings_in_0.62s"
    assert corrective[
        "pending_corrective_route_queue_v3_focused_windows_cpu_suite"
    ] == "30_passed_2_warnings_in_2.56s"
    assert corrective[
        "pending_corrective_route_queue_v3_authoritative_windows_cpu_suite"
    ] == "1331_passed_12_skipped_2_warnings_in_177.86s"
    assert corrective[
        "pending_corrective_route_queue_v3_runtime_started"
    ] is False
    assert corrective[
        "pending_corrective_route_queue_v3_conversion_started"
    ] is False
    assert corrective[
        "pending_corrective_route_queue_v4_summary_sha256"
    ] == _sha256(PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V4)
    assert corrective[
        "pending_corrective_route_queue_v4_case23_preflight_sha256"
    ] == _sha256(
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260724_pending_corrective_route_queue_cpu_v4/"
        "preflights/case23_conversion.json"
    )
    assert corrective["pending_corrective_route_queue_v4_ready_count"] == 6
    assert corrective["pending_corrective_route_queue_v4_identity_count"] == 107
    assert corrective[
        "pending_corrective_route_queue_v4_all_preflights_passed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v4_all_authorization_closed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v4_control_ownership_identities_resealed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_v4_runtime_started"
    ] is False
    assert corrective[
        "pending_corrective_route_queue_v4_conversion_started"
    ] is False
    assert corrective[
        "pending_corrective_route_queue_ready_count"
    ] == 6
    assert corrective[
        "pending_corrective_route_queue_identity_count"
    ] == 107
    assert corrective[
        "pending_corrective_route_queue_execution_order"
    ] == [
        "case23_conversion",
        "case6_pair",
        "case2_pair",
        "case7_pair",
        "case8_validation_pair",
        "case16_validation_pair",
    ]
    assert corrective[
        "pending_corrective_route_queue_all_preflights_passed"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_all_namespaces_absent"
    ] is True
    assert corrective[
        "pending_corrective_route_queue_next_bounded_action"
    ] == "authorize_exactly_one_case23_v4_cpu_conversion"
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "label_capture_authorized",
        "dataset_conversion_authorized",
        "dataset_merge_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
    ):
        assert corrective[f"pending_corrective_route_queue_{field}"] is False
    route_queue = json.loads(
        PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE.read_text()
    )
    assert route_queue["passed"] is True
    assert route_queue["ready_route_count"] == 6
    assert all(route_queue["checks"].values())
    assert all(route["passed"] for route in route_queue["routes"])
    assert sum(
        route["identity_count"] for route in route_queue["routes"]
    ) == 107
    assert route_queue["next_bounded_action"] == (
        "authorize_exactly_one_case23_v4_cpu_conversion"
    )
    assert route_queue["runtime_authorized"] is False
    assert route_queue["gpu_launch_authorized"] is False
    assert route_queue["label_capture_authorized"] is False
    assert route_queue["dataset_conversion_authorized"] is False
    assert route_queue["dataset_merge_authorized"] is False
    assert route_queue["bc_authorized"] is False
    assert route_queue["ppo_authorized"] is False
    assert route_queue["training_started"] is False
    assert route_queue["valid_for_training"] is False
    route_queue_v2 = json.loads(
        PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V2.read_text()
    )
    assert route_queue_v2["passed"] is True
    assert route_queue_v2["git"]["head"] == (
        corrective["pending_corrective_route_queue_v2_preflight_commit"]
    )
    assert route_queue_v2["ready_route_count"] == 6
    assert all(route["passed"] for route in route_queue_v2["routes"])
    assert route_queue_v2["next_bounded_action"] == (
        "authorize_exactly_one_case23_v4_cpu_conversion"
    )
    assert route_queue_v2["dataset_conversion_authorized"] is False
    assert route_queue_v2["bc_authorized"] is False
    assert route_queue_v2["ppo_authorized"] is False
    assert route_queue_v2["training_started"] is False
    route_queue_v3 = json.loads(
        PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V3.read_text()
    )
    assert route_queue_v3["passed"] is True
    assert route_queue_v3["git"]["head"] == (
        corrective["pending_corrective_route_queue_v3_preflight_commit"]
    )
    assert route_queue_v3["git"]["upstream"] == (
        corrective["pending_corrective_route_queue_v3_preflight_commit"]
    )
    assert route_queue_v3["git"]["tracked_worktree_clean"] is True
    assert route_queue_v3["ready_route_count"] == 6
    assert all(route_queue_v3["checks"].values())
    assert all(route["passed"] for route in route_queue_v3["routes"])
    assert sum(
        route["identity_count"] for route in route_queue_v3["routes"]
    ) == 107
    assert route_queue_v3["next_bounded_action"] == (
        "authorize_exactly_one_case23_v4_cpu_conversion"
    )
    assert route_queue_v3["runtime_authorized"] is False
    assert route_queue_v3["gpu_launch_authorized"] is False
    assert route_queue_v3["label_capture_authorized"] is False
    assert route_queue_v3["dataset_conversion_authorized"] is False
    assert route_queue_v3["dataset_merge_authorized"] is False
    assert route_queue_v3["bc_authorized"] is False
    assert route_queue_v3["ppo_authorized"] is False
    assert route_queue_v3["training_started"] is False
    assert route_queue_v3["valid_for_training"] is False
    route_queue_v4 = json.loads(
        PENDING_CORRECTIVE_ROUTE_QUEUE_EVIDENCE_V4.read_text()
    )
    assert route_queue_v4["passed"] is True
    assert route_queue_v4["git"]["head"] == (
        corrective["pending_corrective_route_queue_v4_preflight_commit"]
    )
    assert route_queue_v4["git"]["upstream"] == (
        corrective["pending_corrective_route_queue_v4_preflight_commit"]
    )
    assert route_queue_v4["git"]["tracked_worktree_clean"] is True
    assert route_queue_v4["ready_route_count"] == 6
    assert all(route_queue_v4["checks"].values())
    assert all(route["passed"] for route in route_queue_v4["routes"])
    assert sum(
        route["identity_count"] for route in route_queue_v4["routes"]
    ) == 107
    assert route_queue_v4["next_bounded_action"] == (
        "authorize_exactly_one_case23_v4_cpu_conversion"
    )
    assert route_queue_v4["runtime_authorized"] is False
    assert route_queue_v4["gpu_launch_authorized"] is False
    assert route_queue_v4["label_capture_authorized"] is False
    assert route_queue_v4["dataset_conversion_authorized"] is False
    assert route_queue_v4["dataset_merge_authorized"] is False
    assert route_queue_v4["bc_authorized"] is False
    assert route_queue_v4["ppo_authorized"] is False
    assert route_queue_v4["training_started"] is False
    assert route_queue_v4["valid_for_training"] is False
    conversion_preflight_v4 = json.loads(
        CASE23_CONVERSION_EXECUTION_EVIDENCE_V4.read_text()
    )
    assert conversion_preflight_v4["passed"] is True
    assert conversion_preflight_v4["cpu_contract_ready"] is True
    assert conversion_preflight_v4["git"]["head"] == (
        corrective["pending_corrective_route_queue_v3_preflight_commit"]
    )
    assert conversion_preflight_v4["git"]["upstream"] == (
        corrective["pending_corrective_route_queue_v3_preflight_commit"]
    )
    assert all(conversion_preflight_v4["repository_checks"].values())
    assert all(conversion_preflight_v4["contract_checks"].values())
    assert not any(
        conversion_preflight_v4["authorization_checks"].values()
    )
    assert all(
        identity["passed"]
        for identity in conversion_preflight_v4["identities"].values()
    )
    assert conversion_preflight_v4["conversion_authorized"] is False
    assert (
        conversion_preflight_v4[
            "authorization_consumed_before_conversion"
        ]
        is False
    )
    assert conversion_preflight_v4["output_created"] is False
    assert conversion_preflight_v4["merged_dataset_created"] is False
    assert conversion_preflight_v4["bc_authorized"] is False
    assert conversion_preflight_v4["ppo_authorized"] is False
    assert conversion_preflight_v4["training_started"] is False
    assert conversion_preflight_v4["valid_for_case_merge"] is False
    assert conversion_preflight_v4["valid_for_training"] is False
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
        "projected_effective_action_case_balanced_recursive_validation_v2"
    )
    assert corpus["projection_training_bc_recursive_validation_contract"] == (
        "case_reset_recursive_effective_action_validation_v1"
    )
    assert (
        corpus[
            "projection_training_bc_recursive_validation_uses_previous_effective_action"
        ]
        is True
    )
    assert corpus[
        "projection_training_bc_recursive_validation_case_reset_required"
    ] is True
    assert corpus["projection_training_bc_recursive_validation_split"] == "validation"
    assert corpus[
        "projection_training_bc_recursive_validation_required_for_artifact_emission"
    ] is True
    assert corpus[
        "projection_training_bc_teacher_forced_only_promotion_rejected"
    ] is True
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
        "cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_report_v2"
    )
    assert corpus[
        "projection_training_bc_execution_contract_module_sha256"
    ] == _sha256(PROJECTED_TRAINING_BC_EXECUTION_CONTRACT)
    assert corpus[
        "projection_training_bc_execution_admission_template_sha256"
    ] == _sha256(PROJECTED_TRAINING_BC_EXECUTION_ADMISSION_TEMPLATE)
    assert corpus[
        "projection_training_bc_execution_contract_initial_implementation_commit"
    ] == ("282c5998ec0c91982d2a1f610b18db7acc5f4e1b")
    assert corpus[
        "projection_training_bc_execution_contract_implementation_commit"
    ] == ("8ee358e045a8099384dae2556e66093f86d1aa05")
    assert corpus[
        "projection_training_bc_execution_contract_focused_cpu_suite"
    ] == ("61_passed_2_warnings_in_10.06s")
    assert corpus[
        "projection_training_bc_execution_contract_authoritative_cpu_suite"
    ] == ("1324_passed_12_skipped_2_warnings_in_171.00s")
    assert corpus["projection_training_bc_execution_admission_template_usable"] is False
    assert corpus["projection_training_bc_execution_trainer_integrated"] is True
    assert corpus["projection_training_bc_execution_synthetic_end_to_end_passed"] is True
    assert corpus["projection_training_bc_execution_real_dataset_available"] is False
    assert corpus["projection_training_bc_execution_real_admission_authorized"] is False
    assert corpus[
        "projection_training_bc_execution_trainer_integration_commit"
    ] == ("282c5998ec0c91982d2a1f610b18db7acc5f4e1b")
    assert corpus["projection_training_bc_execution_trainer_focused_cpu_suite"] == (
        "69_passed_2_warnings_in_27.87s"
    )
    assert corpus[
        "projection_training_bc_execution_trainer_authoritative_cpu_suite"
    ] == ("1300_passed_12_skipped_2_warnings_in_170.69s")
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
    assert "run_exactly_one_case32_validation_natural_error_pair" in (
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
    assert hardware["vendor_source_reconciliation_schema"] == (
        "cinebotrl_two_wheel_riser_vendor_source_reconciliation_audit_v1"
    )
    assert hardware["vendor_source_reconciliation_initial_commit"] == (
        "96aa349615bd8333a2e0e382fb6cbd9601dbf837"
    )
    assert hardware["vendor_source_reconciliation_parity_commit"] == (
        "3718ff4ae3bc2ca051fe3283f666491dd0fd57a6"
    )
    assert hardware["vendor_source_reconciliation_contract_sha256"] == _sha256(
        VENDOR_SOURCE_RECONCILIATION
    )
    assert hardware["vendor_source_reconciliation_auditor_sha256"] == _sha256(
        VENDOR_SOURCE_RECONCILIATION_AUDITOR
    )
    assert hardware["vendor_source_reconciliation_summary_sha256"] == _sha256(
        VENDOR_SOURCE_RECONCILIATION_EVIDENCE
    )
    reconciliation = json.loads(
        VENDOR_SOURCE_RECONCILIATION_EVIDENCE.read_text()
    )
    assert reconciliation["passed"] is True
    assert all(reconciliation["checks"].values())
    assert (
        hardware["vendor_source_motor_model"]
        == reconciliation["selected_motor"]
        == "ELVM8075V48EH-M17-HD"
    )
    assert (
        hardware["vendor_source_drive_model"]
        == reconciliation["selected_drive"]
        == "ELD2-CAN7020B"
    )
    assert hardware["vendor_source_selected_drive_has_dedicated_cn6_sto"] is False
    assert hardware["vendor_source_external_safety_power_removal_required"] is True
    assert hardware["vendor_source_fixed_axis_reference"] == (
        "igus_drylin_ZLW_1080_standard"
    )
    assert hardware["vendor_source_camera_height_ceiling_m"] == 1.8
    assert hardware["vendor_source_target_speed_mps"] == 1.0
    assert hardware["vendor_source_mac_windows_byte_parity"] is True
    assert hardware["vendor_source_real_supplier_evidence_collected"] is False
    assert hardware["vendor_source_real_bench_evidence_collected"] is False
    assert hardware["vendor_source_simulation_profile_changed"] is False
    assert hardware["vendor_source_runtime_or_training_authorized"] is False
    assert hardware["vendor_source_authoritative_windows_cpu_suite"] == (
        "1388_passed_12_skipped_2_warnings_in_221.23s"
    )
    assert hardware["ready_for_production_design_review"] is False
    assert hardware["valid_for_production_procurement"] is False
    assert hardware["valid_for_hardware_transfer"] is False


def test_goal_completion_audit_preserves_the_real_end_state() -> None:
    audit = _goal()["current_stage"]["status_refresh_20260723"][
        "goal_completion_audit"
    ]
    assert audit["schema"] == "cinebotrl_two_wheel_riser_goal_completion_audit_v7"
    assert audit["implementation_commit"] == (
        "29ac155bcd9d7f34cdde58301273851330170630"
    )
    assert audit["host_independent_lf_evidence"] is True
    assert audit["mac_and_windows_report_byte_parity_verified"] is True
    assert audit["auditor_code_identity_bound"] is True
    assert audit["all79_full_row_revalidation_required"] is True
    assert audit["learned_all79_admission_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_learned_all79_admission_v1"
    )
    assert audit["learned_all79_admission_template_sha256"] == _sha256(
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "MODEL_BASED_LEARNED_ALL79_ADMISSION_TEMPLATE_20260723.json"
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
    assert audit["learned_policy_artifact_inspector_sha256"] == _sha256(
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_policy_artifact.py"
    )
    assert audit["learned_policy_artifact_input_dimension"] == 65
    assert audit["learned_policy_artifact_action_dimension"] == 3
    assert audit["learned_policy_artifact_parameter_count"] == 142019
    assert audit["learned_policy_artifact_cpu_smoke_required"] is True
    assert audit["learned_policy_artifact_valid_live_windows_cpu"] is True
    assert (
        audit["learned_policy_artifact_malformed_rejected_live_windows_cpu"]
        is True
    )
    assert audit["learned_preflight_windows_python_wsl_git_bridge_passed"] is True
    assert audit["learned_policy_artifact_authoritative_windows_cpu_commit"] == (
        "e7476fbee8fa8963d85dbd880f9edf196eba8a8c"
    )
    assert audit["learned_policy_artifact_authoritative_windows_cpu_suite"] == (
        "1309_passed_12_skipped_2_warnings_in_175.46s"
    )
    assert audit["learned_policy_control_ownership_contract"] == (
        "frozen_lqr_high_level_residual_control_ownership_v1"
    )
    assert audit["learned_policy_action_names"] == [
        "residual_vx_normalized",
        "residual_wz_normalized",
        "residual_riser_target_normalized",
    ]
    assert audit["learned_policy_direct_wheel_effort"] is False
    assert audit["learned_policy_physical_gimbal_joint_action"] is False
    assert audit["learned_policy_wheel_effort_owner"] == "frozen_cascaded_lqr"
    assert audit["learned_policy_gimbal_attitude_owner"] == (
        "deterministic_semantic_attitude_adapter"
    )
    assert audit["learned_policy_riser_hard_limit_owner"] == (
        "deterministic_command_supervisor"
    )
    assert audit["learned_policy_safety_supervisor_owner"] == (
        "deterministic_runtime_gates"
    )
    assert audit["learned_policy_playback_sha256"] == _sha256(
        ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
    )
    assert audit["learned_policy_residual_dataset_sha256"] == _sha256(
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/riser_residual_dataset.py"
    )
    assert audit["learned_policy_control_ownership_implementation_commit"] == (
        "20694e8d5d238d1965a22e3eef4e14fe57682f05"
    )
    assert audit["learned_policy_control_ownership_route_reseal_commit"] == (
        "8c9d1e4b2de2fa0e2007f98004e01b989b6d6883"
    )
    assert audit["learned_policy_control_ownership_queue_evidence_commit"] == (
        "5b269a729a8006a7bec5fd5d9bb6fa594e1e58e7"
    )
    assert audit["learned_policy_control_ownership_goal_binding_commit"] == (
        "011b3e2f1c4c460de28318866c80662a91415953"
    )
    assert audit["learned_policy_control_ownership_focused_local_cpu_suite"] == (
        "214_passed_2_warnings_in_16.45s"
    )
    assert audit["learned_policy_control_ownership_focused_windows_cpu_suite"] == (
        "214_passed_2_warnings_in_79.74s"
    )
    assert audit[
        "learned_policy_control_ownership_authoritative_windows_cpu_suite"
    ] == "1337_passed_12_skipped_2_warnings_in_180.63s"
    assert audit["learned_policy_validation_gate_schema"].endswith(
        "residual_validation_canary_gate_v3"
    )
    assert audit["learned_policy_holdout_gate_schema"].endswith(
        "residual_holdout_gate_v3"
    )
    assert audit["learned_policy_all79_gate_schema"].endswith(
        "residual_all79_gate_v3"
    )
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
    assert audit["pre_training_architecture_contract_passed"] is True
    assert audit["pre_training_observation_dimension"] == 65
    assert audit["pre_training_action_dimension"] == 3
    assert audit["pre_training_corrective_case_datasets_available"] == 4
    assert audit["pre_training_required_train_cases"] == 4
    assert audit["pre_training_required_validation_cases"] == 2
    assert audit["pre_training_pending_route_queue_passed"] is True
    assert audit["pre_training_pending_route_queue_bound_to_goal"] is True
    assert audit["pre_training_pending_route_queue_ready_count"] == 6
    assert audit["pre_training_pending_route_queue_identity_count"] == 107
    assert audit["pre_training_pending_route_queue_execution_order"] == [
        "case23_conversion",
        "case6_pair",
        "case2_pair",
        "case7_pair",
        "case8_validation_pair",
        "case16_validation_pair",
    ]
    assert (
        audit["pre_training_pending_route_queue_all_authorization_closed"]
        is True
    )
    assert audit["pre_training_corrective_route_catalog_consolidated"] is True
    assert audit["pre_training_corrective_route_catalog_schema"] == (
        "cinebotrl_two_wheel_riser_model_based_corrective_route_catalog_v1"
    )
    assert audit["pre_training_corrective_route_catalog_sha256"] == _sha256(
        CORRECTIVE_ROUTE_CATALOG
    )
    assert audit["pre_training_corrective_route_preparer_sha256"] == _sha256(
        CORRECTIVE_ROUTE_PREPARER
    )
    assert audit[
        "pre_training_corrective_route_preflight_report_sha256"
    ] == _sha256(CORRECTIVE_ROUTE_PREFLIGHT)
    assert audit["pre_training_corrective_route_catalog_routes"] == [
        "case7_pair",
        "case8_validation_pair",
        "case16_validation_pair",
    ]
    assert audit["pre_training_corrective_route_catalog_identity_count"] == 62
    assert (
        audit["pre_training_corrective_route_catalog_observation_dimension"]
        == 65
    )
    assert audit["pre_training_corrective_route_catalog_action_dimension"] == 3
    assert (
        audit["pre_training_corrective_route_catalog_mac_windows_byte_parity"]
        is True
    )
    assert (
        audit["pre_training_corrective_route_catalog_runtime_authorized"]
        is False
    )
    assert (
        audit["pre_training_corrective_route_catalog_training_authorized"]
        is False
    )
    assert audit["pre_training_case7_corrective_capture_user_authorized"] is True
    assert (
        audit[
            "pre_training_case7_corrective_capture_route_authorization_token_issued"
        ]
        is True
    )
    assert (
        audit[
            "pre_training_case7_corrective_capture_route_authorization_token_consumed"
        ]
        is True
    )
    assert audit["pre_training_shared_windows_resource_guard_implemented"] is True
    assert audit["pre_training_shared_windows_resource_guard_commit"] == (
        "d0365653571d50523584e80e2ec1943febdfe6d4"
    )
    assert audit["pre_training_shared_windows_resource_guard_sha256"] == (
        _sha256(SHARED_RESOURCE_GUARD)
    )
    assert audit[
        "pre_training_shared_windows_resource_guard_minimum_windows_free_memory_gib"
    ] == 5.0
    assert audit[
        "pre_training_shared_windows_resource_guard_minimum_gpu_free_memory_mib"
    ] == 9216
    assert audit[
        "pre_training_shared_windows_resource_guard_cad_processes_must_be_absent"
    ] is False
    assert audit[
        "pre_training_shared_windows_resource_guard_live_report_sha256"
    ] == _sha256(SHARED_RESOURCE_EVIDENCE)
    resource_report = json.loads(SHARED_RESOURCE_EVIDENCE.read_text())
    assert resource_report["passed"] is True
    assert resource_report["runtime_started"] is False
    assert resource_report["authorization_consumed"] is False
    assert resource_report["checks"]["cad_coexistence_allowed"] is True
    assert resource_report["checks"]["windows_free_memory_sufficient"] is True
    assert resource_report["checks"]["gpu_free_memory_sufficient"] is True
    assert audit["pre_training_shared_windows_resource_guard_live_passed"] is True
    assert (
        audit["pre_training_shared_windows_resource_guard_runtime_started"]
        is False
    )
    assert (
        audit[
            "pre_training_shared_windows_resource_guard_authorization_consumed"
        ]
        is False
    )
    assert (
        audit[
            "pre_training_shared_windows_resource_guard_controller_commands_changed"
        ]
        is False
    )
    assert audit["pre_training_case7_resource_finalizer_seal_implemented"] is True
    assert audit["pre_training_case7_resource_finalizer_seal_commit"] == (
        "0f2de2b8175e59395cd61b45d37a49a071ad81e5"
    )
    assert audit[
        "pre_training_case7_resource_finalizer_seal_summary_sha256"
    ] == _sha256(CASE7_RESOURCE_FINALIZER_SEAL_SUMMARY)
    assert (
        audit[
            "pre_training_case7_resource_finalizer_requires_admission_evidence"
        ]
        is True
    )
    assert (
        audit[
            "pre_training_case7_resource_finalizer_missing_or_tampered_evidence_rejected"
        ]
        is True
    )
    assert audit[
        "pre_training_case7_resource_finalizer_command_equivalence_sha256"
    ] == _sha256(CASE7_RESOURCE_FINALIZER_SEAL_COMMANDS)
    assert audit[
        "pre_training_case7_resource_finalizer_live_report_sha256"
    ] == _sha256(CASE7_RESOURCE_FINALIZER_SEAL_LIVE)
    assert audit["pre_training_case7_resource_finalizer_live_passed"] is False
    assert (
        audit["pre_training_case7_monitored_cad_coexistence_implemented"]
        is True
    )
    assert audit["pre_training_case7_monitored_cad_coexistence_commit"] == (
        "d0365653571d50523584e80e2ec1943febdfe6d4"
    )
    assert audit["pre_training_case7_corrective_capture_completed"] is True
    assert audit[
        "pre_training_case7_corrective_capture_evidence_summary_sha256"
    ] == _sha256(CASE7_CAPTURE_V2_SUMMARY)
    assert (
        audit["pre_training_case7_corrective_capture_admitted_for_conversion"]
        is True
    )
    assert audit["pre_training_case7_corrective_conversion_authorized"] is False
    assert audit["pre_training_case7_corrective_conversion_completed"] is True
    assert (
        audit["pre_training_case7_corrective_conversion_valid_for_case_merge"]
        is True
    )
    assert (
        audit["pre_training_case7_corrective_conversion_valid_for_training"]
        is False
    )
    assert audit["pre_training_corrective_corpus_intake_converted_train_cases"] == [
        6,
        7,
        23,
        30,
    ]
    assert (
        audit[
            "pre_training_corrective_corpus_intake_converted_validation_cases"
        ]
        == [8]
    )
    assert audit["pre_training_corrective_corpus_intake_manifest_ready"] is False
    assert audit["pre_training_case8_validation_pair_completed"] is True
    assert audit[
        "pre_training_case8_validation_pair_evidence_summary_sha256"
    ] == _sha256(CASE8_VALIDATION_PAIR_EXECUTION_SUMMARY)
    assert audit[
        "pre_training_case8_validation_pair_capture_authorized"
    ] is False
    assert audit["pre_training_case8_validation_capture_completed"] is True
    assert audit[
        "pre_training_case8_validation_capture_evidence_summary_sha256"
    ] == _sha256(CASE8_VALIDATION_CAPTURE_SUMMARY)
    assert audit[
        "pre_training_case8_validation_capture_admitted_for_conversion"
    ] is True
    assert audit[
        "pre_training_case8_validation_capture_conversion_authorized"
    ] is False
    assert audit["pre_training_case8_validation_conversion_completed"] is True
    assert audit[
        "pre_training_case8_validation_conversion_evidence_summary_sha256"
    ] == _sha256(CASE8_VALIDATION_CONVERSION_SUMMARY)
    assert audit[
        "pre_training_case8_validation_conversion_valid_for_case_merge"
    ] is True
    assert audit[
        "pre_training_case8_validation_conversion_valid_for_training"
    ] is False
    assert audit["pre_training_next_operation"] == (
        "cpu_only_diagnose_case16_absolute_improvement_and_saturation_regression"
    )
    assert audit["pre_training_no_hidden_cpu_route_blocker"] is True
    assert audit["pre_training_generic_capture_finalizer_implemented"] is True
    assert audit["pre_training_generic_capture_finalizer_cases"] == [6, 23, 30]
    assert audit["pre_training_generic_capture_finalizer_total_samples"] == 22617
    assert (
        audit["pre_training_generic_capture_finalizer_train_live_reopen_passed"]
        is True
    )
    assert (
        audit["pre_training_generic_capture_finalizer_validation_contract_passed"]
        is True
    )
    assert (
        audit["pre_training_generic_capture_finalizer_controller_commands_changed"]
        is False
    )
    assert (
        audit["pre_training_generic_capture_finalizer_runtime_started"] is False
    )
    assert audit["pre_training_generic_capture_finalizer_summary_sha256"] == (
        _sha256(GENERIC_CAPTURE_FINALIZER_EVIDENCE)
    )
    generic_capture_finalizer = json.loads(
        GENERIC_CAPTURE_FINALIZER_EVIDENCE.read_text()
    )
    assert generic_capture_finalizer["passed"] is True
    assert generic_capture_finalizer["case_count"] == 3
    assert generic_capture_finalizer["total_sample_count"] == 22617
    assert generic_capture_finalizer["supported_splits"] == [
        "train",
        "validation",
    ]
    assert generic_capture_finalizer["controller_commands_changed"] is False
    assert generic_capture_finalizer["runtime_started"] is False
    assert generic_capture_finalizer["capture_started"] is False
    assert generic_capture_finalizer["training_started"] is False
    assert audit["pre_training_capture_command_audit_implemented"] is True
    assert audit["pre_training_capture_command_audit_commit"] == (
        "bce0f1ba2a4df99ef321290ab33391e68953f72a"
    )
    assert audit["pre_training_capture_command_audit_summary_sha256"] == (
        _sha256(CAPTURE_COMMAND_AUDIT)
    )
    assert audit["pre_training_capture_command_auditor_sha256"] == _sha256(
        CAPTURE_COMMAND_AUDITOR
    )
    assert audit["pre_training_capture_command_contract_sha256"] == _sha256(
        CAPTURE_COMMAND_CONTRACT
    )
    assert audit[
        "pre_training_capture_command_audit_mac_windows_byte_parity"
    ] is True
    assert audit[
        "pre_training_capture_command_current_compatible_cases"
    ] == [6, 7, 23]
    assert audit["pre_training_capture_command_historical_only_cases"] == [30]
    assert audit[
        "pre_training_capture_command_pending_case7_compatible"
    ] is True
    assert audit[
        "pre_training_capture_command_generic_runtime_wrapper_created"
    ] is False
    assert audit[
        "pre_training_capture_command_audit_focused_local_cpu_suite"
    ] == "52_passed_2_warnings_in_4.05s"
    assert audit[
        "pre_training_capture_command_audit_focused_windows_cpu_suite"
    ] == "52_passed_2_warnings_in_17.98s"
    assert audit[
        "pre_training_capture_command_audit_authoritative_windows_cpu_commit"
    ] == "ee07678edc00b0b74f9e59abcd18bc9ded79954b"
    assert audit[
        "pre_training_capture_command_audit_authoritative_windows_cpu_suite"
    ] == "1435_passed_12_skipped_2_warnings_in_256.25s"
    capture_command_audit = json.loads(CAPTURE_COMMAND_AUDIT.read_text())
    assert capture_command_audit["passed"] is True
    assert all(capture_command_audit["checks"].values())
    capture_routes = {
        route["case"]: route for route in capture_command_audit["routes"]
    }
    assert capture_routes[7]["current_command_compatible"] is True
    assert capture_routes[7]["mismatches"] == []
    assert capture_routes[30]["current_command_compatible"] is False
    assert capture_routes[30]["mismatches"] == [
        "--corrective-teacher-capture-split",
        "playback_identity",
    ]
    assert capture_command_audit["runtime_authorized"] is False
    assert capture_command_audit["capture_started"] is False
    assert capture_command_audit["dataset_created"] is False
    assert capture_command_audit["training_started"] is False
    assert audit["pre_training_next_operation_authorized"] is False
    assert audit["focused_local_cpu_suite"] == "30_passed_2_warnings_in_2.89s"
    assert audit["focused_windows_cpu_suite"] == "30_passed_2_warnings_in_14.83s"
    assert audit["goal_binding_commit"] == (
        "acbf4f26685e40099c5d5df106698627017136ee"
    )
    assert audit["combined_focused_local_cpu_suite"] == (
        "30_passed_2_warnings_in_2.89s"
    )
    assert audit["combined_focused_windows_cpu_suite"] == (
        "30_passed_2_warnings_in_14.83s"
    )
    assert audit["authoritative_windows_cpu_commit"] == (
        "acbf4f26685e40099c5d5df106698627017136ee"
    )
    assert audit["authoritative_windows_cpu_suite"] == (
        "1427_passed_12_skipped_2_warnings_in_244.41s"
    )
    assert _sha256(GOAL_COMPLETION_AUDIT) == (
        "e87a1aab9b89aab46114a65ae2a53929632c8b48a41a3d2eb85190c75dd6871e"
    )
    report = json.loads(GOAL_COMPLETION_AUDIT.read_text())
    assert report["git"]["head"] == "29ac155bcd9d7f34cdde58301273851330170630"
    assert report["inputs"]["auditor"]["sha256"] == (
        "f7d84fb73109fe0ff8438cc6fa5cf8dd6b308556166926469ae9d5ed44d0182e"
    )
    assert report["required_gate_pass_count"] == audit["required_gate_pass_count"]
    assert report["required_gate_count"] == audit["required_gate_count"]
    assert report["completion_blockers"] == audit["completion_blockers"]
    assert report["goal_achieved"] is False
    assert report["runtime_started"] is False
    assert report["bc_started_by_audit"] is False
    assert report["ppo_started_by_audit"] is False
    assert report["pre_training_readiness"]["architecture_contract_passed"] is True
    assert report["pre_training_readiness"]["pending_route_queue_passed"] is True
    assert report["pre_training_readiness"][
        "pending_route_queue_bound_to_goal"
    ] is True
    assert report["pre_training_readiness"][
        "pending_route_queue_ready_count"
    ] == 6
    assert report["pre_training_readiness"][
        "pending_route_queue_identity_count"
    ] == 107
    assert report["pre_training_readiness"][
        "pending_route_queue_all_authorization_closed"
    ] is True
    assert report["pre_training_readiness"][
        "case7_corrective_capture_route_ready"
    ] is True
    assert report["pre_training_readiness"][
        "case7_corrective_capture_authorization_closed"
    ] is True
    assert report["pre_training_readiness"][
        "generic_conversion_proposals_ready"
    ] is True
    assert report["pre_training_readiness"][
        "generic_conversion_execution_ready"
    ] is True
    assert report["pre_training_readiness"]["no_hidden_cpu_route_blocker"] is True
    assert report["pre_training_readiness"]["next_case"] == 7
    assert report["pre_training_readiness"]["next_operation"] == (
        "authorize_exactly_one_case7_corrective_label_capture"
    )
    assert report["pre_training_readiness"]["next_operation_authorized"] is False
    assert report["inputs"]["pending_route_queue"]["sha256"] == (
        "244377cb46a69d744f26449f74a4fa5301c0416c3142857f8213dfbacd05f041"
    )
    assert report["pre_training_readiness"]["ready_for_bc_execution"] is False
