import copy

import pytest

from scripts.two_wheel_balance.build_initial_teacher41_case78_canary_contract import (
    ACTION_SCALES,
    EXPECTED_HASHES,
    EXPECTED_PLAN,
    TRACKING_PROFILE,
    build_contract,
)


def _payloads():
    teacher_result = {
        "case": 78,
        "source_duration_s": EXPECTED_PLAN["source_duration_s"],
        "execution_duration_s": EXPECTED_PLAN["execution_duration_s"],
        "completed_phase_time_s": EXPECTED_PLAN["execution_duration_s"],
        "position_error_p95_m": 0.1166,
        "position_error_max_m": 0.1842,
        "residual_action_abs_max": [0.0, 0.0, 0.0],
        "executed_residual_dataset": None,
        "passed": True,
    }
    return {
        "plan_case_report": {
            **EXPECTED_PLAN,
            "passed": True,
            "valid_for_training": False,
            "timing_transition_kinematic_gate_passed": True,
            "kinematic_checks": {"rates": True},
        },
        "camera_cap_cpu_contract": {
            "case": 78,
            "plan_contract": EXPECTED_PLAN,
            "controller_arguments": {
                "maximum_camera_lever_arm_correction_m": 0.1,
                "controller_wz_kp": 1.05,
                "maximum_duration_scale": 3.0,
                "trajectory_command_source": "deterministic_teacher",
            },
            "runtime_authorized": False,
            "gpu_launch_authorized": False,
        },
        "teacher_gate": {
            "cases": [78],
            "trajectory_command_source": "deterministic_teacher",
            "tracking_profile": TRACKING_PROFILE,
            "position_observation_link": "physical_cam_link_fk",
            "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
            "hardware_proxy_command_contract": "semantic_attitude_position_only",
            "controller_overrides": {"wz_kp": 1.05},
            "maximum_duration_scale": 3.0,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.1,
            "residual_policy": None,
            "passed": True,
            "dynamic_quality_passed": True,
            "results": [teacher_result],
        },
        "teacher_final": {
            "passed": True,
            "case": 78,
            "current_split": "validation",
            "physical_quality_passed": True,
            "shadow_trace_passed": True,
            "labels_applied_to_commands": False,
            "dataset_created": False,
            "valid_for_training": False,
        },
        "label_admission": {
            "case": 78,
            "split": "validation",
            "action_scales": ACTION_SCALES,
            "label_admission_passed": True,
            "labels_applied_to_commands": False,
            "holdout_opened": False,
            "training_started": False,
        },
        "case8_final": {
            "passed": True,
            "dynamic_canary_passed": True,
            "case": 8,
            "case78_authorized": False,
            "dataset_created": False,
        },
        "case8_summary": {
            "passed": True,
            "cases": [8],
            "policy_sha256": EXPECTED_HASHES["policy_torchscript"],
        },
        "policy_final": {
            "passed": True,
            "offline_gate_passed": True,
            "case8_canary_proposal_ready": True,
            "learned_rollout_authorized": False,
            "learned_rollout_started": False,
            "holdout_opened": False,
            "ppo_started": False,
            "torchscript": {"sha256": EXPECTED_HASHES["policy_torchscript"]},
        },
        "policy_report": {
            "offline_gate_passed": True,
            "policy_architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
            "masked_observation_indices": [23, 24, 25],
            "offline_gate_splits": ["validation"],
            "holdout_metrics_computed": False,
            "holdout_used_for_model_selection": False,
            "torchscript_sha256": EXPECTED_HASHES["policy_torchscript"],
        },
    }


def test_builds_case78_cpu_contract_without_runtime_authorization() -> None:
    result = build_contract(_payloads(), "a" * 40)
    assert result["cpu_contract_ready"]
    assert result["case"] == 78
    assert result["controller_contract"]["maximum_camera_lever_arm_correction_m"] == 0.1
    assert result["comparison_contract"]["fresh_zero_required"]
    assert not result["runtime_authorized"]
    assert not result["case16_22_32_authorized"]
    assert not result["holdout_opened"]


def test_rejects_case8_failure_or_camera_cap_drift() -> None:
    payloads = _payloads()
    payloads["case8_final"]["passed"] = False
    payloads["camera_cap_cpu_contract"]["controller_arguments"][
        "maximum_camera_lever_arm_correction_m"
    ] = 0.05
    with pytest.raises(ValueError, match="case-78 learned canary"):
        build_contract(payloads, "a" * 40)


def test_rejects_teacher_clock_or_holdout_drift() -> None:
    payloads = _payloads()
    payloads["teacher_gate"]["results"][0]["execution_duration_s"] += 1.0
    payloads["policy_report"]["holdout_metrics_computed"] = True
    with pytest.raises(ValueError, match="case-78 learned canary"):
        build_contract(payloads, "a" * 40)
