import copy

import pytest

from scripts.two_wheel_balance.build_model_based_zero_residual_case78_contract import (
    EXPECTED_CONTROLLER,
    EXPECTED_HASHES,
    EXPECTED_PLAN,
    NAMESPACE,
    build_contract,
)


def _payloads():
    return {
        "case8_final": {
            "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_final_v1",
            "case": 8,
            "passed": True,
            "zero_residual_preservation_passed": True,
            "metric_absolute_deltas": {
                "position_m": 0.0,
                "attitude_deg": 0.0,
                "pitch_deg": 0.0,
                "riser_m": 0.0,
                "proxy_deg": 0.0,
            },
            "zero_checkpoint_sha256": EXPECTED_HASHES["zero_policy_torchscript"],
            "dataset_creation_authorized": False,
            "training_authorized": False,
            "ppo_authorized": False,
            "holdout_opened": False,
        },
        "plan_case_report": {
            **EXPECTED_PLAN,
            "passed": True,
            "timing_transition_kinematic_gate_passed": True,
            "kinematic_checks": {"rates": True},
            "valid_for_training": False,
        },
        "camera_cap_cpu_contract": {
            "case": 78,
            "plan_contract": EXPECTED_PLAN,
            "controller_arguments": {
                "maximum_camera_lever_arm_correction_m": 0.1,
                "controller_wz_kp": 1.05,
            },
            "dynamic_gate_thresholds": {
                "maximum_pitch_deg": 12.0,
                "maximum_position_p95_m": 0.15,
                "maximum_position_error_m": 0.25,
                "maximum_attitude_p95_deg": 5.0,
                "maximum_attitude_error_deg": 10.0,
                "maximum_riser_servo_error_m": 0.03,
                "maximum_proxy_servo_error_deg": 5.0,
                "maximum_internal_proxy_rate_deg_s": 360.0,
                "maximum_saturation_ratio": 0.2,
            },
            "runtime_authorized": False,
            "gpu_launch_authorized": False,
            "dataset_creation_authorized": False,
        },
        "teacher_gate": {
            "cases": [78],
            "trajectory_command_source": "deterministic_teacher",
            "tracking_profile": EXPECTED_CONTROLLER["tracking_profile"],
            "position_observation_link": "physical_cam_link_fk",
            "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
            "hardware_proxy_command_contract": "semantic_attitude_position_only",
            "maximum_camera_lever_arm_correction_m": 0.1,
            "residual_policy": None,
            "dynamic_quality_passed": True,
            "passed": True,
            "results": [
                {
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
            ],
        },
        "zero_policy_report": {
            "passed": True,
            "policy_architecture": "model_based_shared_encoder_zero_initialized_residual_v1",
            "command_contract": EXPECTED_CONTROLLER["policy_residual_contract"],
            "residual_action_scales": [0.05, 0.05, 0.02],
            "residual_head_exact_zero": True,
            "checkpoint": {"sha256": EXPECTED_HASHES["zero_policy_checkpoint"]},
            "torchscript": {"sha256": EXPECTED_HASHES["zero_policy_torchscript"]},
            "runtime_authorized": False,
            "training_authorized": False,
            "ppo_authorized": False,
        },
        "case78_failure_audit": {
            "passed": True,
            "failed_dynamic_gate": "position_p95_bounded",
            "architecture_audit": {
                "required_contract_satisfied": False,
                "checkpoint_classification": "planner_imitation_bc_initialization_only",
            },
            "decision": {
                "bc_retraining_authorized": False,
                "ppo_authorized": False,
            },
        },
    }


def test_builds_case78_cpu_contract_without_runtime_authorization() -> None:
    result = build_contract(_payloads(), "a" * 40)
    assert result["cpu_contract_ready"] is True
    assert result["namespace"] == NAMESPACE
    assert result["controller_contract"]["policy_command_base"] == "model_based_planner"
    assert result["controller_contract"]["residual_action_scales"] == [0.05, 0.05, 0.02]
    assert result["controller_contract"]["maximum_camera_lever_arm_correction_m"] == 0.1
    assert result["runtime_authorization_token_issued"] is False
    assert result["dataset_creation_authorized"] is False
    assert result["ppo_authorized"] is False


def test_rejects_case8_failure_or_nonzero_delta() -> None:
    payloads = _payloads()
    payloads["case8_final"]["passed"] = False
    payloads["case8_final"]["metric_absolute_deltas"]["position_m"] = 0.001
    with pytest.raises(ValueError, match="case-78 contract failed"):
        build_contract(payloads, "a" * 40)


def test_rejects_camera_cap_or_teacher_clock_drift() -> None:
    payloads = _payloads()
    payloads["camera_cap_cpu_contract"]["controller_arguments"][
        "maximum_camera_lever_arm_correction_m"
    ] = 0.05
    payloads["teacher_gate"]["results"][0]["execution_duration_s"] += 1.0
    with pytest.raises(ValueError, match="case-78 contract failed"):
        build_contract(payloads, "a" * 40)


def test_rejects_nonzero_policy_or_training_reopening() -> None:
    payloads = copy.deepcopy(_payloads())
    payloads["zero_policy_report"]["residual_head_exact_zero"] = False
    payloads["case78_failure_audit"]["decision"]["ppo_authorized"] = True
    with pytest.raises(ValueError, match="case-78 contract failed"):
        build_contract(payloads, "a" * 40)
