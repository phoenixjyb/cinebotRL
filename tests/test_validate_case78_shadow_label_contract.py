from scripts.two_wheel_balance.validate_case78_shadow_label_contract import (
    EXPECTED_CONTROLLER,
    EXPECTED_IMPLEMENTATION,
    EXPECTED_NAMESPACE,
    EXPECTED_REVIEWED_PARENT,
    EXPECTED_SCALES,
    EXPECTED_THRESHOLDS,
    semantic_checks,
)


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    contract = {
        "schema": "cinebotrl_two_wheel_riser_case78_shadow_label_cpu_contract_v1",
        "case": 78,
        "current_split": "validation",
        "namespace": EXPECTED_NAMESPACE,
        "reviewed_parent_commit": EXPECTED_REVIEWED_PARENT,
        "implementation_commit": EXPECTED_IMPLEMENTATION,
        "plan_contract": {
            "case": 78,
            "plan_sha256": "28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569",
            "source_pose_count": 6870,
            "execution_state_count": 6870,
            "source_duration_s": 135.487646,
            "execution_duration_s": 192.29956737098348,
        },
        "controller_arguments": EXPECTED_CONTROLLER,
        "dynamic_gate_thresholds": EXPECTED_THRESHOLDS,
        "measurement_contract": {
            "visited_state_source": "deterministic_controller",
            "raw_labels_applied_to_commands": False,
            "applied_residual_actions_must_be_zero": True,
            "trace_only": True,
            "dataset_present": False,
            "record_policy_rate_timestamps": True,
            "record_source_and_execution_clocks": True,
            "record_raw_and_normalized_labels": True,
            "record_command_reconstruction_fields": True,
        },
        "one_case_only": True,
        "maximum_runtime_seconds": 5400,
        "heartbeat_interval_policy_steps": 2000,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "shadow_measurement_authorized": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
    }
    audit = {
        "decision": "retain_teacher40_scale_case78_series_measurement_required",
        "teacher40_action_contract_retained": True,
        "teacher40_candidate_scale": EXPECTED_SCALES,
        "candidate_scale_maximum_compatibility_passed": True,
        "case78_shadow_measurement_required_before_label_capture": True,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    split = {
        "split_admitted": True,
        "admitted_split_cases": {"train": [4], "validation": [78]},
        "case78_labels_available": False,
        "label_capture_authorized": False,
    }
    case78 = {
        "case": 78,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "maximum_camera_lever_arm_correction_m": 0.1,
        "camera_recovery_governor_enabled": False,
        "termination": None,
        "executed_residual_dataset": None,
    }
    gate = {"cases": [78], "passed": True, "results": [case78]}
    final = {
        "case": 78,
        "passed": True,
        "dynamic_qualification_passed": True,
        "dataset_created": False,
        "training_started": False,
    }
    return contract, audit, split, gate, final


def test_shadow_label_contract_keeps_deterministic_control_and_learning_closed() -> None:
    checks = semantic_checks(*_inputs())
    assert all(checks.values())


def test_shadow_label_contract_rejects_zero_policy_or_scale_change() -> None:
    values = list(_inputs())
    values[0]["controller_arguments"] = {
        **EXPECTED_CONTROLLER,
        "zero_policy_action": True,
        "residual_action_scales": [0.4, 0.4, 0.1],
    }
    checks = semantic_checks(*values)
    assert not checks["controller_is_exact"]


def test_shadow_label_contract_rejects_runtime_or_training_authorization() -> None:
    values = list(_inputs())
    values[0]["runtime_authorized"] = True
    values[0]["bc_authorized"] = True
    checks = semantic_checks(*values)
    assert not checks["cpu_only_and_no_learning"]


def test_shadow_label_contract_rejects_unpassed_physics() -> None:
    values = list(_inputs())
    values[3]["results"][0]["dynamic_quality_passed"] = False
    checks = semantic_checks(*values)
    assert not checks["passed_canary_is_bound"]
