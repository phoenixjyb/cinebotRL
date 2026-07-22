import copy

from scripts.two_wheel_balance.validate_initial_teacher41_case8_canary_contract import (
    EXPECTED_COMMON_CONTROLLER,
    EXPECTED_COMPARISON,
    EXPECTED_IDENTITIES,
    EXPECTED_PLAN,
    EXPECTED_POLICY_SHA256,
    EXPECTED_ROLLOUTS,
    EXPECTED_THRESHOLDS,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
    semantic_checks,
)


def _contract():
    return {
        "schema": SCHEMA,
        "case": 8,
        "split": "validation",
        "namespace": NAMESPACE,
        "reviewed_policy_parent_commit": REVIEWED_PARENT,
        "plan_contract": copy.deepcopy(EXPECTED_PLAN),
        "common_controller_arguments": copy.deepcopy(EXPECTED_COMMON_CONTROLLER),
        "rollouts": copy.deepcopy(EXPECTED_ROLLOUTS),
        "dynamic_gate_thresholds": copy.deepcopy(EXPECTED_THRESHOLDS),
        "comparison_gate": copy.deepcopy(EXPECTED_COMPARISON),
        "identities": {
            name: {
                "path": name,
                "sha256": EXPECTED_POLICY_SHA256 if name == "policy_torchscript" else "x",
            }
            for name in EXPECTED_IDENTITIES
        },
        "one_case_only": True,
        "cpu_preflight_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_canary_authorized": False,
        "case78_authorized": False,
        "broad_rollout_authorized": False,
        "dataset_creation_authorized": False,
        "raw_teacher_capture_authorized": False,
        "policy_trace_capture_authorized": False,
        "shadow_teacher_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
    }


def _plan_report():
    return {
        **EXPECTED_PLAN,
        "passed": True,
        "valid_for_training": False,
        "timing_transition_kinematic_gate_passed": True,
        "kinematic_checks": {"rates": True},
        "derivation_checks": {"source_unchanged": True},
    }


def _plan_admission():
    return {
        "passed": True,
        "valid_for_training": False,
        "selected_plan": {
            "case": 8,
            "plan_sha256": EXPECTED_PLAN["plan_sha256"],
            "passed": True,
        },
    }


def _teacher_gate(*, raw: bool = False):
    metrics = {
        "position_error_p95_m": 0.131,
        "position_error_max_m": 0.143,
        "attitude_error_p95_deg": 0.149,
        "attitude_error_max_deg": 0.223,
        "pitch_p95_deg": 5.98,
        "pitch_max_deg": 6.15,
        "riser_servo_error_p95_m": 0.011,
        "riser_servo_error_max_m": 0.012,
        "proxy_servo_error_p95_deg": 0.118,
        "proxy_servo_error_max_deg": 0.229,
    }
    result = {
        "case": 8,
        "source_duration_s": EXPECTED_PLAN["source_duration_s"],
        "execution_duration_s": EXPECTED_PLAN["execution_duration_s"],
        "dynamic_quality_passed": True,
        "passed": True,
        "residual_action_abs_max": [0.0, 0.0, 0.0],
        "executed_residual_dataset": None,
        **metrics,
    }
    if raw:
        result.update(
            {
                "raw_residual_label_applied_to_commands": False,
                "executed_raw_teacher_capture": "case_0008.npz",
            }
        )
    return {
        "cases": [8],
        "trajectory_command_source": "deterministic_teacher",
        "tracking_profile": EXPECTED_COMMON_CONTROLLER["tracking_profile"],
        "phase_feedforward_contract": EXPECTED_COMMON_CONTROLLER[
            "phase_feedforward_contract"
        ],
        "position_observation_link": EXPECTED_COMMON_CONTROLLER[
            "position_observation_link"
        ],
        "target_attitude_contract": EXPECTED_COMMON_CONTROLLER[
            "target_attitude_contract"
        ],
        "hardware_proxy_command_contract": EXPECTED_COMMON_CONTROLLER[
            "hardware_proxy_command_contract"
        ],
        "camera_lever_arm_compensation_enabled": True,
        "camera_lever_arm_compensation_gain": 1.0,
        "maximum_camera_lever_arm_correction_m": 0.05,
        "controller_overrides": {"wz_kp": 1.05},
        "maximum_duration_scale": 3.0,
        "residual_policy": None,
        "raw_teacher_capture_started": raw,
        "normalized_dataset_capture_started": False,
        "dynamic_quality_passed": True,
        "passed": True,
        "results": [result],
    }


def _policy_final():
    return {
        "passed": True,
        "offline_gate_passed": True,
        "case8_canary_proposal_ready": True,
        "learned_rollout_authorized": False,
        "learned_rollout_started": False,
        "holdout_opened": False,
        "ppo_authorized": False,
        "ppo_started": False,
        "torchscript": {"sha256": EXPECTED_POLICY_SHA256},
    }


def _policy_report():
    return {
        "offline_gate_passed": True,
        "policy_architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
        "masked_observation_indices": [23, 24, 25],
        "previous_action_observation_contract": "masked_after_normalization_v1",
        "offline_gate_splits": ["validation"],
        "dataset_sha256": "03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497",
        "holdout_used_for_model_selection": False,
        "holdout_metrics_computed": False,
        "learned_rollout_started": False,
        "ppo_started": False,
        "torchscript_sha256": EXPECTED_POLICY_SHA256,
    }


def _policy_admission():
    return {
        "passed": True,
        "validation_only_model_selection": True,
        "holdout_opened": False,
        "learned_rollout_authorized": False,
        "ppo_authorized": False,
    }


def _checks(contract=None, teacher=None):
    return semantic_checks(
        contract or _contract(),
        plan_report=_plan_report(),
        plan_admission=_plan_admission(),
        teacher_gate=teacher or _teacher_gate(),
        raw_teacher_gate=_teacher_gate(raw=True),
        policy_final=_policy_final(),
        policy_report=_policy_report(),
        policy_admission=_policy_admission(),
    )


def test_healthy_case8_contract_is_cpu_only() -> None:
    checks = _checks()
    assert all(checks.values()), checks


def test_rejects_changed_camera_or_action_contract() -> None:
    contract = _contract()
    contract["common_controller_arguments"]["residual_action_scales"] = [0.4, 0.4, 0.1]
    contract["common_controller_arguments"]["target_attitude_contract"] = "physical_joint_angles"
    checks = _checks(contract=contract)
    assert not checks["controller_contract_matches"]


def test_rejects_changed_teacher_clock_or_profile() -> None:
    teacher = _teacher_gate()
    teacher["tracking_profile"] = "other_profile"
    teacher["results"][0]["execution_duration_s"] += 1.0
    checks = _checks(teacher=teacher)
    assert not checks["teacher_reference_exact"]
    assert not checks["teacher_result_exact"]


def test_rejects_runtime_token_capture_or_holdout_opening() -> None:
    contract = _contract()
    contract["runtime_authorization_token_sha256"] = "forged"
    contract["dataset_creation_authorized"] = True
    contract["holdout_opened"] = True
    checks = _checks(contract=contract)
    assert not checks["no_runtime_token"]
    assert not checks["no_capture_or_training_side_effects"]


def test_rejects_policy_identity_drift() -> None:
    contract = _contract()
    contract["identities"]["policy_torchscript"]["sha256"] = "forged"
    checks = _checks(contract=contract)
    assert not checks["policy_identity_exact"]
