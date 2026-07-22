import copy

from scripts.two_wheel_balance.validate_model_based_zero_residual_case8_contract import (
    EXPECTED_CONTROLLER,
    EXPECTED_IDENTITIES,
    EXPECTED_PLAN,
    EXPECTED_PRESERVATION,
    EXPECTED_ROLLOUTS,
    EXPECTED_THRESHOLDS,
    NAMESPACE,
    REVIEWED_CONTROLLER_PARENT,
    SCHEMA,
    ZERO_POLICY_CHECKPOINT_SHA256,
    ZERO_POLICY_REPORT_SHA256,
    ZERO_POLICY_SOURCE_COMMIT,
    ZERO_POLICY_TORCHSCRIPT_SHA256,
    semantic_checks,
)


def _contract():
    return {
        "schema": SCHEMA,
        "case": 8,
        "split": "validation",
        "namespace": NAMESPACE,
        "reviewed_controller_parent_commit": REVIEWED_CONTROLLER_PARENT,
        "zero_policy_source_commit": ZERO_POLICY_SOURCE_COMMIT,
        "plan_contract": copy.deepcopy(EXPECTED_PLAN),
        "controller_contract": copy.deepcopy(EXPECTED_CONTROLLER),
        "rollouts": copy.deepcopy(EXPECTED_ROLLOUTS),
        "dynamic_gate_thresholds": copy.deepcopy(EXPECTED_THRESHOLDS),
        "preservation_gate": copy.deepcopy(EXPECTED_PRESERVATION),
        "identities": {
            name: {
                "path": name,
                "sha256": {
                    "zero_policy_checkpoint": ZERO_POLICY_CHECKPOINT_SHA256,
                    "zero_policy_torchscript": ZERO_POLICY_TORCHSCRIPT_SHA256,
                    "zero_policy_report": ZERO_POLICY_REPORT_SHA256,
                }.get(name, "x"),
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
        "timing_transition_kinematic_gate_passed": True,
        "kinematic_checks": {"rates": True},
        "derivation_checks": {"source_unchanged": True},
    }


def _plan_admission():
    return {
        "passed": True,
        "selected_plan": {
            "case": 8,
            "plan_sha256": EXPECTED_PLAN["plan_sha256"],
            "passed": True,
        },
    }


def _teacher_gate():
    return {
        "cases": [8],
        "trajectory_command_source": "deterministic_teacher",
        "tracking_profile": EXPECTED_CONTROLLER["tracking_profile"],
        "phase_feedforward_contract": EXPECTED_CONTROLLER[
            "phase_feedforward_contract"
        ],
        "position_observation_link": EXPECTED_CONTROLLER[
            "position_observation_link"
        ],
        "target_attitude_contract": EXPECTED_CONTROLLER["target_attitude_contract"],
        "hardware_proxy_command_contract": EXPECTED_CONTROLLER[
            "hardware_proxy_command_contract"
        ],
        "dynamic_quality_passed": True,
        "passed": True,
        "results": [
            {
                "case": 8,
                "source_duration_s": EXPECTED_PLAN["source_duration_s"],
                "execution_duration_s": EXPECTED_PLAN["execution_duration_s"],
                "dynamic_quality_passed": True,
                "passed": True,
                "residual_action_abs_max": [0.0, 0.0, 0.0],
                "executed_residual_dataset": None,
            }
        ],
    }


def _zero_policy_report():
    return {
        "passed": True,
        "source_commit": ZERO_POLICY_SOURCE_COMMIT,
        "policy_architecture": "model_based_shared_encoder_zero_initialized_residual_v1",
        "command_contract": EXPECTED_CONTROLLER["policy_residual_contract"],
        "residual_action_scales": [0.05, 0.05, 0.02],
        "residual_head_exact_zero": True,
        "checkpoint": {"sha256": ZERO_POLICY_CHECKPOINT_SHA256},
        "torchscript": {"sha256": ZERO_POLICY_TORCHSCRIPT_SHA256},
        "runtime_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
    }


def _checks(contract=None, policy=None):
    return semantic_checks(
        contract or _contract(),
        plan_report=_plan_report(),
        plan_admission=_plan_admission(),
        teacher_gate=_teacher_gate(),
        zero_policy_report=policy or _zero_policy_report(),
    )


def test_healthy_contract_is_cpu_only() -> None:
    checks = _checks()
    assert all(checks.values()), checks


def test_rejects_old_planner_imitation_layering_and_scales() -> None:
    contract = _contract()
    contract["controller_contract"]["policy_command_base"] = "phase_feedforward"
    contract["controller_contract"]["residual_action_scales"] = [0.35, 0.4, 0.1]
    assert not _checks(contract=contract)["controller_contract_matches"]


def test_rejects_relaxed_physical_or_preservation_gates() -> None:
    contract = _contract()
    contract["dynamic_gate_thresholds"]["maximum_position_p95_m"] = 0.2
    contract["preservation_gate"]["maximum_position_metric_delta_m"] = 0.1
    checks = _checks(contract=contract)
    assert not checks["dynamic_thresholds_unchanged"]
    assert not checks["preservation_gate_exact"]


def test_rejects_nonzero_or_training_authorized_policy() -> None:
    policy = _zero_policy_report()
    policy["residual_head_exact_zero"] = False
    policy["training_authorized"] = True
    assert not _checks(policy=policy)["zero_policy_build_exact"]


def test_rejects_runtime_token_capture_or_holdout() -> None:
    contract = _contract()
    contract["runtime_authorization_token_sha256"] = "forged"
    contract["dataset_creation_authorized"] = True
    contract["holdout_opened"] = True
    checks = _checks(contract=contract)
    assert not checks["no_runtime_token"]
    assert not checks["no_runtime_or_learning_side_effects"]
