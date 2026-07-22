from scripts.two_wheel_balance.validate_case78_camera_cap_contract import (
    EXPECTED_CAP_CONTROLLER,
    EXPECTED_CONTROLLER,
    EXPECTED_IDENTITIES,
    EXPECTED_PLAN,
    EXPECTED_THRESHOLDS,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
    semantic_checks,
)


def valid_inputs() -> tuple[dict, dict, dict, dict, dict]:
    contract = {
        "schema": SCHEMA,
        "case": 78,
        "current_split": "unused",
        "namespace": NAMESPACE,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "identities": {name: {} for name in EXPECTED_IDENTITIES},
        "plan_contract": EXPECTED_PLAN,
        "controller_arguments": EXPECTED_CAP_CONTROLLER,
        "dynamic_gate_thresholds": EXPECTED_THRESHOLDS,
        "one_case_only": True,
        "maximum_runtime_seconds": 5400,
        "heartbeat_interval_policy_steps": 2000,
        "cpu_preflight_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_qualification_authorized": False,
        "dataset_creation_authorized": False,
        "split_change_authorized": False,
        "holdout_opened": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    baseline_gate = {
        "results": [{
            "completed_phase_time_s": 192.0,
            "execution_duration_s": 192.0,
            "termination": None,
            "position_error_p95_m": 0.162649892749212,
            "position_error_max_m": 0.22962387152256802,
            "checks": {"position_p95_bounded": False, "pitch_bounded": True},
        }]
    }
    baseline_final = {
        "dynamic_qualification_passed": False,
        "case78_validation_admitted": False,
        "split_changed": False,
    }
    recovery_outcome = {
        "audit_passed": True,
        "camera_recovery_candidate_rejected": True,
        "runtime_authorized": False,
    }
    cap_audit = {
        "audit_passed": True,
        "cpu_candidate_supported": True,
        "current_cap_m": 0.05,
        "candidate_cap_m": 0.10,
        "dynamic_proof_obtained": False,
        "gpu_launch_authorized": False,
    }
    return contract, baseline_gate, baseline_final, recovery_outcome, cap_audit


def test_camera_cap_contract_changes_only_correction_cap() -> None:
    checks = semantic_checks(*valid_inputs())
    assert all(checks.values())
    changed = {
        name
        for name in EXPECTED_CAP_CONTROLLER
        if EXPECTED_CAP_CONTROLLER.get(name) != EXPECTED_CONTROLLER.get(name)
    }
    assert changed == {"maximum_camera_lever_arm_correction_m"}


def test_camera_cap_contract_rejects_gate_or_runtime_relaxation() -> None:
    inputs = list(valid_inputs())
    contract = inputs[0]
    contract["dynamic_gate_thresholds"] = {
        **EXPECTED_THRESHOLDS,
        "maximum_position_p95_m": 0.17,
    }
    contract["runtime_authorized"] = True
    checks = semantic_checks(*inputs)
    assert not checks["thresholds_unchanged"]
    assert not checks["cpu_only"]


def test_camera_cap_contract_rejects_failed_or_dynamic_audit() -> None:
    inputs = list(valid_inputs())
    cap_audit = inputs[-1]
    cap_audit["dynamic_proof_obtained"] = True
    checks = semantic_checks(*inputs)
    assert not checks["camera_cap_candidate_supported_cpu_only"]
