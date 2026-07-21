from scripts.two_wheel_balance.validate_case78_camera_recovery_contract import (
    EXPECTED_CONTROLLER,
    EXPECTED_IDENTITIES,
    EXPECTED_PLAN,
    EXPECTED_RECOVERY_CONTROLLER,
    EXPECTED_THRESHOLDS,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
    semantic_checks,
)


def valid_inputs() -> tuple[dict, dict, dict, dict]:
    contract = {
        "schema": SCHEMA,
        "case": 78,
        "current_split": "unused",
        "namespace": NAMESPACE,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "identities": {name: {} for name in EXPECTED_IDENTITIES},
        "plan_contract": EXPECTED_PLAN,
        "controller_arguments": EXPECTED_RECOVERY_CONTROLLER,
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
    checks = {"position_p95_bounded": False, "pitch_bounded": True}
    prior_gate = {
        "results": [
            {
                "completed_phase_time_s": 192.0,
                "execution_duration_s": 192.0,
                "termination": None,
                "position_error_p95_m": 0.162649892749212,
                "position_error_max_m": 0.22962387152256802,
                "checks": checks,
            }
        ]
    }
    prior_final = {
        "dynamic_qualification_passed": False,
        "case78_validation_admitted": False,
        "split_changed": False,
    }
    recovery_audit = {
        "candidate_supported_for_bounded_canary": True,
        "offline_trace_estimate_is_physical_proof": False,
        "projected_candidate_steps": 87392,
        "maximum_steps": 115381,
    }
    return contract, prior_gate, prior_final, recovery_audit


def test_recovery_contract_changes_only_governor_arguments() -> None:
    checks = semantic_checks(*valid_inputs())
    assert all(checks.values())
    changed = {
        key
        for key in EXPECTED_RECOVERY_CONTROLLER
        if EXPECTED_RECOVERY_CONTROLLER.get(key) != EXPECTED_CONTROLLER.get(key)
    }
    assert changed == {
        "enable_camera_error_recovery_governor",
        "camera_recovery_error_start_m",
        "camera_recovery_error_full_m",
        "minimum_camera_recovery_scale",
    }


def test_recovery_contract_rejects_gate_or_runtime_relaxation() -> None:
    contract, prior_gate, prior_final, recovery_audit = valid_inputs()
    contract["dynamic_gate_thresholds"] = {
        **EXPECTED_THRESHOLDS,
        "maximum_position_p95_m": 0.17,
    }
    contract["runtime_authorized"] = True
    checks = semantic_checks(
        contract, prior_gate, prior_final, recovery_audit
    )
    assert not checks["thresholds_unchanged"]
    assert not checks["cpu_only"]

