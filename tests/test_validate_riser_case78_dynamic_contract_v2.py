import json
from pathlib import Path

from scripts.two_wheel_balance.validate_riser_case78_dynamic_contract_v2 import (
    EXPECTED_CONTROLLER,
    EXPECTED_HEARTBEAT,
    EXPECTED_IDENTITIES,
    EXPECTED_PLAN,
    EXPECTED_THRESHOLDS,
    EXPECTED_WALL_BOUND,
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
        "controller_arguments": EXPECTED_CONTROLLER,
        "dynamic_gate_thresholds": EXPECTED_THRESHOLDS,
        "runtime_heartbeat": EXPECTED_HEARTBEAT,
        "wall_timeout_derivation": EXPECTED_WALL_BOUND,
        "one_case_only": True,
        "maximum_runtime_seconds": 5400,
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
    fallback = {
        "decision": "transparent_split_reset_pending_case78_dynamic_qualification",
        "split_changed": False,
        "case78_validation_admitted": False,
    }
    summary = {
        "items": [
            {
                "case": 78,
                "plan_sha256": EXPECTED_PLAN["plan_sha256"],
                "checks": {"integrity": True},
                "kinematic_checks": {"position": True},
                "timing_transition_kinematic_gate_passed": True,
            }
        ]
    }
    wall_audit = {
        "audit_passed": True,
        "proposed_maximum_wall_duration_s": 5400,
        "runtime_retry_authorized": False,
    }
    timeout_final = {
        "playback_exit_code": 124,
        "dynamic_qualification_passed": False,
        "case78_validation_admitted": False,
    }
    return contract, fallback, summary, wall_audit, timeout_final


def test_v2_semantic_contract_is_cpu_only_and_ready() -> None:
    checks = semantic_checks(*valid_inputs())
    assert all(checks.values())


def test_checked_in_v2_contract_matches_semantic_contract() -> None:
    _, fallback, summary, wall_audit, timeout_final = valid_inputs()
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/case78_dynamic_cpu_contract_v2.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks = semantic_checks(
        contract, fallback, summary, wall_audit, timeout_final
    )
    assert all(checks.values())


def test_v2_semantic_contract_rejects_runtime_or_timeout_mutation() -> None:
    contract, fallback, summary, wall_audit, timeout_final = valid_inputs()
    contract["runtime_authorized"] = True
    contract["maximum_runtime_seconds"] = 900
    checks = semantic_checks(
        contract, fallback, summary, wall_audit, timeout_final
    )
    assert not checks["cpu_only"]
    assert not checks["one_case_no_capture"]


def test_v2_semantic_contract_rejects_heartbeat_command_mutation() -> None:
    contract, fallback, summary, wall_audit, timeout_final = valid_inputs()
    contract["runtime_heartbeat"] = {
        **EXPECTED_HEARTBEAT,
        "changes_commands": True,
    }
    checks = semantic_checks(
        contract, fallback, summary, wall_audit, timeout_final
    )
    assert not checks["heartbeat_contract_matches"]
