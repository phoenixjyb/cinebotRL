from copy import deepcopy
from pathlib import Path

from scripts.two_wheel_balance.validate_riser_case78_dynamic_contract import (
    EXPECTED_CONTROLLER,
    EXPECTED_IDENTITIES,
    EXPECTED_PLAN,
    EXPECTED_THRESHOLDS,
    EXPECTED_TIMING,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
    semantic_checks,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/two_wheel_balance/run_riser_case78_dynamic_preflight.sh"


def _payloads() -> tuple[dict, dict, dict]:
    contract = {
        "schema": SCHEMA,
        "case": 78,
        "current_split": "unused",
        "reviewed_parent_commit": REVIEWED_PARENT,
        "namespace": NAMESPACE,
        "identities": {name: {} for name in EXPECTED_IDENTITIES},
        "plan_contract": EXPECTED_PLAN,
        "controller_arguments": EXPECTED_CONTROLLER,
        "dynamic_gate_thresholds": EXPECTED_THRESHOLDS,
        "wall_timeout_derivation": EXPECTED_TIMING,
        "one_case_only": True,
        "maximum_runtime_seconds": 900,
        "cpu_preflight_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "split_change_authorized": False,
        "holdout_opened": False,
    }
    fallback = {
        "decision": (
            "transparent_split_reset_pending_case78_dynamic_qualification"
        ),
        "split_changed": False,
        "case78_validation_admitted": False,
    }
    plan_summary = {
        "items": [{
            **EXPECTED_PLAN,
            "checks": {"source_time_verbatim": True},
            "kinematic_checks": {"position_p95_bounded": True},
            "timing_transition_kinematic_gate_passed": True,
        }]
    }
    return contract, fallback, plan_summary


def test_semantic_contract_passes_healthy_payload() -> None:
    checks = semantic_checks(*_payloads())
    assert all(checks.values())


def test_semantic_contract_rejects_shorter_timeout() -> None:
    contract, fallback, summary = deepcopy(_payloads())
    contract["maximum_runtime_seconds"] = 600
    contract["wall_timeout_derivation"]["wall_timeout_s"] = 600
    checks = semantic_checks(contract, fallback, summary)
    assert not checks["timing_contract_matches"]
    assert not checks["one_case_no_capture"]


def test_semantic_contract_rejects_applied_split() -> None:
    contract, fallback, summary = deepcopy(_payloads())
    fallback["split_changed"] = True
    fallback["case78_validation_admitted"] = True
    checks = semantic_checks(contract, fallback, summary)
    assert not checks["fallback_pending_case78"]


def test_preflight_wrapper_contains_no_runtime_path() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "runtime_authorization_not_issued" in source
    assert "smoke_riser_reference_playback.py" not in source
    assert "timeout --signal" not in source
    assert "mkdir -p" not in source
    assert "runtime_started" in source
    assert "dataset_created" in source
