import copy
import hashlib
import json
from pathlib import Path

import pytest

from rl_platform.tasks.two_wheel_balance.riser_model_based_learned_all79_contract import (
    ALL79_CASES,
    BC_REPORT_SCHEMA,
    CODE_IDENTITY_KEYS,
    DEFAULT_EVALUATION_CONFIG,
    HOLDOUT_GATE_SCHEMA,
    MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA,
    VALIDATION_GATE_SCHEMA,
    validate_learned_all79_admission,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_corpus import (
    DEFAULT_RESERVED_HOLDOUT_CASES,
)


ROOT = Path(__file__).parents[1]
TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_LEARNED_ALL79_ADMISSION_TEMPLATE_20260723.json"
)
EXECUTION_COMMIT = "a" * 40
VALIDATION_CASES = [8, 16]


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _gate_report(schema: str, cases: list[int], policy_sha: str) -> dict:
    rows = [
        {
            "case": case,
            "checks": {
                "learned_hard_gate": True,
                "teacher_hard_gate": True,
                "bounded_residual": True,
            },
            "teacher": {},
            "learned": {},
            "zero": {},
            "learned_residual_action_abs_max": [0.5, 0.4, 0.3],
            "learned_beats_zero_position_p95": True,
        }
        for case in cases
    ]
    return {
        "schema": schema,
        "policy_sha256": policy_sha,
        "cases": cases,
        "case_count": len(cases),
        "maximum_regression_fraction": 0.05,
        "minimum_zero_improvement_fraction": 0.05,
        "expected_tracking_profile": "riser_phase_consistent_v2",
        "means": {
            "teacher_position_p95_m": 0.10,
            "learned_position_p95_m": 0.08,
            "zero_position_p95_m": 0.10,
        },
        "aggregate_checks": {
            "all_case_checks": True,
            "learned_position_mean_within_teacher_budget": True,
            "learned_beats_zero_by_required_mean": True,
            "learned_beats_zero_on_majority_of_cases": True,
        },
        "rows": rows,
        "passed": True,
        "ppo_authorized": False,
    }


def _fixture(tmp_path: Path):
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    policy_identity = _identity(policy)
    bc_report_path = tmp_path / "bc_report.json"
    bc_report = {
        "schema": BC_REPORT_SCHEMA,
        "execution_commit": EXECUTION_COMMIT,
        "split_cases": {"train": [2, 6, 7, 23], "validation": VALIDATION_CASES},
        "torchscript": policy_identity,
        "offline_gate_passed": True,
        "passed": True,
        "valid_for_dynamic_canary": True,
        "training_started": True,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
    }
    bc_report_path.write_text(json.dumps(bc_report), encoding="utf-8")

    validation_path = tmp_path / "validation.json"
    validation = _gate_report(
        VALIDATION_GATE_SCHEMA,
        VALIDATION_CASES,
        policy_identity["sha256"],
    )
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    holdout_path = tmp_path / "holdout.json"
    holdout = _gate_report(
        HOLDOUT_GATE_SCHEMA,
        DEFAULT_RESERVED_HOLDOUT_CASES,
        policy_identity["sha256"],
    )
    holdout_path.write_text(json.dumps(holdout), encoding="utf-8")

    code_paths = {}
    code_identities = {}
    for name in CODE_IDENTITY_KEYS:
        path = tmp_path / f"{name}.py"
        path.write_text(name, encoding="utf-8")
        code_paths[name] = path
        code_identities[name] = _identity(path)
    admission = {
        "schema": MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA,
        "bc_report": _identity(bc_report_path),
        "policy": policy_identity,
        "validation_gate_report": _identity(validation_path),
        "holdout_gate_report": _identity(holdout_path),
        "execution_commit": EXECUTION_COMMIT,
        "code": code_identities,
        "evaluation_config": DEFAULT_EVALUATION_CONFIG,
        "validation_cases": VALIDATION_CASES,
        "holdout_cases": DEFAULT_RESERVED_HOLDOUT_CASES,
        "all79_cases": ALL79_CASES,
        "model_selection_complete": True,
        "validation_gate_passed": True,
        "holdout_gate_passed": True,
        "holdout_opened_only_after_model_selection": True,
        "all79_evaluation_approved": True,
        "learned_rollout_authorized": True,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    return {
        "admission": admission,
        "bc_report_path": bc_report_path,
        "bc_report": bc_report,
        "policy_path": policy,
        "validation_report_path": validation_path,
        "validation_report": validation,
        "holdout_report_path": holdout_path,
        "holdout_report": holdout,
        "code_paths": code_paths,
    }


def _validate(fixture, *, require_authorized: bool = True) -> None:
    validate_learned_all79_admission(
        fixture["admission"],
        identity_root=fixture["bc_report_path"].parent,
        bc_report_path=fixture["bc_report_path"],
        bc_report=fixture["bc_report"],
        policy_path=fixture["policy_path"],
        validation_report_path=fixture["validation_report_path"],
        validation_report=fixture["validation_report"],
        holdout_report_path=fixture["holdout_report_path"],
        holdout_report=fixture["holdout_report"],
        code_paths=fixture["code_paths"],
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=require_authorized,
    )


def test_authorized_all79_admission_binds_all_prerequisites(tmp_path: Path) -> None:
    _validate(_fixture(tmp_path))


def test_admission_preserves_majority_zero_baseline_gate_semantics(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    holdout = fixture["holdout_report"]
    holdout["rows"][0]["learned_beats_zero_position_p95"] = False
    fixture["holdout_report_path"].write_text(
        json.dumps(holdout),
        encoding="utf-8",
    )
    fixture["admission"]["holdout_gate_report"] = _identity(
        fixture["holdout_report_path"]
    )
    _validate(fixture)


@pytest.mark.parametrize(
    "mutation",
    (
        "bc_hash",
        "policy_hash",
        "code_hash",
        "validation_failed",
        "holdout_failed",
        "all79_cases",
        "config",
        "ppo",
        "authorization",
    ),
)
def test_admission_rejects_forged_or_open_downstream_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    admission = fixture["admission"]
    if mutation == "bc_hash":
        admission["bc_report"]["sha256"] = "0" * 64
    elif mutation == "policy_hash":
        admission["policy"]["sha256"] = "0" * 64
    elif mutation == "code_hash":
        admission["code"]["playback"]["sha256"] = "0" * 64
    elif mutation == "validation_failed":
        fixture["validation_report"]["passed"] = False
    elif mutation == "holdout_failed":
        fixture["holdout_report"]["aggregate_checks"][
            "learned_beats_zero_on_majority_of_cases"
        ] = False
    elif mutation == "all79_cases":
        admission["all79_cases"] = ALL79_CASES[:-1]
    elif mutation == "config":
        admission["evaluation_config"] = copy.deepcopy(DEFAULT_EVALUATION_CONFIG)
        admission["evaluation_config"]["residual_action_scales"] = [0.1, 0.1, 0.1]
    elif mutation == "ppo":
        admission["ppo_authorized"] = True
    else:
        admission["all79_evaluation_approved"] = False
    with pytest.raises(ValueError, match="admission failed"):
        _validate(fixture)


def test_checked_in_template_is_structural_but_unusable() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["schema"] == MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA
    assert template["evaluation_config"] == DEFAULT_EVALUATION_CONFIG
    assert template["validation_cases"] == []
    assert template["holdout_cases"] == DEFAULT_RESERVED_HOLDOUT_CASES
    assert template["all79_cases"] == ALL79_CASES
    assert template["all79_evaluation_approved"] is False
    assert template["learned_rollout_authorized"] is False
    assert template["ppo_authorized"] is False
    assert template["training_started"] is False
