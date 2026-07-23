"""Hash-bound admission contract for final learned all-79 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .riser_model_based_corrective_corpus import DEFAULT_RESERVED_HOLDOUT_CASES


MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_learned_all79_admission_v1"
)
BC_REPORT_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_report_v1"
)
VALIDATION_GATE_SCHEMA = (
    "cinebotrl_two_wheel_riser_residual_validation_canary_gate_v1"
)
HOLDOUT_GATE_SCHEMA = "cinebotrl_two_wheel_riser_residual_holdout_gate_v1"
CODE_IDENTITY_KEYS = {
    "playback",
    "rollout_gate",
    "completion_auditor",
    "admission_contract",
}
ALL79_CASES = list(range(1, 80))
DEFAULT_EVALUATION_CONFIG = {
    "tracking_profile": "riser_phase_consistent_v2",
    "policy_command_contract": "model_based_planner_plus_bounded_policy_residual_v1",
    "residual_action_scales": [0.05, 0.05, 0.02],
    "maximum_regression_fraction": 0.05,
    "minimum_zero_improvement_fraction": 0.05,
    "position_error_p95_m_max": 0.15,
    "position_error_max_m_max": 0.25,
    "attitude_error_p95_deg_max": 5.0,
    "attitude_error_max_deg_max": 10.0,
}
ADMISSION_FIELDS = {
    "schema",
    "bc_report",
    "policy",
    "validation_gate_report",
    "holdout_gate_report",
    "execution_commit",
    "code",
    "evaluation_config",
    "validation_cases",
    "holdout_cases",
    "all79_cases",
    "model_selection_complete",
    "validation_gate_passed",
    "holdout_gate_passed",
    "holdout_opened_only_after_model_selection",
    "all79_evaluation_approved",
    "learned_rollout_authorized",
    "residual_capture_authorized",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
}
GATE_REPORT_FIELDS = {
    "schema",
    "policy_sha256",
    "cases",
    "case_count",
    "maximum_regression_fraction",
    "minimum_zero_improvement_fraction",
    "expected_tracking_profile",
    "means",
    "aggregate_checks",
    "rows",
    "passed",
    "ppo_authorized",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_identity(
    identity: object,
    *,
    directory: Path,
    expected_path: Path,
) -> bool:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not identity["path"]
        or not _exact_digest(identity.get("sha256"), 64)
    ):
        return False
    path = Path(identity["path"])
    path = path if path.is_absolute() else directory / path
    path = path.resolve()
    return (
        path == expected_path.resolve()
        and path.is_file()
        and identity["sha256"] == sha256_file(path)
    )


def _gate_report_valid(
    report: Mapping[str, Any],
    *,
    schema: str,
    cases: list[int],
    policy_sha256: str,
) -> bool:
    rows = report.get("rows")
    means = report.get("means")
    aggregates = report.get("aggregate_checks")
    return (
        set(report) == GATE_REPORT_FIELDS
        and report.get("schema") == schema
        and report.get("policy_sha256") == policy_sha256
        and report.get("cases") == cases
        and report.get("case_count") == len(cases)
        and report.get("maximum_regression_fraction")
        == DEFAULT_EVALUATION_CONFIG["maximum_regression_fraction"]
        and report.get("minimum_zero_improvement_fraction")
        == DEFAULT_EVALUATION_CONFIG["minimum_zero_improvement_fraction"]
        and report.get("expected_tracking_profile")
        == DEFAULT_EVALUATION_CONFIG["tracking_profile"]
        and isinstance(rows, list)
        and len(rows) == len(cases)
        and all(
            isinstance(row, Mapping)
            and row.get("case") == case
            and isinstance(row.get("checks"), Mapping)
            and bool(row["checks"])
            and all(value is True for value in row["checks"].values())
            and isinstance(row.get("learned_beats_zero_position_p95"), bool)
            for case, row in zip(cases, rows, strict=True)
        )
        and isinstance(means, Mapping)
        and set(means)
        == {
            "teacher_position_p95_m",
            "learned_position_p95_m",
            "zero_position_p95_m",
        }
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in means.values()
        )
        and isinstance(aggregates, Mapping)
        and set(aggregates)
        == {
            "all_case_checks",
            "learned_position_mean_within_teacher_budget",
            "learned_beats_zero_by_required_mean",
            "learned_beats_zero_on_majority_of_cases",
        }
        and all(value is True for value in aggregates.values())
        and report.get("passed") is True
        and report.get("ppo_authorized") is False
    )


def _mapping_matches_json_file(
    path: Path,
    value: Mapping[str, Any],
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload == dict(value)


def validate_learned_all79_admission(
    admission: Mapping[str, Any],
    *,
    identity_root: Path,
    bc_report_path: Path,
    bc_report: Mapping[str, Any],
    policy_path: Path,
    validation_report_path: Path,
    validation_report: Mapping[str, Any],
    holdout_report_path: Path,
    holdout_report: Mapping[str, Any],
    code_paths: Mapping[str, Path],
    expected_execution_commit: str,
    require_authorized: bool,
) -> None:
    validation_cases = bc_report.get("split_cases", {}).get("validation", [])
    policy_identity = bc_report.get("torchscript")
    policy_sha = (
        policy_identity.get("sha256")
        if isinstance(policy_identity, Mapping)
        else None
    )
    code = admission.get("code")
    code_valid = (
        isinstance(code, Mapping)
        and set(code) == CODE_IDENTITY_KEYS
        and set(code_paths) == CODE_IDENTITY_KEYS
        and all(
            _resolve_identity(
                code[name],
                directory=identity_root,
                expected_path=code_paths[name],
            )
            for name in CODE_IDENTITY_KEYS
        )
    )
    checks = {
        "fields": set(admission) == ADMISSION_FIELDS,
        "schema": admission.get("schema")
        == MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA,
        "bc_report": _resolve_identity(
            admission.get("bc_report"),
            directory=identity_root,
            expected_path=bc_report_path,
        )
        and _mapping_matches_json_file(bc_report_path, bc_report),
        "bc_result": bc_report.get("schema") == BC_REPORT_SCHEMA
        and bc_report.get("passed") is True
        and bc_report.get("offline_gate_passed") is True
        and bc_report.get("valid_for_dynamic_canary") is True
        and bc_report.get("training_started") is True
        and bc_report.get("ppo_authorized") is False
        and bc_report.get("learned_rollout_authorized") is False,
        "policy": _resolve_identity(
            admission.get("policy"),
            directory=identity_root,
            expected_path=policy_path,
        )
        and isinstance(policy_identity, Mapping)
        and admission.get("policy", {}).get("sha256")
        == policy_identity.get("sha256")
        and _exact_digest(policy_sha, 64),
        "validation_report": _resolve_identity(
            admission.get("validation_gate_report"),
            directory=identity_root,
            expected_path=validation_report_path,
        )
        and _mapping_matches_json_file(validation_report_path, validation_report)
        and isinstance(validation_cases, list)
        and len(validation_cases) >= 2
        and _gate_report_valid(
            validation_report,
            schema=VALIDATION_GATE_SCHEMA,
            cases=validation_cases,
            policy_sha256=str(policy_sha),
        ),
        "holdout_report": _resolve_identity(
            admission.get("holdout_gate_report"),
            directory=identity_root,
            expected_path=holdout_report_path,
        )
        and _mapping_matches_json_file(holdout_report_path, holdout_report)
        and _gate_report_valid(
            holdout_report,
            schema=HOLDOUT_GATE_SCHEMA,
            cases=DEFAULT_RESERVED_HOLDOUT_CASES,
            policy_sha256=str(policy_sha),
        ),
        "commit": admission.get("execution_commit")
        == expected_execution_commit
        == bc_report.get("execution_commit")
        and _exact_digest(expected_execution_commit, 40),
        "code": code_valid,
        "config": admission.get("evaluation_config") == DEFAULT_EVALUATION_CONFIG,
        "cases": admission.get("validation_cases") == validation_cases
        and admission.get("holdout_cases") == DEFAULT_RESERVED_HOLDOUT_CASES
        and admission.get("all79_cases") == ALL79_CASES,
        "prerequisites": admission.get("model_selection_complete") is True
        and admission.get("validation_gate_passed") is True
        and admission.get("holdout_gate_passed") is True
        and admission.get("holdout_opened_only_after_model_selection") is True,
        "downstream_closed": admission.get("residual_capture_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "authorized": not require_authorized
        or (
            admission.get("all79_evaluation_approved") is True
            and admission.get("learned_rollout_authorized") is True
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"learned all-79 admission failed: {checks}")
