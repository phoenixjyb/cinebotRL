"""Hash-bound admission contract for learned validation and holdout rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .riser_model_based_corrective_bc_contract import (
    MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA,
    sha256_file,
    validate_bc_execution_report,
)
from .riser_model_based_corrective_corpus import DEFAULT_RESERVED_HOLDOUT_CASES
from .riser_model_based_learned_all79_contract import (
    DEFAULT_EVALUATION_CONFIG,
    VALIDATION_GATE_SCHEMA,
    _exact_digest,
    _exact_source_manifest_valid,
    _gate_report_valid,
    _mapping_matches_json_file,
    _plan_manifest_valid,
    _resolve_identity,
)


MODEL_BASED_LEARNED_SPLIT_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_learned_split_admission_v1"
)
SPLIT_MODES = {"validation_canary", "holdout"}
CODE_IDENTITY_KEYS = {
    "playback",
    "rollout_gate",
    "admission_contract",
    "preflight_validator",
    "execution_wrapper",
}
ADMISSION_FIELDS = {
    "schema",
    "mode",
    "bc_report",
    "policy",
    "plan_manifest",
    "source_manifest",
    "lqr_gains",
    "robot_build_audit",
    "robot_usd",
    "drive_profile_selection",
    "prior_validation_gate_report",
    "execution_commit",
    "code",
    "evaluation_config",
    "cases",
    "model_selection_complete",
    "prior_validation_gate_passed",
    "split_evaluation_approved",
    "learned_rollout_authorized",
    "residual_capture_authorized",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
}


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _identity_path(identity: object, directory: Path) -> Path | None:
    if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
        return None
    value = identity.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else directory / path
    return path.resolve()


def _bc_report_valid(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> bool:
    admission_path = _identity_path(
        report.get("admission"),
        report_path.parent,
    )
    if admission_path is None or not admission_path.is_file():
        return False
    admission = _load_json_object(admission_path)
    if admission is None:
        return False
    try:
        validate_bc_execution_report(
            report,
            admission_path=admission_path,
            admission=admission,
            report_directory=report_path.parent,
        )
    except ValueError:
        return False
    return (
        report.get("schema") == MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA
        and report.get("passed") is True
        and report.get("offline_gate_passed") is True
        and report.get("valid_for_dynamic_canary") is True
        and report.get("training_started") is True
        and report.get("ppo_authorized") is False
        and report.get("learned_rollout_authorized") is False
    )


def validate_learned_split_admission(
    admission: Mapping[str, Any],
    *,
    identity_root: Path,
    mode: str,
    bc_report_path: Path,
    bc_report: Mapping[str, Any],
    policy_path: Path,
    plan_manifest_path: Path,
    source_manifest_path: Path,
    lqr_gains_path: Path,
    robot_build_audit_path: Path,
    robot_usd_path: Path,
    drive_profile_selection_path: Path,
    prior_validation_report_path: Path | None,
    prior_validation_report: Mapping[str, Any] | None,
    code_paths: Mapping[str, Path],
    expected_execution_commit: str,
    require_authorized: bool,
) -> None:
    if mode not in SPLIT_MODES:
        raise ValueError("unknown learned split mode")
    validation_cases = bc_report.get("split_cases", {}).get("validation", [])
    policy_identity = bc_report.get("torchscript")
    policy_sha = (
        policy_identity.get("sha256")
        if isinstance(policy_identity, Mapping)
        else None
    )
    expected_cases = (
        validation_cases
        if mode == "validation_canary"
        else DEFAULT_RESERVED_HOLDOUT_CASES
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
    prior_validation_valid = (
        admission.get("prior_validation_gate_report") is None
        and prior_validation_report_path is None
        and prior_validation_report is None
        if mode == "validation_canary"
        else (
            prior_validation_report_path is not None
            and prior_validation_report is not None
            and _resolve_identity(
                admission.get("prior_validation_gate_report"),
                directory=identity_root,
                expected_path=prior_validation_report_path,
            )
            and _mapping_matches_json_file(
                prior_validation_report_path,
                prior_validation_report,
            )
            and isinstance(validation_cases, list)
            and len(validation_cases) >= 2
            and _gate_report_valid(
                prior_validation_report,
                schema=VALIDATION_GATE_SCHEMA,
                cases=validation_cases,
                policy_sha256=str(policy_sha),
                report_directory=prior_validation_report_path.parent,
            )
        )
    )
    checks = {
        "fields": set(admission) == ADMISSION_FIELDS,
        "schema": admission.get("schema")
        == MODEL_BASED_LEARNED_SPLIT_ADMISSION_SCHEMA,
        "mode": admission.get("mode") == mode,
        "bc_report": _resolve_identity(
            admission.get("bc_report"),
            directory=identity_root,
            expected_path=bc_report_path,
        )
        and _mapping_matches_json_file(bc_report_path, bc_report)
        and _bc_report_valid(bc_report, report_path=bc_report_path),
        "policy": _resolve_identity(
            admission.get("policy"),
            directory=identity_root,
            expected_path=policy_path,
        )
        and isinstance(policy_identity, Mapping)
        and admission.get("policy", {}).get("sha256")
        == policy_identity.get("sha256")
        and _exact_digest(policy_sha, 64),
        "source_manifest": _resolve_identity(
            admission.get("source_manifest"),
            directory=identity_root,
            expected_path=source_manifest_path,
        )
        and _exact_source_manifest_valid(source_manifest_path),
        "plan_manifest": _resolve_identity(
            admission.get("plan_manifest"),
            directory=identity_root,
            expected_path=plan_manifest_path,
        )
        and _plan_manifest_valid(
            plan_manifest_path,
            source_manifest_path=source_manifest_path,
        ),
        "runtime_assets": all(
            _resolve_identity(
                admission.get(name),
                directory=identity_root,
                expected_path=path,
            )
            for name, path in {
                "lqr_gains": lqr_gains_path,
                "robot_build_audit": robot_build_audit_path,
                "robot_usd": robot_usd_path,
                "drive_profile_selection": drive_profile_selection_path,
            }.items()
        ),
        "prior_validation": prior_validation_valid,
        "commit": admission.get("execution_commit")
        == expected_execution_commit
        == bc_report.get("execution_commit")
        and _exact_digest(expected_execution_commit, 40),
        "code": code_valid,
        "config": admission.get("evaluation_config") == DEFAULT_EVALUATION_CONFIG,
        "cases": isinstance(expected_cases, list)
        and len(expected_cases) >= 2
        and admission.get("cases") == expected_cases
        and not set(validation_cases).intersection(DEFAULT_RESERVED_HOLDOUT_CASES),
        "stage": (
            admission.get("model_selection_complete") is False
            and admission.get("prior_validation_gate_passed") is False
            if mode == "validation_canary"
            else (
                admission.get("model_selection_complete") is True
                and admission.get("prior_validation_gate_passed") is True
            )
        ),
        "downstream_closed": admission.get("residual_capture_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "authorized": not require_authorized
        or (
            admission.get("split_evaluation_approved") is True
            and admission.get("learned_rollout_authorized") is True
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"learned split admission failed: {checks}")


def admission_identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}
