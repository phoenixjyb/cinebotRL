"""Hash-bound admission and report contracts for projection-aware BC."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .riser_model_based_bc_loss import MODEL_BASED_PROJECTED_BC_LOSS
from .riser_model_based_corrective_bc_adapter import (
    MODEL_BASED_CORRECTIVE_BC_OPTIMIZER_CONTRACT,
    MODEL_BASED_CORRECTIVE_BC_VALIDATION_CONTRACT,
)
from .riser_model_based_corrective_corpus import (
    DEFAULT_RESERVED_HOLDOUT_CASES,
    MINIMUM_TRAIN_CASES,
    MINIMUM_VALIDATION_CASES,
)
from .riser_model_based_corrective_training_dataset import (
    MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
)
from .riser_residual_dataset import (
    ACTION_NAMES,
    BASE_OBSERVATION_NAMES,
    LOOKAHEAD_CHANNEL_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
)
from .riser_residual_policy import (
    MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
    MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE,
)

MODEL_BASED_CORRECTIVE_BC_EXECUTION_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_admission_v1"
)
MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_report_v1"
)
DEFAULT_BC_TRAINING_CONFIG = {
    "policy_architecture": (
        MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
    ),
    "observation_dimension": len(OBSERVATION_NAMES),
    "base_observation_dimension": len(BASE_OBSERVATION_NAMES),
    "lookahead_horizon_count": len(LOOKAHEAD_HORIZONS_S),
    "lookahead_channel_count": len(LOOKAHEAD_CHANNEL_NAMES),
    "action_dimension": len(ACTION_NAMES),
    "zero_initialize_action_head": True,
    "optimizer": "AdamW",
    "epochs_max": 200,
    "patience": 20,
    "batch_size": 4096,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "gradient_clip_norm": 5.0,
    "state_hidden_sizes": [128, 128],
    "lookahead_hidden_sizes": [64, 64],
    "fusion_hidden_sizes": [256, 128],
    "seed": 20260723,
    "device": "cuda",
    "minimum_improvement_fraction": 0.05,
    "maximum_normalized_prediction_abs": 0.95,
    "model_selection_split": "validation",
    "optimizer_steps_per_epoch": 1,
}
CODE_IDENTITY_KEYS = {
    "trainer",
    "adapter",
    "loss_module",
    "policy_module",
    "training_dataset_module",
    "admission_module",
}
IDENTITY_FIELDS = {"path", "sha256"}
ADMISSION_FIELDS = {
    "schema",
    "dataset",
    "dataset_schema",
    "dataset_promotion_commit",
    "execution_commit",
    "code",
    "optimizer_contract",
    "validation_contract",
    "loss_contract",
    "projection_contract",
    "training_config",
    "split_cases",
    "reserved_holdout_cases",
    "holdout_opened",
    "bc_execution_approved",
    "bc_authorized",
    "ppo_authorized",
    "learned_rollout_authorized",
    "training_started",
}
REPORT_FIELDS = {
    "schema",
    "admission",
    "dataset",
    "execution_commit",
    "optimizer_contract",
    "validation_contract",
    "loss_contract",
    "training_config",
    "split_cases",
    "reserved_holdout_cases",
    "epochs_run",
    "optimizer_steps",
    "best_epoch",
    "history",
    "split_metrics",
    "offline_gate_passed",
    "holdout_used_for_model_selection",
    "holdout_metrics_computed",
    "checkpoint",
    "torchscript",
    "bc_authorized",
    "training_started",
    "ppo_authorized",
    "learned_rollout_authorized",
    "passed",
    "valid_for_dynamic_canary",
}
HISTORY_FIELDS = {
    "epoch",
    "train_loss_total",
    "train_loss_pointwise",
    "train_loss_requested_slew",
    "validation_loss_total",
}
SPLIT_METRIC_FIELDS = {
    "loss_total",
    "case_balanced_mse_per_action",
    "zero_requested_case_balanced_mse_per_action",
    "improves_over_zero_requested",
    "requested_action_abs_max",
    "requested_slew_violation_count",
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
        and all(char in "0123456789abcdef" for char in value)
    )


def _identity_matches(identity: object, path: Path) -> bool:
    return (
        isinstance(identity, Mapping)
        and set(identity) == IDENTITY_FIELDS
        and isinstance(identity.get("path"), str)
        and bool(identity.get("path"))
        and path.is_file()
        and identity.get("sha256") == sha256_file(path)
    )


def _split_contract_valid(
    split_cases: object,
    reserved_holdout_cases: object,
) -> bool:
    if (
        not isinstance(split_cases, Mapping)
        or set(split_cases) != {"train", "validation"}
        or reserved_holdout_cases != DEFAULT_RESERVED_HOLDOUT_CASES
    ):
        return False
    train = split_cases["train"]
    validation = split_cases["validation"]
    if not isinstance(train, list) or not isinstance(validation, list):
        return False
    all_cases = [*train, *validation]
    return (
        len(train) >= MINIMUM_TRAIN_CASES
        and len(validation) >= MINIMUM_VALIDATION_CASES
        and all(isinstance(case, int) and case > 0 for case in all_cases)
        and len(set(all_cases)) == len(all_cases)
        and not set(all_cases).intersection(DEFAULT_RESERVED_HOLDOUT_CASES)
    )


def validate_bc_execution_admission(
    admission: Mapping[str, object],
    *,
    dataset_path: Path,
    dataset_metadata: Mapping[str, object],
    code_paths: Mapping[str, Path],
    expected_execution_commit: str,
    require_authorized: bool,
) -> None:
    if set(admission) != ADMISSION_FIELDS:
        raise ValueError("projection-aware BC admission fields mismatch")
    code = admission.get("code")
    code_valid = (
        isinstance(code, Mapping)
        and set(code) == CODE_IDENTITY_KEYS
        and set(code_paths) == CODE_IDENTITY_KEYS
        and all(_identity_matches(code[name], code_paths[name]) for name in code)
    )
    split_cases = dataset_metadata.get("split_cases")
    checks = {
        "schema": admission.get("schema")
        == MODEL_BASED_CORRECTIVE_BC_EXECUTION_ADMISSION_SCHEMA,
        "dataset": _identity_matches(admission.get("dataset"), dataset_path),
        "dataset_schema": admission.get("dataset_schema")
        == MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA
        == dataset_metadata.get("schema"),
        "promotion_commit": admission.get("dataset_promotion_commit")
        == dataset_metadata.get("promotion_commit")
        and _exact_digest(admission.get("dataset_promotion_commit"), 40),
        "execution_commit": admission.get("execution_commit")
        == expected_execution_commit
        and _exact_digest(expected_execution_commit, 40),
        "code": code_valid,
        "optimizer": admission.get("optimizer_contract")
        == MODEL_BASED_CORRECTIVE_BC_OPTIMIZER_CONTRACT,
        "validation": admission.get("validation_contract")
        == MODEL_BASED_CORRECTIVE_BC_VALIDATION_CONTRACT,
        "loss": admission.get("loss_contract") == MODEL_BASED_PROJECTED_BC_LOSS,
        "projection": admission.get("projection_contract")
        == MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
        "config": admission.get("training_config") == DEFAULT_BC_TRAINING_CONFIG,
        "splits": admission.get("split_cases") == split_cases
        and _split_contract_valid(
            admission.get("split_cases"),
            admission.get("reserved_holdout_cases"),
        ),
        "dataset_ready": dataset_metadata.get(
            "valid_for_projection_aware_bc_input"
        )
        is True
        and dataset_metadata.get("valid_for_training") is True
        and dataset_metadata.get("bc_authorized") is False
        and dataset_metadata.get("ppo_authorized") is False
        and dataset_metadata.get("learned_rollout_authorized") is False
        and dataset_metadata.get("training_started") is False,
        "holdout": admission.get("holdout_opened") is False,
        "authorization_consistent": admission.get("bc_execution_approved")
        is admission.get("bc_authorized")
        and isinstance(admission.get("bc_authorized"), bool),
        "execution_closed": admission.get("ppo_authorized") is False
        and admission.get("learned_rollout_authorized") is False
        and admission.get("training_started") is False,
        "authorized": not require_authorized
        or (
            admission.get("bc_execution_approved") is True
            and admission.get("bc_authorized") is True
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"projection-aware BC admission failed: {checks}")


def _artifact_identity_valid(identity: object, report_directory: Path) -> bool:
    if not isinstance(identity, Mapping) or set(identity) != IDENTITY_FIELDS:
        return False
    value = identity.get("path")
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    path = path if path.is_absolute() else report_directory / path
    return path.is_file() and identity.get("sha256") == sha256_file(path)


def _finite_vector(value: object, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(array).all())


def _boolean_vector(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, bool) for item in value)
    )


def validate_bc_execution_report(
    report: Mapping[str, object],
    *,
    admission_path: Path,
    admission: Mapping[str, object],
    report_directory: Path,
) -> None:
    if set(report) != REPORT_FIELDS:
        raise ValueError("projection-aware BC report fields mismatch")
    history = report.get("history")
    epochs_run = report.get("epochs_run")
    history_valid = (
        isinstance(history, list)
        and isinstance(epochs_run, int)
        and epochs_run > 0
        and len(history) == epochs_run
    )
    if history_valid:
        for expected_epoch, item in enumerate(history, start=1):
            if (
                not isinstance(item, Mapping)
                or set(item) != HISTORY_FIELDS
                or item.get("epoch") != expected_epoch
                or not all(
                    isinstance(item.get(name), (int, float))
                    and np.isfinite(item[name])
                    and item[name] >= 0.0
                    for name in HISTORY_FIELDS - {"epoch"}
                )
            ):
                history_valid = False
                break
    metrics = report.get("split_metrics")
    metrics_valid = isinstance(metrics, Mapping) and set(metrics) == {
        "train",
        "validation",
    }
    if metrics_valid:
        for value in metrics.values():
            if (
                not isinstance(value, Mapping)
                or set(value) != SPLIT_METRIC_FIELDS
                or not isinstance(value.get("loss_total"), (int, float))
                or not np.isfinite(value["loss_total"])
                or value["loss_total"] < 0.0
                or not _finite_vector(value.get("case_balanced_mse_per_action"), 3)
                or not _finite_vector(
                    value.get("zero_requested_case_balanced_mse_per_action"), 3
                )
                or not _boolean_vector(
                    value.get("improves_over_zero_requested"), 3
                )
                or not _finite_vector(value.get("requested_action_abs_max"), 3)
                or not _finite_vector(value.get("requested_slew_violation_count"), 3)
            ):
                metrics_valid = False
                break
    successful = report.get("offline_gate_passed") is True
    artifacts_valid = (
        _artifact_identity_valid(report.get("checkpoint"), report_directory)
        and _artifact_identity_valid(report.get("torchscript"), report_directory)
        if successful
        else report.get("checkpoint") is None and report.get("torchscript") is None
    )
    best_epoch = report.get("best_epoch")
    validation = metrics.get("validation", {}) if isinstance(metrics, Mapping) else {}
    validation_improves = validation.get("improves_over_zero_requested") == [
        True,
        True,
        True,
    ]
    requested_margin = (
        _finite_vector(validation.get("requested_action_abs_max"), 3)
        and max(validation["requested_action_abs_max"])
        < DEFAULT_BC_TRAINING_CONFIG["maximum_normalized_prediction_abs"]
    )
    validation_slew_safe = validation.get("requested_slew_violation_count") == [
        0,
        0,
        0,
    ]
    checks = {
        "schema": report.get("schema")
        == MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA,
        "admission": _identity_matches(report.get("admission"), admission_path),
        "dataset": report.get("dataset") == admission.get("dataset"),
        "commit": report.get("execution_commit") == admission.get("execution_commit"),
        "contracts": report.get("optimizer_contract")
        == admission.get("optimizer_contract")
        and report.get("validation_contract") == admission.get("validation_contract")
        and report.get("loss_contract") == admission.get("loss_contract"),
        "config": report.get("training_config") == admission.get("training_config"),
        "splits": report.get("split_cases") == admission.get("split_cases")
        and report.get("reserved_holdout_cases")
        == admission.get("reserved_holdout_cases"),
        "history": history_valid
        and isinstance(report.get("optimizer_steps"), int)
        and report.get("optimizer_steps") == epochs_run
        and isinstance(best_epoch, int)
        and 1 <= best_epoch <= epochs_run,
        "metrics": metrics_valid,
        "holdout": report.get("holdout_used_for_model_selection") is False
        and report.get("holdout_metrics_computed") is False,
        "authorized": admission.get("bc_authorized") is True
        and report.get("bc_authorized") is True,
        "training": report.get("training_started") is True,
        "downstream_closed": report.get("ppo_authorized") is False
        and report.get("learned_rollout_authorized") is False,
        "success_consistent": report.get("passed") is successful
        and report.get("valid_for_dynamic_canary") is successful,
        "success_gates": not successful
        or (validation_improves and requested_margin and validation_slew_safe),
        "artifacts": artifacts_valid,
    }
    if not all(checks.values()):
        raise ValueError(f"projection-aware BC report failed: {checks}")
