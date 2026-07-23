"""Fail-closed promotion of a corrective review corpus to a BC input schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .riser_model_based_bc_loss import (
    MODEL_BASED_PROJECTED_BC_LOSS,
    REQUESTED_OUTPUT_SLEW_REGULARIZATION,
)
from .riser_model_based_corrective_corpus import (
    CORPUS_ARRAYS,
    CORPUS_METADATA_FIELDS,
    DEFAULT_RESERVED_HOLDOUT_CASES,
    MINIMUM_TRAIN_CASES,
    MINIMUM_VALIDATION_CASES,
    MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA,
    SPLIT_CODES,
    load_corpus,
    validate_corpus,
)
from .riser_residual_policy import MODEL_BASED_RESIDUAL_SAFETY_PROJECTION

MODEL_BASED_CORRECTIVE_TRAINING_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_training_admission_v1"
)
MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_training_v1"
)
TRANSITION_CONTRACT = "same_case_previous_row_elapsed_delta_v1"
CASE_BALANCING_CONTRACT = "unit_total_weight_per_case_v1"
DERIVED_ARRAYS = (
    "previous_row_index",
    "delta_time_s",
    "transition_valid",
    "case_balanced_sample_weights",
)
TRAINING_ARRAYS = CORPUS_ARRAYS + DERIVED_ARRAYS
ADMISSION_FIELDS = {
    "schema",
    "source_corpus_sha256",
    "promotion_commit",
    "loss_module_sha256",
    "loss_audit_summary_sha256",
    "promotion_module_sha256",
    "promotion_script_sha256",
    "loss_contract",
    "projection_contract",
    "requested_slew_regularization_contract",
    "minimum_train_cases",
    "minimum_validation_cases",
    "reserved_holdout_cases",
    "training_schema_promotion_approved",
    "bc_authorized",
    "ppo_authorized",
    "learned_rollout_authorized",
    "training_started",
}
TRAINING_METADATA_FIELDS = {
    "schema",
    "row_count",
    "case_count",
    "split_cases",
    "reserved_holdout_cases",
    "holdout_rows_present",
    "observation_names",
    "action_names",
    "action_scales",
    "observation_contract",
    "lookahead_horizons_s",
    "command_contract",
    "training_target_contract",
    "previous_action_contract",
    "previous_action_rebuilt",
    "requested_actions_used_as_training_targets",
    "effective_actions_used_as_training_targets",
    "trajectory_leakage",
    "source_datasets",
    "minimum_train_cases",
    "minimum_validation_cases",
    "source_review_corpus",
    "source_review_metadata_sha256",
    "promotion_admission",
    "promotion_commit",
    "promotion_module",
    "promotion_script",
    "loss_module",
    "loss_audit_summary",
    "loss_contract",
    "projection_contract",
    "requested_slew_regularization_contract",
    "transition_contract",
    "case_balancing_contract",
    "dataset_admission_passed",
    "valid_for_projection_aware_bc_input",
    "bc_authorized",
    "ppo_authorized",
    "learned_rollout_authorized",
    "training_started",
    "valid_for_training",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exact_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _portable_identity(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _review_metadata_from_training(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    review = {
        field: metadata[field]
        for field in CORPUS_METADATA_FIELDS
        if field
        not in {
            "schema",
            "dataset_admission_passed",
            "valid_for_bc_admission_review",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
            "valid_for_training",
        }
    }
    review.update(
        {
            "schema": MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA,
            "dataset_admission_passed": True,
            "valid_for_bc_admission_review": True,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
            "valid_for_training": False,
        }
    )
    return review


def validate_admission(
    admission: Mapping[str, object],
    *,
    source_corpus_sha256: str,
    loss_module_sha256: str,
    loss_audit_summary_sha256: str,
    promotion_module_sha256: str,
    promotion_script_sha256: str,
) -> None:
    if set(admission) != ADMISSION_FIELDS:
        raise ValueError("projection-aware training admission fields mismatch")
    checks = {
        "schema": admission.get("schema")
        == MODEL_BASED_CORRECTIVE_TRAINING_ADMISSION_SCHEMA,
        "source": admission.get("source_corpus_sha256") == source_corpus_sha256,
        "commit": _exact_digest(admission.get("promotion_commit"), 40),
        "loss_module": admission.get("loss_module_sha256") == loss_module_sha256,
        "loss_audit": admission.get("loss_audit_summary_sha256")
        == loss_audit_summary_sha256,
        "promotion_module": admission.get("promotion_module_sha256")
        == promotion_module_sha256,
        "promotion_script": admission.get("promotion_script_sha256")
        == promotion_script_sha256,
        "loss_contract": admission.get("loss_contract")
        == MODEL_BASED_PROJECTED_BC_LOSS,
        "projection_contract": admission.get("projection_contract")
        == MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
        "slew_contract": admission.get("requested_slew_regularization_contract")
        == REQUESTED_OUTPUT_SLEW_REGULARIZATION,
        "minimum_cases": admission.get("minimum_train_cases") == MINIMUM_TRAIN_CASES
        and admission.get("minimum_validation_cases") == MINIMUM_VALIDATION_CASES,
        "holdout": admission.get("reserved_holdout_cases")
        == DEFAULT_RESERVED_HOLDOUT_CASES,
        "promotion": admission.get("training_schema_promotion_approved") is True,
        "learning_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("learned_rollout_authorized") is False
        and admission.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"projection-aware training admission failed: {checks}")


def _validate_loss_audit(audit: Mapping[str, object]) -> None:
    checks = {
        "schema": audit.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_bc_loss_audit_v1",
        "passed": audit.get("passed") is True
        and audit.get("valid_for_bc_loss_contract_review") is True,
        "loss": audit.get("loss_contract") == MODEL_BASED_PROJECTED_BC_LOSS,
        "projection": audit.get("projection_contract")
        == MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
        "slew": audit.get("requested_slew_regularization_contract")
        == REQUESTED_OUTPUT_SLEW_REGULARIZATION,
        "learning_closed": audit.get("valid_for_training") is False
        and audit.get("bc_authorized") is False
        and audit.get("ppo_authorized") is False
        and audit.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"projection-aware loss audit failed: {checks}")


def build_training_dataset(
    corpus_path: Path,
    admission_path: Path,
    *,
    loss_module_path: Path,
    loss_audit_summary_path: Path,
    promotion_module_path: Path,
    promotion_script_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    review_metadata, review_payload = load_corpus(corpus_path)
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    loss_audit = json.loads(loss_audit_summary_path.read_text(encoding="utf-8"))
    if not isinstance(admission, dict) or not isinstance(loss_audit, dict):
        raise ValueError("projection-aware admission and loss audit must be objects")
    corpus_sha = sha256_file(corpus_path)
    loss_module_sha = sha256_file(loss_module_path)
    loss_audit_sha = sha256_file(loss_audit_summary_path)
    promotion_module_sha = sha256_file(promotion_module_path)
    promotion_script_sha = sha256_file(promotion_script_path)
    _validate_loss_audit(loss_audit)
    validate_admission(
        admission,
        source_corpus_sha256=corpus_sha,
        loss_module_sha256=loss_module_sha,
        loss_audit_summary_sha256=loss_audit_sha,
        promotion_module_sha256=promotion_module_sha,
        promotion_script_sha256=promotion_script_sha,
    )

    payload = {name: np.asarray(review_payload[name]).copy() for name in CORPUS_ARRAYS}
    row_count = int(review_metadata["row_count"])
    cases = np.asarray(payload["case_ids"], dtype=np.int64)
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    previous_row_index = np.full(row_count, -1, dtype=np.int64)
    delta_time_s = np.ones(row_count, dtype=np.float32)
    transition_valid = np.zeros(row_count, dtype=bool)
    case_balanced_weights = np.zeros(row_count, dtype=np.float32)
    for case in np.unique(cases):
        indices = np.flatnonzero(cases == case)
        previous_row_index[indices[1:]] = indices[:-1]
        delta_time_s[indices[1:]] = np.diff(elapsed[indices]).astype(np.float32)
        transition_valid[indices[1:]] = True
        case_balanced_weights[indices] = 1.0 / len(indices)
    payload.update(
        {
            "previous_row_index": previous_row_index,
            "delta_time_s": delta_time_s,
            "transition_valid": transition_valid,
            "case_balanced_sample_weights": case_balanced_weights,
        }
    )
    metadata = {
        field: review_metadata[field]
        for field in CORPUS_METADATA_FIELDS
        if field
        not in {
            "schema",
            "dataset_admission_passed",
            "valid_for_bc_admission_review",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
            "valid_for_training",
        }
    }
    metadata.update(
        {
            "schema": MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
            "source_review_corpus": _portable_identity(corpus_path),
            "source_review_metadata_sha256": _sha256_json(review_metadata),
            "promotion_admission": _portable_identity(admission_path),
            "promotion_commit": admission["promotion_commit"],
            "promotion_module": _portable_identity(promotion_module_path),
            "promotion_script": _portable_identity(promotion_script_path),
            "loss_module": _portable_identity(loss_module_path),
            "loss_audit_summary": _portable_identity(loss_audit_summary_path),
            "loss_contract": MODEL_BASED_PROJECTED_BC_LOSS,
            "projection_contract": MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
            "requested_slew_regularization_contract": (
                REQUESTED_OUTPUT_SLEW_REGULARIZATION
            ),
            "transition_contract": TRANSITION_CONTRACT,
            "case_balancing_contract": CASE_BALANCING_CONTRACT,
            "dataset_admission_passed": True,
            "valid_for_projection_aware_bc_input": True,
            "bc_authorized": False,
            "ppo_authorized": False,
            "learned_rollout_authorized": False,
            "training_started": False,
            "valid_for_training": True,
        }
    )
    validate_training_dataset(metadata, payload)
    return metadata, payload


def validate_training_dataset(
    metadata: Mapping[str, object], payload: Mapping[str, np.ndarray]
) -> None:
    if set(metadata) != TRAINING_METADATA_FIELDS:
        raise ValueError("projection-aware training metadata fields mismatch")
    if set(payload) != set(TRAINING_ARRAYS):
        raise ValueError("projection-aware training arrays mismatch")
    review_metadata = _review_metadata_from_training(metadata)
    review_payload = {name: np.asarray(payload[name]) for name in CORPUS_ARRAYS}
    validate_corpus(review_metadata, review_payload)
    if metadata.get("source_review_metadata_sha256") != _sha256_json(review_metadata):
        raise ValueError("projection-aware source review metadata mismatch")

    count = int(metadata["row_count"])
    cases = np.asarray(payload["case_ids"], dtype=np.int64)
    labels = np.asarray(payload["split_labels"], dtype=np.int64)
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    previous = np.asarray(payload["previous_row_index"])
    delta = np.asarray(payload["delta_time_s"], dtype=np.float64)
    valid = np.asarray(payload["transition_valid"])
    weights = np.asarray(payload["case_balanced_sample_weights"], dtype=np.float64)
    if (
        previous.shape != (count,)
        or previous.dtype.kind not in "iu"
        or delta.shape != (count,)
        or valid.shape != (count,)
        or valid.dtype != np.bool_
        or weights.shape != (count,)
        or not np.isfinite(delta).all()
        or not np.isfinite(weights).all()
        or np.any(delta <= 0.0)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("projection-aware derived array contract mismatch")
    for case in np.unique(cases):
        indices = np.flatnonzero(cases == case)
        if previous[indices[0]] != -1 or valid[indices[0]]:
            raise ValueError("projection-aware case initialization mismatch")
        if not np.array_equal(previous[indices[1:]], indices[:-1]):
            raise ValueError("projection-aware previous-row mapping mismatch")
        if not np.all(valid[indices[1:]]):
            raise ValueError("projection-aware transition mask mismatch")
        if not np.allclose(
            delta[indices[1:]],
            np.diff(elapsed[indices]),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("projection-aware transition timing mismatch")
        if not np.isclose(np.sum(weights[indices]), 1.0, atol=1e-6):
            raise ValueError("projection-aware case weights mismatch")
        if len(np.unique(labels[indices])) != 1:
            raise ValueError("projection-aware case split leakage")
    identity_fields = (
        "source_review_corpus",
        "promotion_admission",
        "promotion_module",
        "promotion_script",
        "loss_module",
        "loss_audit_summary",
    )
    identities_valid = all(
        isinstance(metadata.get(name), Mapping)
        and set(metadata[name]) == {"path", "sha256"}
        and isinstance(metadata[name]["path"], str)
        and bool(metadata[name]["path"])
        and _exact_digest(metadata[name]["sha256"], 64)
        for name in identity_fields
    )
    checks = {
        "schema": metadata.get("schema")
        == MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
        "commit": _exact_digest(metadata.get("promotion_commit"), 40),
        "loss": metadata.get("loss_contract") == MODEL_BASED_PROJECTED_BC_LOSS,
        "projection": metadata.get("projection_contract")
        == MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
        "slew": metadata.get("requested_slew_regularization_contract")
        == REQUESTED_OUTPUT_SLEW_REGULARIZATION,
        "transition": metadata.get("transition_contract") == TRANSITION_CONTRACT,
        "balancing": metadata.get("case_balancing_contract") == CASE_BALANCING_CONTRACT,
        "identities": identities_valid,
        "admitted": metadata.get("dataset_admission_passed") is True
        and metadata.get("valid_for_projection_aware_bc_input") is True
        and metadata.get("valid_for_training") is True,
        "learning_closed": metadata.get("bc_authorized") is False
        and metadata.get("ppo_authorized") is False
        and metadata.get("learned_rollout_authorized") is False
        and metadata.get("training_started") is False,
        "holdout": metadata.get("holdout_rows_present") is False
        and metadata.get("reserved_holdout_cases") == DEFAULT_RESERVED_HOLDOUT_CASES,
        "splits": set(np.unique(labels).tolist()) == set(SPLIT_CODES.values()),
    }
    if not all(checks.values()):
        raise ValueError(f"projection-aware training metadata failed: {checks}")


def save_training_dataset(
    path: Path, metadata: Mapping[str, object], payload: Mapping[str, np.ndarray]
) -> None:
    validate_training_dataset(metadata, payload)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite projection-aware training dataset: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(dict(metadata), sort_keys=True)),
        **{name: np.asarray(payload[name]) for name in TRAINING_ARRAYS},
    )


def load_training_dataset(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"metadata_json", *TRAINING_ARRAYS}:
            raise ValueError(
                "projection-aware training dataset archive fields mismatch"
            )
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {name: np.asarray(data[name]) for name in TRAINING_ARRAYS}
    validate_training_dataset(metadata, payload)
    return metadata, payload
