"""Fail-closed multi-case corpus for model-based corrective residual learning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .riser_model_based_corrective_dataset import (
    MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA,
    PREVIOUS_ACTION_CONTRACT,
    REQUIRED_ARRAYS as CASE_REQUIRED_ARRAYS,
    TRAINING_TARGET_CONTRACT,
    load_case_dataset,
)
from .riser_residual_dataset import (
    ACTION_NAMES,
    LOOKAHEAD_HORIZONS_S,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


MODEL_BASED_CORRECTIVE_CORPUS_MANIFEST_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_corpus_manifest_v1"
)
MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_merged_v1"
)
MINIMUM_TRAIN_CASES = 4
MINIMUM_VALIDATION_CASES = 2
SPLIT_CODES = {"train": 0, "validation": 1}
DEFAULT_RESERVED_HOLDOUT_CASES = [3, 5, 13, 19, 24]

MANIFEST_FIELDS = {
    "schema",
    "command_contract",
    "action_scales",
    "training_target_contract",
    "previous_action_contract",
    "reserved_holdout_cases",
    "case_datasets",
}
CASE_ENTRY_FIELDS = {"case", "split", "path", "sha256"}
SOURCE_DATASET_FIELDS = {
    "case",
    "split",
    "path",
    "sha256",
    "sample_count",
    "source_capture_sha256",
    "source_final_status_sha256",
    "source_runtime_commit",
    "source_plan_sha256",
    "source_corrective_profile_sha256",
    "source_paired_final_status_sha256",
}
CORPUS_ARRAYS = tuple(CASE_REQUIRED_ARRAYS) + ("source_index", "split_labels")
CORPUS_METADATA_FIELDS = {
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
    "dataset_admission_passed",
    "valid_for_bc_admission_review",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
    "valid_for_training",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _exact_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _resolve_case_path(manifest_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("case dataset path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def validate_manifest(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("model-based corrective corpus manifest fields mismatch")
    if manifest.get("schema") != MODEL_BASED_CORRECTIVE_CORPUS_MANIFEST_SCHEMA:
        raise ValueError("wrong model-based corrective corpus manifest schema")
    if manifest.get("command_contract") != MODEL_BASED_POLICY_RESIDUAL_CONTRACT:
        raise ValueError("corpus manifest command contract mismatch")
    if manifest.get("action_scales") != MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist():
        raise ValueError("corpus manifest action scales mismatch")
    if manifest.get("training_target_contract") != TRAINING_TARGET_CONTRACT:
        raise ValueError("corpus manifest training target mismatch")
    if manifest.get("previous_action_contract") != PREVIOUS_ACTION_CONTRACT:
        raise ValueError("corpus manifest previous-action contract mismatch")
    holdout = manifest.get("reserved_holdout_cases")
    if holdout != DEFAULT_RESERVED_HOLDOUT_CASES:
        raise ValueError("reserved holdout cases changed or opened")
    entries = manifest.get("case_datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("corpus manifest has no case datasets")
    normalized: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw in entries:
        if not isinstance(raw, Mapping) or set(raw) != CASE_ENTRY_FIELDS:
            raise ValueError("corpus case entry fields mismatch")
        case = raw.get("case")
        split = raw.get("split")
        if isinstance(case, bool) or not isinstance(case, int) or case <= 0:
            raise ValueError("corpus case identity must be a positive integer")
        if case in seen:
            raise ValueError("corpus contains a duplicate case")
        if case in holdout:
            raise ValueError("corpus attempts to open a reserved holdout case")
        if split not in SPLIT_CODES:
            raise ValueError("corpus split must be train or validation")
        if not _exact_sha256(raw.get("sha256")):
            raise ValueError("corpus case identity is not a SHA-256")
        seen.add(case)
        normalized.append(dict(raw))
    counts = {
        split: sum(entry["split"] == split for entry in normalized)
        for split in SPLIT_CODES
    }
    if counts["train"] < MINIMUM_TRAIN_CASES:
        raise ValueError("model-based corrective corpus needs at least four train cases")
    if counts["validation"] < MINIMUM_VALIDATION_CASES:
        raise ValueError(
            "model-based corrective corpus needs at least two validation cases"
        )
    return sorted(
        normalized,
        key=lambda item: (SPLIT_CODES[str(item["split"])], int(item["case"])),
    )


def build_corpus(
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("corpus manifest must be an object")
    entries = validate_manifest(manifest)
    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in CASE_REQUIRED_ARRAYS
    }
    source_indices: list[np.ndarray] = []
    split_labels: list[np.ndarray] = []
    source_datasets: list[dict[str, object]] = []
    split_cases = {name: [] for name in SPLIT_CODES}
    for source_index, entry in enumerate(entries):
        case = int(entry["case"])
        split = str(entry["split"])
        path = _resolve_case_path(manifest_path, entry["path"])
        actual_sha = sha256_file(path)
        if actual_sha != entry["sha256"]:
            raise ValueError(f"case {case} dataset SHA-256 mismatch")
        case_metadata, case_payload = load_case_dataset(
            path, expected_case=case, expected_split=split
        )
        if case_metadata.get("schema") != MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA:
            raise ValueError(f"case {case} dataset schema mismatch")
        count = int(case_metadata["sample_count"])
        for name in CASE_REQUIRED_ARRAYS:
            chunks[name].append(np.asarray(case_payload[name]))
        source_indices.append(np.full(count, source_index, dtype=np.int16))
        split_labels.append(np.full(count, SPLIT_CODES[split], dtype=np.int8))
        split_cases[split].append(case)
        source_datasets.append(
            {
                "case": case,
                "split": split,
                "path": str(entry["path"]),
                "sha256": actual_sha,
                "sample_count": count,
                "source_capture_sha256": case_metadata["source_capture_sha256"],
                "source_final_status_sha256": case_metadata[
                    "source_final_status_sha256"
                ],
                "source_runtime_commit": case_metadata["source_runtime_commit"],
                "source_plan_sha256": case_metadata["source_plan_sha256"],
                "source_corrective_profile_sha256": case_metadata[
                    "source_corrective_profile_sha256"
                ],
                "source_paired_final_status_sha256": case_metadata[
                    "source_paired_final_status_sha256"
                ],
            }
        )
    payload = {
        name: np.concatenate(values, axis=0) for name, values in chunks.items()
    }
    payload["source_index"] = np.concatenate(source_indices)
    payload["split_labels"] = np.concatenate(split_labels)
    metadata = {
        "schema": MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA,
        "row_count": len(payload["observations"]),
        "case_count": len(entries),
        "split_cases": split_cases,
        "reserved_holdout_cases": DEFAULT_RESERVED_HOLDOUT_CASES,
        "holdout_rows_present": False,
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "command_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "training_target_contract": TRAINING_TARGET_CONTRACT,
        "previous_action_contract": PREVIOUS_ACTION_CONTRACT,
        "previous_action_rebuilt": True,
        "requested_actions_used_as_training_targets": False,
        "effective_actions_used_as_training_targets": True,
        "trajectory_leakage": False,
        "source_datasets": source_datasets,
        "minimum_train_cases": MINIMUM_TRAIN_CASES,
        "minimum_validation_cases": MINIMUM_VALIDATION_CASES,
        "dataset_admission_passed": True,
        "valid_for_bc_admission_review": True,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    validate_corpus(metadata, payload)
    return metadata, payload


def validate_corpus(
    metadata: Mapping[str, object], payload: Mapping[str, np.ndarray]
) -> None:
    if set(metadata) != CORPUS_METADATA_FIELDS:
        raise ValueError("model-based corrective corpus metadata fields mismatch")
    if set(payload) != set(CORPUS_ARRAYS):
        raise ValueError("model-based corrective corpus arrays mismatch")
    count = len(np.asarray(payload["observations"]))
    if count < 1 or any(
        len(np.asarray(payload[name])) != count for name in CORPUS_ARRAYS
    ):
        raise ValueError("model-based corrective corpus row counts mismatch")
    for name, ndim in CASE_REQUIRED_ARRAYS.items():
        if np.asarray(payload[name]).ndim != ndim:
            raise ValueError(f"model-based corrective corpus {name} rank mismatch")
    if np.asarray(payload["source_index"]).ndim != 1:
        raise ValueError("model-based corrective corpus source index rank mismatch")
    if np.asarray(payload["split_labels"]).ndim != 1:
        raise ValueError("model-based corrective corpus split label rank mismatch")
    observations = np.asarray(payload["observations"], dtype=np.float64)
    actions = np.asarray(payload["actions"], dtype=np.float64)
    raw_cases = np.asarray(payload["case_ids"])
    raw_sources = np.asarray(payload["source_index"])
    raw_labels = np.asarray(payload["split_labels"])
    cases = raw_cases.astype(np.int64)
    sources = raw_sources.astype(np.int64)
    labels = raw_labels.astype(np.int64)
    if (
        raw_cases.dtype.kind not in "iu"
        or raw_sources.dtype.kind not in "iu"
        or raw_labels.dtype.kind not in "iu"
    ):
        raise ValueError("model-based corrective corpus identities must be integers")
    if observations.shape != (count, len(OBSERVATION_NAMES)):
        raise ValueError("model-based corrective corpus observation shape mismatch")
    if actions.shape != (count, len(ACTION_NAMES)):
        raise ValueError("model-based corrective corpus action shape mismatch")
    command_widths = {
        "requested_actions_audit": len(ACTION_NAMES),
        "effective_residual_commands": 3,
        "requested_residual_commands_audit": 3,
        "model_based_commands": 3,
        "final_high_level_commands": 3,
        "requested_vs_effective_residual_delta": 3,
        "command_clipped": 3,
    }
    for name, width in command_widths.items():
        if np.asarray(payload[name]).shape != (count, width):
            raise ValueError(f"model-based corrective corpus {name} shape mismatch")
    if not all(
        np.isfinite(np.asarray(payload[name], dtype=np.float64)).all()
        for name in CASE_REQUIRED_ARRAYS
    ):
        raise ValueError("model-based corrective corpus contains non-finite values")
    if np.max(np.abs(actions)) >= 0.95 - 1e-6:
        raise ValueError("model-based corrective corpus violates reserved action margin")
    if set(np.unique(labels).tolist()) != set(SPLIT_CODES.values()):
        raise ValueError("corpus must contain train and validation but no holdout rows")
    unique_cases = sorted(int(case) for case in np.unique(cases))
    if metadata.get("row_count") != count or metadata.get("case_count") != len(
        unique_cases
    ):
        raise ValueError("model-based corrective corpus metadata count mismatch")
    split_cases = metadata.get("split_cases")
    if not isinstance(split_cases, Mapping) or set(split_cases) != set(SPLIT_CODES):
        raise ValueError("model-based corrective corpus split metadata mismatch")
    for split, code in SPLIT_CODES.items():
        declared = sorted(int(case) for case in split_cases[split])
        observed = sorted(int(case) for case in np.unique(cases[labels == code]))
        if declared != observed:
            raise ValueError(f"model-based corrective corpus {split} cases mismatch")
    source_datasets = metadata.get("source_datasets")
    if not isinstance(source_datasets, list) or len(source_datasets) != len(
        unique_cases
    ):
        raise ValueError("model-based corrective corpus source identities mismatch")
    expected_sources = np.arange(len(source_datasets), dtype=np.int64)
    if not np.array_equal(np.unique(sources), expected_sources):
        raise ValueError("model-based corrective corpus source indices are not dense")
    seen_source_cases: set[int] = set()
    for source, identity in enumerate(source_datasets):
        if not isinstance(identity, Mapping) or set(identity) != SOURCE_DATASET_FIELDS:
            raise ValueError("model-based corrective corpus source fields mismatch")
        mask = sources == source
        if len(np.unique(cases[mask])) != 1 or len(np.unique(labels[mask])) != 1:
            raise ValueError("model-based corrective corpus source leakage")
        case = int(np.unique(cases[mask])[0])
        split_code = int(np.unique(labels[mask])[0])
        if case in seen_source_cases:
            raise ValueError("model-based corrective corpus source case is duplicated")
        if (
            identity.get("case") != case
            or identity.get("split") not in SPLIT_CODES
            or SPLIT_CODES[str(identity["split"])] != split_code
            or identity.get("sample_count") != int(np.count_nonzero(mask))
            or not isinstance(identity.get("path"), str)
            or not identity["path"]
            or not _exact_sha256(identity.get("sha256"))
            or not _exact_sha256(identity.get("source_capture_sha256"))
            or not _exact_sha256(identity.get("source_final_status_sha256"))
            or not _exact_commit(identity.get("source_runtime_commit"))
            or not _exact_sha256(identity.get("source_plan_sha256"))
            or not _exact_sha256(
                identity.get("source_corrective_profile_sha256")
            )
            or not _exact_sha256(
                identity.get("source_paired_final_status_sha256")
            )
        ):
            raise ValueError("model-based corrective corpus source identity mismatch")
        seen_source_cases.add(case)
    for case in unique_cases:
        if case in DEFAULT_RESERVED_HOLDOUT_CASES:
            raise ValueError("model-based corrective corpus contains a holdout row")
        mask = cases == case
        indices = np.flatnonzero(mask)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"model-based corrective case {case} rows are not contiguous")
        previous = observations[mask][:, PREVIOUS_ACTION_INDICES]
        case_actions = actions[mask]
        if not np.allclose(previous[0], 0.0, atol=1e-12) or not np.allclose(
            previous[1:], case_actions[:-1], rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"model-based corrective case {case} recurrence mismatch")
        elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)[mask]
        execution = np.asarray(payload["execution_time_s"], dtype=np.float64)[mask]
        source_time = np.asarray(payload["source_time_s"], dtype=np.float64)[mask]
        if abs(float(elapsed[0])) > 1e-9 or not np.all(np.diff(elapsed) > 0.0):
            raise ValueError(f"model-based corrective case {case} elapsed clock mismatch")
        if not np.all(np.diff(execution) >= 0.0) or not np.all(
            np.diff(source_time) >= 0.0
        ):
            raise ValueError(f"model-based corrective case {case} clock mismatch")
    effective_residual = np.asarray(
        payload["effective_residual_commands"], dtype=np.float64
    )
    requested_residual = np.asarray(
        payload["requested_residual_commands_audit"], dtype=np.float64
    )
    requested_actions = np.asarray(
        payload["requested_actions_audit"], dtype=np.float64
    )
    model_commands = np.asarray(payload["model_based_commands"], dtype=np.float64)
    final_commands = np.asarray(
        payload["final_high_level_commands"], dtype=np.float64
    )
    delta = np.asarray(
        payload["requested_vs_effective_residual_delta"], dtype=np.float64
    )
    expected_delta = effective_residual - requested_residual
    if not np.allclose(
        effective_residual,
        actions * MODEL_BASED_POLICY_RESIDUAL_SCALES,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("model-based corrective corpus effective labels mismatch")
    if not np.allclose(
        requested_residual,
        requested_actions * MODEL_BASED_POLICY_RESIDUAL_SCALES,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("model-based corrective corpus requested audit mismatch")
    if not np.allclose(
        final_commands, model_commands + effective_residual, rtol=0.0, atol=2e-7
    ):
        raise ValueError("model-based corrective corpus final command mismatch")
    if not np.allclose(delta, expected_delta, rtol=0.0, atol=2e-7):
        raise ValueError("model-based corrective corpus supervisor delta mismatch")
    if not np.array_equal(
        np.asarray(payload["command_clipped"], dtype=bool),
        np.abs(expected_delta) > 2e-7,
    ):
        raise ValueError("model-based corrective corpus clipping mask mismatch")
    checks = {
        "schema": metadata.get("schema") == MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA,
        "holdout": metadata.get("reserved_holdout_cases")
        == DEFAULT_RESERVED_HOLDOUT_CASES
        and metadata.get("holdout_rows_present") is False,
        "names": metadata.get("observation_names") == list(OBSERVATION_NAMES)
        and metadata.get("action_names") == list(ACTION_NAMES),
        "scales": metadata.get("action_scales")
        == MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "observation": metadata.get("observation_contract")
        == "executed_state_with_execution_time_lookahead_v2"
        and metadata.get("lookahead_horizons_s") == list(LOOKAHEAD_HORIZONS_S),
        "semantics": metadata.get("command_contract")
        == MODEL_BASED_POLICY_RESIDUAL_CONTRACT
        and metadata.get("training_target_contract") == TRAINING_TARGET_CONTRACT
        and metadata.get("previous_action_contract") == PREVIOUS_ACTION_CONTRACT
        and metadata.get("previous_action_rebuilt") is True,
        "effective_only": metadata.get("requested_actions_used_as_training_targets")
        is False
        and metadata.get("effective_actions_used_as_training_targets") is True,
        "counts": metadata.get("minimum_train_cases") == MINIMUM_TRAIN_CASES
        and metadata.get("minimum_validation_cases") == MINIMUM_VALIDATION_CASES
        and len(split_cases["train"]) >= MINIMUM_TRAIN_CASES
        and len(split_cases["validation"]) >= MINIMUM_VALIDATION_CASES,
        "sources": seen_source_cases == set(unique_cases),
        "closed": metadata.get("trajectory_leakage") is False
        and metadata.get("dataset_admission_passed") is True
        and metadata.get("valid_for_bc_admission_review") is True
        and metadata.get("bc_authorized") is False
        and metadata.get("ppo_authorized") is False
        and metadata.get("training_started") is False
        and metadata.get("valid_for_training") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"model-based corrective corpus metadata failed: {checks}")


def save_corpus(
    path: Path, metadata: Mapping[str, object], payload: Mapping[str, np.ndarray]
) -> None:
    validate_corpus(metadata, payload)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite corrective corpus: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(dict(metadata), sort_keys=True)),
        **{name: np.asarray(payload[name]) for name in CORPUS_ARRAYS},
    )


def load_corpus(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"metadata_json", *CORPUS_ARRAYS}:
            raise ValueError("model-based corrective corpus archive fields mismatch")
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {name: np.asarray(data[name]) for name in CORPUS_ARRAYS}
    validate_corpus(metadata, payload)
    return metadata, payload
