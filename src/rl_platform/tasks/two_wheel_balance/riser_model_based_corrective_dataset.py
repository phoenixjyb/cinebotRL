"""Fail-closed case-dataset conversion for model-based corrective captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .riser_corrective_capture import (
    CORRECTIVE_CAPTURE_CASE,
    CORRECTIVE_CAPTURE_SCHEMA,
    CORRECTIVE_CAPTURE_SPLIT,
    load_corrective_capture,
    validate_corrective_route,
)
from .riser_residual_dataset import (
    ACTION_NAMES,
    MODEL_BASED_POLICY_PREVIOUS_ACTION_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_case_dataset_v1"
)
CAPTURE_FINAL_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_final_v2"
)
TRAINING_TARGET_CONTRACT = "effective_post_supervisor_residual_v1"
PREVIOUS_ACTION_CONTRACT = MODEL_BASED_POLICY_PREVIOUS_ACTION_CONTRACT

REQUIRED_ARRAYS = {
    "observations": 2,
    "actions": 2,
    "requested_actions_audit": 2,
    "effective_residual_commands": 2,
    "requested_residual_commands_audit": 2,
    "model_based_commands": 2,
    "final_high_level_commands": 2,
    "requested_vs_effective_residual_delta": 2,
    "command_clipped": 2,
    "case_ids": 1,
    "elapsed_time_s": 1,
    "execution_time_s": 1,
    "source_time_s": 1,
}

REQUIRED_METADATA = {
    "schema",
    "case",
    "split",
    "sample_count",
    "observation_names",
    "action_names",
    "action_scales",
    "observation_contract",
    "command_contract",
    "training_target_contract",
    "previous_action_contract",
    "previous_action_rebuilt",
    "source_capture_schema",
    "source_capture_sha256",
    "source_final_status_sha256",
    "source_runtime_commit",
    "source_plan_sha256",
    "source_corrective_profile_sha256",
    "source_paired_final_status_sha256",
    "requested_actions_used_as_training_targets",
    "effective_actions_used_as_training_targets",
    "valid_for_case_merge",
    "merged_dataset_created",
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


def _exact_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_conversion_source(
    final_status: Mapping[str, object],
    *,
    capture_path: Path,
    capture_metadata: Mapping[str, object],
    expected_case: int = CORRECTIVE_CAPTURE_CASE,
    expected_split: str = CORRECTIVE_CAPTURE_SPLIT,
) -> None:
    validate_corrective_route(expected_case, expected_split)
    capture_identity = final_status.get("capture")
    if not isinstance(capture_identity, Mapping):
        capture_identity = {}
    gate_checks = final_status.get("gate_checks")
    archive_checks = final_status.get("archive_checks")
    checks = {
        "schema": final_status.get("schema") == CAPTURE_FINAL_SCHEMA,
        "case_split": final_status.get("case") == expected_case
        and final_status.get("split") == expected_split
        and capture_metadata.get("case") == expected_case
        and capture_metadata.get("split") == expected_split,
        "capture_admitted": final_status.get(
            "capture_admitted_for_dataset_conversion"
        )
        is True
        and final_status.get("passed") is True,
        "dynamic": final_status.get("dynamic_quality_passed") is True,
        "gate_checks": isinstance(gate_checks, Mapping)
        and bool(gate_checks)
        and all(value is True for value in gate_checks.values()),
        "archive_checks": isinstance(archive_checks, Mapping)
        and bool(archive_checks)
        and all(value is True for value in archive_checks.values()),
        "capture_identity": capture_identity.get("sha256")
        == sha256_file(capture_path),
        "runtime_identity": final_status.get("runtime_commit")
        == capture_metadata.get("runtime_commit"),
        "training_closed": final_status.get("normalized_training_dataset_created")
        is False
        and final_status.get("bc_authorized") is False
        and final_status.get("ppo_authorized") is False
        and final_status.get("training_started") is False
        and final_status.get("valid_for_training") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"corrective capture is not admitted for conversion: {checks}")


def validate_case_dataset(
    metadata: Mapping[str, object],
    payload: Mapping[str, np.ndarray],
    *,
    expected_case: int = CORRECTIVE_CAPTURE_CASE,
    expected_split: str = CORRECTIVE_CAPTURE_SPLIT,
) -> None:
    validate_corrective_route(expected_case, expected_split)
    if set(metadata) != REQUIRED_METADATA:
        raise ValueError("model-based corrective dataset metadata fields mismatch")
    for name, ndim in REQUIRED_ARRAYS.items():
        if name not in payload or np.asarray(payload[name]).ndim != ndim:
            raise ValueError(f"missing or invalid converted dataset field {name}")
    count = len(np.asarray(payload["observations"]))
    if count < 2 or any(len(np.asarray(payload[name])) != count for name in REQUIRED_ARRAYS):
        raise ValueError("converted dataset row counts do not match")
    if metadata.get("sample_count") != count:
        raise ValueError("converted dataset metadata row count mismatch")

    widths = {
        "observations": len(OBSERVATION_NAMES),
        "actions": len(ACTION_NAMES),
        "requested_actions_audit": len(ACTION_NAMES),
        "effective_residual_commands": 3,
        "requested_residual_commands_audit": 3,
        "model_based_commands": 3,
        "final_high_level_commands": 3,
        "requested_vs_effective_residual_delta": 3,
        "command_clipped": 3,
    }
    for name, width in widths.items():
        if np.asarray(payload[name]).shape[1] != width:
            raise ValueError(f"converted dataset {name} dimension mismatch")
    if not all(
        np.isfinite(np.asarray(payload[name], dtype=np.float64)).all()
        for name in REQUIRED_ARRAYS
    ):
        raise ValueError("converted dataset contains non-finite values")

    actions = np.asarray(payload["actions"], dtype=np.float64)
    requested_actions = np.asarray(payload["requested_actions_audit"], dtype=np.float64)
    effective_residual = np.asarray(
        payload["effective_residual_commands"], dtype=np.float64
    )
    requested_residual = np.asarray(
        payload["requested_residual_commands_audit"], dtype=np.float64
    )
    model = np.asarray(payload["model_based_commands"], dtype=np.float64)
    final = np.asarray(payload["final_high_level_commands"], dtype=np.float64)
    delta = np.asarray(
        payload["requested_vs_effective_residual_delta"], dtype=np.float64
    )
    expected_delta = effective_residual - requested_residual
    if np.max(np.abs(actions)) >= 0.95 - 1e-6:
        raise ValueError("effective training target violates reserved action margin")
    if not np.allclose(
        effective_residual,
        actions * MODEL_BASED_POLICY_RESIDUAL_SCALES,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("effective training target does not reconstruct")
    if not np.allclose(
        requested_residual,
        requested_actions * MODEL_BASED_POLICY_RESIDUAL_SCALES,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("requested audit action does not reconstruct")
    if not np.allclose(final, model + effective_residual, rtol=0.0, atol=2e-7):
        raise ValueError("converted final command does not reconstruct")
    if not np.allclose(delta, expected_delta, rtol=0.0, atol=2e-7):
        raise ValueError("converted requested/effective delta mismatch")
    if not np.array_equal(
        np.asarray(payload["command_clipped"], dtype=bool),
        np.abs(expected_delta) > 2e-7,
    ):
        raise ValueError("converted clipping mask mismatch")

    observations = np.asarray(payload["observations"], dtype=np.float64)
    previous = observations[:, PREVIOUS_ACTION_INDICES]
    if not np.allclose(previous[0], 0.0, atol=1e-12) or not np.allclose(
        previous[1:], actions[:-1], rtol=0.0, atol=1e-7
    ):
        raise ValueError("effective previous-action recurrence mismatch")
    cases = np.unique(np.asarray(payload["case_ids"]))
    if cases.tolist() != [expected_case]:
        raise ValueError("converted dataset opens an unreviewed case")
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    execution = np.asarray(payload["execution_time_s"], dtype=np.float64)
    source = np.asarray(payload["source_time_s"], dtype=np.float64)
    if abs(float(elapsed[0])) > 1e-9 or not np.all(np.diff(elapsed) > 0.0):
        raise ValueError("converted elapsed clock must start at zero and increase")
    if not np.all(np.diff(execution) >= 0.0) or not np.all(np.diff(source) >= 0.0):
        raise ValueError("converted source/execution clocks must be monotonic")

    metadata_checks = {
        "schema": metadata.get("schema")
        == MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA,
        "case_split": metadata.get("case") == expected_case
        and metadata.get("split") == expected_split,
        "names": metadata.get("observation_names") == list(OBSERVATION_NAMES)
        and metadata.get("action_names") == list(ACTION_NAMES),
        "scales": metadata.get("action_scales")
        == MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "observation": metadata.get("observation_contract")
        == "executed_state_with_execution_time_lookahead_v2",
        "command": metadata.get("command_contract")
        == MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "target": metadata.get("training_target_contract")
        == TRAINING_TARGET_CONTRACT,
        "previous": metadata.get("previous_action_contract")
        == PREVIOUS_ACTION_CONTRACT
        and metadata.get("previous_action_rebuilt") is True,
        "source_schema": metadata.get("source_capture_schema")
        == CORRECTIVE_CAPTURE_SCHEMA,
        "source_hashes": _exact_digest(metadata.get("source_capture_sha256"), 64)
        and _exact_digest(metadata.get("source_final_status_sha256"), 64)
        and _exact_digest(metadata.get("source_runtime_commit"), 40)
        and _exact_digest(metadata.get("source_plan_sha256"), 64)
        and _exact_digest(
            metadata.get("source_corrective_profile_sha256"), 64
        )
        and _exact_digest(
            metadata.get("source_paired_final_status_sha256"), 64
        ),
        "effective_only": metadata.get(
            "requested_actions_used_as_training_targets"
        )
        is False
        and metadata.get("effective_actions_used_as_training_targets") is True,
        "closed": metadata.get("valid_for_case_merge") is True
        and metadata.get("merged_dataset_created") is False
        and metadata.get("bc_authorized") is False
        and metadata.get("ppo_authorized") is False
        and metadata.get("training_started") is False
        and metadata.get("valid_for_training") is False,
    }
    if not all(metadata_checks.values()):
        raise ValueError(f"invalid converted dataset metadata: {metadata_checks}")


def convert_admitted_capture(
    capture_path: Path,
    final_status_path: Path,
    *,
    expected_case: int = CORRECTIVE_CAPTURE_CASE,
    expected_split: str = CORRECTIVE_CAPTURE_SPLIT,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    capture_metadata, capture = load_corrective_capture(
        capture_path,
        expected_case=expected_case,
        expected_split=expected_split,
    )
    final_status = json.loads(final_status_path.read_text(encoding="utf-8"))
    if not isinstance(final_status, dict):
        raise ValueError("capture final status must be an object")
    validate_conversion_source(
        final_status,
        capture_path=capture_path,
        capture_metadata=capture_metadata,
        expected_case=expected_case,
        expected_split=expected_split,
    )

    observations = np.asarray(capture["observations"], dtype=np.float32).copy()
    actions = np.asarray(
        capture["effective_corrective_normalized_actions"], dtype=np.float32
    ).copy()
    observations[0, PREVIOUS_ACTION_INDICES] = 0.0
    observations[1:, PREVIOUS_ACTION_INDICES] = actions[:-1]
    payload = {
        "observations": observations,
        "actions": actions,
        "requested_actions_audit": np.asarray(
            capture["requested_corrective_normalized_actions"], dtype=np.float32
        ),
        "effective_residual_commands": np.asarray(
            capture["effective_corrective_residual_commands"], dtype=np.float32
        ),
        "requested_residual_commands_audit": np.asarray(
            capture["requested_corrective_residual_commands"], dtype=np.float32
        ),
        "model_based_commands": np.asarray(
            capture["model_based_commands"], dtype=np.float32
        ),
        "final_high_level_commands": np.asarray(
            capture["final_high_level_commands"], dtype=np.float32
        ),
        "requested_vs_effective_residual_delta": np.asarray(
            capture["requested_vs_effective_residual_delta"], dtype=np.float32
        ),
        "command_clipped": np.asarray(capture["command_clipped"], dtype=bool),
        "case_ids": np.asarray(capture["case_ids"], dtype=np.int16),
        "elapsed_time_s": np.asarray(capture["elapsed_time_s"], dtype=np.float64),
        "execution_time_s": np.asarray(
            capture["execution_time_s"], dtype=np.float64
        ),
        "source_time_s": np.asarray(capture["source_time_s"], dtype=np.float64),
    }
    metadata = {
        "schema": MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA,
        "case": expected_case,
        "split": expected_split,
        "sample_count": len(actions),
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "command_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "training_target_contract": TRAINING_TARGET_CONTRACT,
        "previous_action_contract": PREVIOUS_ACTION_CONTRACT,
        "previous_action_rebuilt": True,
        "source_capture_schema": capture_metadata["schema"],
        "source_capture_sha256": sha256_file(capture_path),
        "source_final_status_sha256": sha256_file(final_status_path),
        "source_runtime_commit": capture_metadata["runtime_commit"],
        "source_plan_sha256": capture_metadata["plan_sha256"],
        "source_corrective_profile_sha256": capture_metadata[
            "corrective_profile_sha256"
        ],
        "source_paired_final_status_sha256": capture_metadata[
            "paired_final_status_sha256"
        ],
        "requested_actions_used_as_training_targets": False,
        "effective_actions_used_as_training_targets": True,
        "valid_for_case_merge": True,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    validate_case_dataset(
        metadata,
        payload,
        expected_case=expected_case,
        expected_split=expected_split,
    )
    return metadata, payload


def save_case_dataset(
    path: Path,
    metadata: Mapping[str, object],
    payload: Mapping[str, np.ndarray],
    *,
    expected_case: int = CORRECTIVE_CAPTURE_CASE,
    expected_split: str = CORRECTIVE_CAPTURE_SPLIT,
) -> None:
    validate_case_dataset(
        metadata,
        payload,
        expected_case=expected_case,
        expected_split=expected_split,
    )
    if path.exists():
        raise FileExistsError(f"refusing to overwrite converted dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(dict(metadata), sort_keys=True)),
        **{name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS},
    )


def load_case_dataset(
    path: Path,
    *,
    expected_case: int = CORRECTIVE_CAPTURE_CASE,
    expected_split: str = CORRECTIVE_CAPTURE_SPLIT,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {name: np.asarray(data[name]) for name in REQUIRED_ARRAYS}
    validate_case_dataset(
        metadata,
        payload,
        expected_case=expected_case,
        expected_split=expected_split,
    )
    return metadata, payload
