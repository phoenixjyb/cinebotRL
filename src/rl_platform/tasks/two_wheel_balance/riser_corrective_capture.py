"""Fail-closed archive contract for admitted model-based corrective labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .riser_residual_dataset import (
    ACTION_NAMES,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)


CORRECTIVE_CAPTURE_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_v2"
)
CORRECTIVE_CAPTURE_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_admission_v2"
)
CORRECTIVE_CAPTURE_CASE = 30
CORRECTIVE_CAPTURE_SPLIT = "train"
CORRECTIVE_CAPTURE_MAXIMUM_NORMALIZED_ACTION = 0.95

REQUIRED_ARRAYS = {
    "observations": 2,
    "model_based_commands": 2,
    "requested_corrective_residual_commands": 2,
    "requested_corrective_normalized_actions": 2,
    "effective_corrective_residual_commands": 2,
    "effective_corrective_normalized_actions": 2,
    "requested_vs_effective_residual_delta": 2,
    "command_clipped": 2,
    "final_high_level_commands": 2,
    "case_ids": 1,
    "elapsed_time_s": 1,
    "execution_time_s": 1,
    "source_time_s": 1,
    "initialization_mask": 1,
    "amplitude_limited": 2,
    "slew_limited": 2,
    "perturbation_active": 1,
    "sample_plan_sha256": 1,
    "sample_runtime_commit": 1,
}

REQUIRED_METADATA = {
    "schema",
    "case",
    "split",
    "sample_count",
    "source_duration_s",
    "execution_duration_s",
    "plan_sha256",
    "runtime_commit",
    "corrective_profile_sha256",
    "paired_final_status_sha256",
    "observation_names",
    "action_names",
    "residual_action_scales",
    "observation_contract",
    "sample_alignment_contract",
    "clock_contract",
    "initialization_contract",
    "teacher_applied_to_commands",
    "safety_supervisor_contract",
    "training_target_contract",
    "dynamic_quality_required_before_save",
    "normalized_training_dataset_created",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
    "valid_for_training",
}


def _exact_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_capture_admission(payload: Mapping[str, object]) -> None:
    checks = {
        "schema": payload.get("schema") == CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "case_split": payload.get("case") == CORRECTIVE_CAPTURE_CASE
        and payload.get("split") == CORRECTIVE_CAPTURE_SPLIT,
        "admitted_pair": payload.get("corrective_target_admission_passed") is True,
        "runtime": payload.get("runtime_authorized") is True,
        "capture": payload.get("label_capture_authorized") is True,
        "normalized_dataset_closed": payload.get("dataset_creation_authorized")
        is False,
        "training_closed": payload.get("bc_authorized") is False
        and payload.get("ppo_authorized") is False
        and payload.get("training_started") is False,
        "plan": _exact_digest(payload.get("plan_sha256"), 64),
        "profile": _exact_digest(payload.get("corrective_profile_sha256"), 64),
        "paired_evidence": _exact_digest(
            payload.get("paired_final_status_sha256"), 64
        ),
        "runtime_commit": _exact_digest(payload.get("runtime_commit"), 40),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid corrective capture admission: {checks}")


def load_capture_admission(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corrective capture admission must be an object")
    validate_capture_admission(payload)
    return payload


def validate_corrective_capture(
    metadata: Mapping[str, object],
    payload: Mapping[str, np.ndarray],
    *,
    expected_case: int = CORRECTIVE_CAPTURE_CASE,
) -> None:
    if set(metadata) != REQUIRED_METADATA:
        raise ValueError("corrective capture metadata fields do not match contract")
    for name, ndim in REQUIRED_ARRAYS.items():
        value = np.asarray(payload.get(name))
        if name not in payload or value.ndim != ndim:
            raise ValueError(f"missing or invalid corrective capture field {name}")

    count = len(np.asarray(payload["observations"]))
    if count < 2 or any(len(np.asarray(payload[name])) != count for name in REQUIRED_ARRAYS):
        raise ValueError("corrective capture row counts do not match")
    if metadata.get("sample_count") != count:
        raise ValueError("corrective capture metadata row count mismatch")

    widths = {
        "observations": len(OBSERVATION_NAMES),
        "model_based_commands": 3,
        "requested_corrective_residual_commands": 3,
        "requested_corrective_normalized_actions": len(ACTION_NAMES),
        "effective_corrective_residual_commands": 3,
        "effective_corrective_normalized_actions": len(ACTION_NAMES),
        "requested_vs_effective_residual_delta": 3,
        "command_clipped": 3,
        "final_high_level_commands": 3,
        "amplitude_limited": 3,
        "slew_limited": 3,
    }
    for name, width in widths.items():
        if np.asarray(payload[name]).shape[1] != width:
            raise ValueError(f"corrective capture {name} dimension mismatch")

    numeric = [
        name
        for name in REQUIRED_ARRAYS
        if name not in {"sample_plan_sha256", "sample_runtime_commit"}
    ]
    if not all(np.isfinite(np.asarray(payload[name], dtype=np.float64)).all() for name in numeric):
        raise ValueError("corrective capture contains non-finite values")

    cases = np.unique(np.asarray(payload["case_ids"]))
    if cases.tolist() != [expected_case] or metadata.get("case") != expected_case:
        raise ValueError("corrective capture mixes or opens an unreviewed case")
    if metadata.get("split") != CORRECTIVE_CAPTURE_SPLIT:
        raise ValueError("corrective capture is not in the training split")

    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    execution = np.asarray(payload["execution_time_s"], dtype=np.float64)
    source = np.asarray(payload["source_time_s"], dtype=np.float64)
    if abs(float(elapsed[0])) > 1e-9 or not np.all(np.diff(elapsed) > 0.0):
        raise ValueError("capture elapsed clock must start at zero and increase")
    if abs(float(execution[0])) > 1e-9 or not np.all(np.diff(execution) >= 0.0):
        raise ValueError("capture execution clock must start at zero and be monotonic")
    if abs(float(source[0])) > 1e-9 or not np.all(np.diff(source) >= 0.0):
        raise ValueError("capture source clock must start at zero and be monotonic")
    if execution[-1] > float(metadata["execution_duration_s"]) + 1e-9:
        raise ValueError("capture execution clock exceeds its sealed duration")
    if source[-1] > float(metadata["source_duration_s"]) + 1e-9:
        raise ValueError("capture source clock exceeds its sealed duration")

    if np.any(np.asarray(payload["initialization_mask"], dtype=bool)):
        raise ValueError("initialization samples leaked into corrective capture")
    requested_normalized = np.asarray(
        payload["requested_corrective_normalized_actions"], dtype=np.float64
    )
    effective_normalized = np.asarray(
        payload["effective_corrective_normalized_actions"], dtype=np.float64
    )
    if (
        max(
            float(np.max(np.abs(requested_normalized))),
            float(np.max(np.abs(effective_normalized))),
        )
        >= CORRECTIVE_CAPTURE_MAXIMUM_NORMALIZED_ACTION - 1e-6
    ):
        raise ValueError("corrective capture violates the reserved action margin")
    requested_residual = np.asarray(
        payload["requested_corrective_residual_commands"], dtype=np.float64
    )
    effective_residual = np.asarray(
        payload["effective_corrective_residual_commands"], dtype=np.float64
    )
    if not np.allclose(
        requested_residual,
        requested_normalized * MODEL_BASED_POLICY_RESIDUAL_SCALES,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("requested corrective residual does not reconstruct")
    if not np.allclose(
        effective_residual,
        effective_normalized * MODEL_BASED_POLICY_RESIDUAL_SCALES,
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("effective corrective residual does not reconstruct")
    model = np.asarray(payload["model_based_commands"], dtype=np.float64)
    final = np.asarray(payload["final_high_level_commands"], dtype=np.float64)
    if not np.allclose(final, model + effective_residual, rtol=0.0, atol=2e-7):
        raise ValueError("effective residual does not reconstruct final command")
    expected_delta = effective_residual - requested_residual
    delta = np.asarray(
        payload["requested_vs_effective_residual_delta"], dtype=np.float64
    )
    if not np.allclose(delta, expected_delta, rtol=0.0, atol=2e-7):
        raise ValueError("requested-versus-effective residual delta mismatch")
    clipped = np.asarray(payload["command_clipped"], dtype=bool)
    expected_clipped = np.abs(expected_delta) > 2e-7
    if not np.array_equal(clipped, expected_clipped):
        raise ValueError("command clipping mask does not match supervised commands")

    plan_sha = metadata.get("plan_sha256")
    runtime_commit = metadata.get("runtime_commit")
    if not _exact_digest(plan_sha, 64) or not _exact_digest(runtime_commit, 40):
        raise ValueError("corrective capture has invalid code or plan identity")
    if not np.all(np.asarray(payload["sample_plan_sha256"]).astype(str) == plan_sha):
        raise ValueError("per-sample plan identity mismatch")
    if not np.all(
        np.asarray(payload["sample_runtime_commit"]).astype(str) == runtime_commit
    ):
        raise ValueError("per-sample runtime identity mismatch")

    metadata_checks = {
        "schema": metadata.get("schema") == CORRECTIVE_CAPTURE_SCHEMA,
        "observation_names": metadata.get("observation_names")
        == list(OBSERVATION_NAMES),
        "action_names": metadata.get("action_names") == list(ACTION_NAMES),
        "scales": metadata.get("residual_action_scales")
        == MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "causal_observation": metadata.get("observation_contract")
        == "current_physical_cam_link_pre_action_with_known_reference_lookahead_v1",
        "alignment": metadata.get("sample_alignment_contract")
        == "pre_action_observation_requested_and_effective_command_v2",
        "clocks": metadata.get("clock_contract")
        == "elapsed_execution_and_authoritative_source_time_separate_v1",
        "initialization": metadata.get("initialization_contract")
        == "separate_and_excluded_from_capture_v1",
        "teacher_applied": metadata.get("teacher_applied_to_commands") is True,
        "supervisor": metadata.get("safety_supervisor_contract")
        == "requested_teacher_intent_and_effective_applied_command_separate_v1",
        "training_target": metadata.get("training_target_contract")
        == "effective_post_supervisor_residual_v1",
        "dynamic_gate": metadata.get("dynamic_quality_required_before_save") is True,
        "training_closed": metadata.get("normalized_training_dataset_created")
        is False
        and metadata.get("bc_authorized") is False
        and metadata.get("ppo_authorized") is False
        and metadata.get("training_started") is False
        and metadata.get("valid_for_training") is False,
        "profile": _exact_digest(metadata.get("corrective_profile_sha256"), 64),
        "paired_evidence": _exact_digest(
            metadata.get("paired_final_status_sha256"), 64
        ),
    }
    if not all(metadata_checks.values()):
        raise ValueError(f"invalid corrective capture metadata: {metadata_checks}")


def save_corrective_capture(
    path: Path,
    payload: Mapping[str, np.ndarray],
    *,
    source_duration_s: float,
    execution_duration_s: float,
    plan_sha256: str,
    runtime_commit: str,
    corrective_profile_sha256: str,
    paired_final_status_sha256: str,
) -> None:
    arrays = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
    metadata = {
        "schema": CORRECTIVE_CAPTURE_SCHEMA,
        "case": CORRECTIVE_CAPTURE_CASE,
        "split": CORRECTIVE_CAPTURE_SPLIT,
        "sample_count": len(arrays["observations"]),
        "source_duration_s": float(source_duration_s),
        "execution_duration_s": float(execution_duration_s),
        "plan_sha256": plan_sha256,
        "runtime_commit": runtime_commit,
        "corrective_profile_sha256": corrective_profile_sha256,
        "paired_final_status_sha256": paired_final_status_sha256,
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "residual_action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "observation_contract": (
            "current_physical_cam_link_pre_action_with_known_reference_lookahead_v1"
        ),
        "sample_alignment_contract": (
            "pre_action_observation_requested_and_effective_command_v2"
        ),
        "clock_contract": (
            "elapsed_execution_and_authoritative_source_time_separate_v1"
        ),
        "initialization_contract": "separate_and_excluded_from_capture_v1",
        "teacher_applied_to_commands": True,
        "safety_supervisor_contract": (
            "requested_teacher_intent_and_effective_applied_command_separate_v1"
        ),
        "training_target_contract": "effective_post_supervisor_residual_v1",
        "dynamic_quality_required_before_save": True,
        "normalized_training_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    validate_corrective_capture(metadata, arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


def load_corrective_capture(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {name: np.asarray(data[name]) for name in REQUIRED_ARRAYS}
    validate_corrective_capture(metadata, payload)
    return metadata, payload
