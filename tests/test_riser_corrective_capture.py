import json
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (
    CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
    load_corrective_capture,
    save_corrective_capture,
    validate_capture_admission,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)


PLAN_SHA = "a" * 64
PROFILE_SHA = "b" * 64
PAIR_SHA = "c" * 64
COMMIT = "d" * 40


def _admission() -> dict[str, object]:
    return {
        "schema": CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "case": 30,
        "split": "train",
        "corrective_target_admission_passed": True,
        "runtime_authorized": True,
        "label_capture_authorized": True,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "plan_sha256": PLAN_SHA,
        "corrective_profile_sha256": PROFILE_SHA,
        "paired_final_status_sha256": PAIR_SHA,
        "runtime_commit": COMMIT,
    }


def _payload(count: int = 4) -> dict[str, np.ndarray]:
    normalized = np.array(
        [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [0.2, -0.3, 0.4], [0.3, -0.4, 0.5]],
        dtype=np.float32,
    )[:count]
    residual = normalized * MODEL_BASED_POLICY_RESIDUAL_SCALES
    model = np.column_stack(
        (
            np.linspace(0.0, 0.1, count),
            np.linspace(0.0, -0.1, count),
            np.linspace(0.4, 0.5, count),
        )
    )
    return {
        "observations": np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32),
        "model_based_commands": model.astype(np.float32),
        "corrective_residual_commands": residual.astype(np.float32),
        "corrective_normalized_actions": normalized,
        "final_high_level_commands": (model + residual).astype(np.float32),
        "case_ids": np.full(count, 30, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "execution_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "source_time_s": np.arange(count, dtype=np.float64) * 0.002,
        "initialization_mask": np.zeros(count, dtype=bool),
        "amplitude_limited": np.zeros((count, 3), dtype=bool),
        "slew_limited": np.ones((count, 3), dtype=bool),
        "perturbation_active": np.array([False, True, True, False])[:count],
        "sample_plan_sha256": np.full(count, PLAN_SHA),
        "sample_runtime_commit": np.full(count, COMMIT),
    }


def _save(path: Path, payload: dict[str, np.ndarray] | None = None) -> None:
    save_corrective_capture(
        path,
        _payload() if payload is None else payload,
        source_duration_s=18.144412,
        execution_duration_s=29.22248819392579,
        plan_sha256=PLAN_SHA,
        runtime_commit=COMMIT,
        corrective_profile_sha256=PROFILE_SHA,
        paired_final_status_sha256=PAIR_SHA,
    )


def test_capture_round_trip_keeps_clocks_identities_and_training_closed(tmp_path) -> None:
    path = tmp_path / "case_0030_corrective_teacher_capture_v1.npz"
    _save(path)
    metadata, payload = load_corrective_capture(path)
    assert metadata["case"] == 30
    assert metadata["split"] == "train"
    assert metadata["valid_for_training"] is False
    assert metadata["training_started"] is False
    assert metadata["clock_contract"].startswith("elapsed_execution")
    assert np.all(payload["sample_plan_sha256"] == PLAN_SHA)
    assert np.all(payload["sample_runtime_commit"] == COMMIT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case", 3),
        ("split", "holdout"),
        ("runtime_authorized", False),
        ("label_capture_authorized", False),
        ("dataset_creation_authorized", True),
        ("bc_authorized", True),
        ("plan_sha256", "0" * 63),
    ],
)
def test_capture_admission_rejects_unreviewed_or_training_open_state(field, value) -> None:
    admission = _admission()
    admission[field] = value
    with pytest.raises(ValueError, match="admission"):
        validate_capture_admission(admission)


def test_capture_admission_accepts_only_the_bounded_case30_route() -> None:
    validate_capture_admission(_admission())


@pytest.mark.parametrize("mutation", ["initialization", "overflow", "clock", "identity", "clip"])
def test_capture_rejects_leakage_overflow_clock_identity_and_clipping(tmp_path, mutation) -> None:
    payload = _payload()
    if mutation == "initialization":
        payload["initialization_mask"][0] = True
    elif mutation == "overflow":
        payload["corrective_normalized_actions"][2, 0] = 0.95
        payload["corrective_residual_commands"] = (
            payload["corrective_normalized_actions"]
            * MODEL_BASED_POLICY_RESIDUAL_SCALES
        )
        payload["final_high_level_commands"] = (
            payload["model_based_commands"]
            + payload["corrective_residual_commands"]
        )
    elif mutation == "clock":
        payload["source_time_s"][2] = -1.0
    elif mutation == "identity":
        payload["sample_runtime_commit"][1] = "e" * 40
    else:
        payload["final_high_level_commands"][1, 0] += 0.01
    with pytest.raises(ValueError):
        _save(tmp_path / f"{mutation}.npz", payload)


def test_capture_rejects_missing_clock_and_noncausal_metadata(tmp_path) -> None:
    path = tmp_path / "valid.npz"
    _save(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
    metadata["observation_contract"] = "post_action_future_state"
    from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (
        validate_corrective_capture,
    )

    with pytest.raises(ValueError, match="metadata"):
        validate_corrective_capture(metadata, arrays)
    arrays.pop("source_time_s")
    with pytest.raises(ValueError, match="source_time_s"):
        validate_corrective_capture(metadata | {
            "observation_contract": "current_physical_cam_link_pre_action_with_known_reference_lookahead_v1"
        }, arrays)
