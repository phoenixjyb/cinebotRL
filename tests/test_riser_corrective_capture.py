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


def _admission(case: int = 30, split: str = "train") -> dict[str, object]:
    return {
        "schema": CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "case": case,
        "split": split,
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
    requested_residual = normalized * MODEL_BASED_POLICY_RESIDUAL_SCALES
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
        "requested_corrective_residual_commands": requested_residual.astype(
            np.float32
        ),
        "requested_corrective_normalized_actions": normalized,
        "effective_corrective_residual_commands": requested_residual.astype(
            np.float32
        ),
        "effective_corrective_normalized_actions": normalized.copy(),
        "requested_vs_effective_residual_delta": np.zeros(
            (count, 3), dtype=np.float32
        ),
        "command_clipped": np.zeros((count, 3), dtype=bool),
        "final_high_level_commands": (model + requested_residual).astype(
            np.float32
        ),
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
    path = tmp_path / "case_0030_corrective_teacher_capture_v2.npz"
    _save(path)
    metadata, payload = load_corrective_capture(path)
    assert metadata["case"] == 30
    assert metadata["split"] == "train"
    assert metadata["valid_for_training"] is False
    assert metadata["training_started"] is False
    assert metadata["clock_contract"].startswith("elapsed_execution")
    assert metadata["training_target_contract"] == "effective_post_supervisor_residual_v1"
    assert np.all(payload["sample_plan_sha256"] == PLAN_SHA)
    assert np.all(payload["sample_runtime_commit"] == COMMIT)


def test_capture_round_trip_accepts_case23_only_with_explicit_route(tmp_path) -> None:
    payload = _payload()
    payload["case_ids"] = np.full(len(payload["case_ids"]), 23, dtype=np.int16)
    path = tmp_path / "case_0023_corrective_teacher_capture_v2.npz"
    save_corrective_capture(
        path,
        payload,
        source_duration_s=18.144412,
        execution_duration_s=29.22248819392579,
        plan_sha256=PLAN_SHA,
        runtime_commit=COMMIT,
        corrective_profile_sha256=PROFILE_SHA,
        paired_final_status_sha256=PAIR_SHA,
        case=23,
        split="train",
    )
    with pytest.raises(ValueError, match="unreviewed case"):
        load_corrective_capture(path)
    metadata, loaded = load_corrective_capture(
        path, expected_case=23, expected_split="train"
    )
    assert metadata["case"] == 23
    assert metadata["split"] == "train"
    assert np.all(loaded["case_ids"] == 23)


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


def test_capture_admission_accepts_case23_only_when_explicitly_expected() -> None:
    admission = _admission(case=23)
    with pytest.raises(ValueError, match="admission"):
        validate_capture_admission(admission)
    validate_capture_admission(admission, expected_case=23)


@pytest.mark.parametrize(
    ("expected_case", "expected_split"),
    [(0, "train"), (-1, "train"), (True, "train"), (23, "holdout")],
)
def test_capture_route_rejects_invalid_case_or_holdout(
    expected_case, expected_split
) -> None:
    with pytest.raises(ValueError, match="expected corrective"):
        validate_capture_admission(
            _admission(case=23),
            expected_case=expected_case,
            expected_split=expected_split,
        )


@pytest.mark.parametrize(
    "mutation",
    ["initialization", "overflow", "clock", "identity", "final_mismatch", "clip_mask"],
)
def test_capture_rejects_leakage_overflow_clock_identity_and_mismatch(tmp_path, mutation) -> None:
    payload = _payload()
    if mutation == "initialization":
        payload["initialization_mask"][0] = True
    elif mutation == "overflow":
        payload["requested_corrective_normalized_actions"][2, 0] = 0.95
        payload["requested_corrective_residual_commands"] = (
            payload["requested_corrective_normalized_actions"]
            * MODEL_BASED_POLICY_RESIDUAL_SCALES
        )
    elif mutation == "clock":
        payload["source_time_s"][2] = -1.0
    elif mutation == "identity":
        payload["sample_runtime_commit"][1] = "e" * 40
    elif mutation == "final_mismatch":
        payload["final_high_level_commands"][1, 0] += 0.01
    else:
        payload["command_clipped"][1, 0] = True
    with pytest.raises(ValueError):
        _save(tmp_path / f"{mutation}.npz", payload)


def test_capture_accepts_supervisor_clipping_with_effective_training_target(
    tmp_path,
) -> None:
    payload = _payload()
    row = 1
    payload["effective_corrective_residual_commands"][row, 0] = 0.002
    payload["effective_corrective_normalized_actions"][row, 0] = 0.04
    delta = (
        payload["effective_corrective_residual_commands"][row]
        - payload["requested_corrective_residual_commands"][row]
    )
    payload["requested_vs_effective_residual_delta"][row] = delta
    payload["command_clipped"][row] = np.abs(delta) > 2e-7
    payload["final_high_level_commands"][row] = (
        payload["model_based_commands"][row]
        + payload["effective_corrective_residual_commands"][row]
    )
    path = tmp_path / "recorded_clip.npz"
    _save(path, payload)
    metadata, loaded = load_corrective_capture(path)
    assert metadata["training_target_contract"] == "effective_post_supervisor_residual_v1"
    assert loaded["command_clipped"][row, 0]
    assert loaded["effective_corrective_normalized_actions"][row, 0] == pytest.approx(
        0.04
    )


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
