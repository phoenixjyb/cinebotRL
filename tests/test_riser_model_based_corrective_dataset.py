import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (
    save_corrective_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (
    MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA,
    convert_admitted_capture,
    load_case_dataset,
    save_case_dataset,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


PLAN_SHA = "a" * 64
PROFILE_SHA = "b" * 64
PAIR_SHA = "c" * 64
COMMIT = "d" * 40


def _fixture(
    tmp_path: Path, *, case: int = 30, split: str = "train"
) -> tuple[Path, Path]:
    count = 4
    requested = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3],
            [0.3, -0.2, 0.4],
            [0.4, -0.3, 0.5],
        ],
        dtype=np.float32,
    )
    effective = requested.copy()
    effective[2, 0] = 0.25
    requested_residual = requested * MODEL_BASED_POLICY_RESIDUAL_SCALES
    effective_residual = effective * MODEL_BASED_POLICY_RESIDUAL_SCALES
    delta = effective_residual - requested_residual
    model = np.tile(np.array([0.1, 0.0, 0.5]), (count, 1))
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, PREVIOUS_ACTION_INDICES] = 0.8
    capture_path = tmp_path / f"case_{case:04d}_corrective_teacher_capture_v2.npz"
    save_corrective_capture(
        capture_path,
        {
            "observations": observations,
            "model_based_commands": model,
            "requested_corrective_residual_commands": requested_residual,
            "requested_corrective_normalized_actions": requested,
            "effective_corrective_residual_commands": effective_residual,
            "effective_corrective_normalized_actions": effective,
            "requested_vs_effective_residual_delta": delta,
            "command_clipped": np.abs(delta) > 2e-7,
            "final_high_level_commands": model + effective_residual,
            "case_ids": np.full(count, case, dtype=np.int16),
            "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
            "execution_time_s": np.arange(count, dtype=np.float64) * 0.004,
            "source_time_s": np.arange(count, dtype=np.float64) * 0.002,
            "initialization_mask": np.zeros(count, dtype=bool),
            "amplitude_limited": np.zeros((count, 3), dtype=bool),
            "slew_limited": np.ones((count, 3), dtype=bool),
            "perturbation_active": np.array([False, True, True, False]),
            "sample_plan_sha256": np.full(count, PLAN_SHA),
            "sample_runtime_commit": np.full(count, COMMIT),
        },
        source_duration_s=18.0,
        execution_duration_s=29.0,
        plan_sha256=PLAN_SHA,
        runtime_commit=COMMIT,
        corrective_profile_sha256=PROFILE_SHA,
        paired_final_status_sha256=PAIR_SHA,
        case=case,
        split=split,
    )
    final_status = {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_capture_final_v2",
        "runtime_commit": COMMIT,
        "case": case,
        "split": split,
        "gate_checks": {"dynamic": True, "heartbeat": True},
        "archive_checks": {"loaded": True, "supervisor_contract": True},
        "capture": {
            "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest()
        },
        "dynamic_quality_passed": True,
        "capture_admitted_for_dataset_conversion": True,
        "normalized_training_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }
    final_status_path = tmp_path / "final_status.json"
    final_status_path.write_text(json.dumps(final_status), encoding="utf-8")
    return capture_path, final_status_path


def test_conversion_uses_effective_labels_and_rebuilds_previous_action(tmp_path) -> None:
    capture, final_status = _fixture(tmp_path)
    metadata, payload = convert_admitted_capture(capture, final_status)
    assert metadata["schema"] == MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA
    assert metadata["action_scales"] == [0.05, 0.05, 0.02]
    assert metadata["requested_actions_used_as_training_targets"] is False
    assert metadata["effective_actions_used_as_training_targets"] is True
    assert metadata["valid_for_training"] is False
    assert payload["actions"][2, 0] == pytest.approx(0.25)
    assert payload["requested_actions_audit"][2, 0] == pytest.approx(0.3)
    np.testing.assert_allclose(payload["observations"][0, PREVIOUS_ACTION_INDICES], 0.0)
    np.testing.assert_allclose(
        payload["observations"][1:, PREVIOUS_ACTION_INDICES],
        payload["actions"][:-1],
    )


def test_converted_case_dataset_round_trip_and_refuses_overwrite(tmp_path) -> None:
    capture, final_status = _fixture(tmp_path)
    metadata, payload = convert_admitted_capture(capture, final_status)
    output = tmp_path / "converted.npz"
    save_case_dataset(output, metadata, payload)
    restored_metadata, restored = load_case_dataset(output)
    assert restored_metadata == metadata
    np.testing.assert_array_equal(restored["actions"], payload["actions"])
    with pytest.raises(FileExistsError, match="overwrite"):
        save_case_dataset(output, metadata, payload)


def test_case23_conversion_requires_explicit_case_route(tmp_path) -> None:
    capture, final_status = _fixture(tmp_path, case=23)
    with pytest.raises(ValueError, match="unreviewed case"):
        convert_admitted_capture(capture, final_status)
    metadata, payload = convert_admitted_capture(
        capture, final_status, expected_case=23
    )
    assert metadata["case"] == 23
    assert np.unique(payload["case_ids"]).tolist() == [23]
    output = tmp_path / "case23_converted.npz"
    save_case_dataset(output, metadata, payload, expected_case=23)
    with pytest.raises(ValueError, match="unreviewed case"):
        load_case_dataset(output)
    restored_metadata, _ = load_case_dataset(output, expected_case=23)
    assert restored_metadata["case"] == 23


def test_case23_cli_requires_explicit_case_and_rejects_holdout(tmp_path) -> None:
    capture, final_status = _fixture(tmp_path, case=23)
    output = tmp_path / "case23_cli.npz"
    script = (
        Path(__file__).parents[1]
        / "scripts/two_wheel_balance/convert_model_based_corrective_capture.py"
    )
    base = [
        sys.executable,
        str(script),
        "--capture",
        str(capture),
        "--final-status",
        str(final_status),
        "--output",
        str(output),
    ]
    default = subprocess.run(base, check=False, capture_output=True, text=True)
    assert default.returncode != 0
    explicit = subprocess.run(
        base + ["--expected-case", "23"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(explicit.stdout)["case"] == 23
    holdout = subprocess.run(
        base + ["--expected-case", "23", "--expected-split", "holdout"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert holdout.returncode != 0


@pytest.mark.parametrize("mutation", ["admission", "capture_hash", "training_open"])
def test_conversion_rejects_unadmitted_or_mismatched_source(tmp_path, mutation) -> None:
    capture, final_status = _fixture(tmp_path)
    status = json.loads(final_status.read_text(encoding="utf-8"))
    if mutation == "admission":
        status["capture_admitted_for_dataset_conversion"] = False
    elif mutation == "capture_hash":
        status["capture"]["sha256"] = "0" * 64
    else:
        status["bc_authorized"] = True
    final_status.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="not admitted"):
        convert_admitted_capture(capture, final_status)


def test_conversion_rejects_effective_action_or_recurrence_tampering(tmp_path) -> None:
    capture, final_status = _fixture(tmp_path)
    metadata, payload = convert_admitted_capture(capture, final_status)
    payload["actions"][1, 0] += 0.1
    with pytest.raises(ValueError, match="effective training target"):
        save_case_dataset(tmp_path / "bad_action.npz", metadata, payload)
    metadata, payload = convert_admitted_capture(capture, final_status)
    payload["observations"][2, PREVIOUS_ACTION_INDICES[0]] += 0.1
    with pytest.raises(ValueError, match="previous-action recurrence"):
        save_case_dataset(tmp_path / "bad_recurrence.npz", metadata, payload)


def test_cli_preflight_does_not_write_and_execute_remains_training_closed(tmp_path) -> None:
    capture, final_status = _fixture(tmp_path)
    output = tmp_path / "converted.npz"
    script = (
        Path(__file__).parents[1]
        / "scripts/two_wheel_balance/convert_model_based_corrective_capture.py"
    )
    base = [
        sys.executable,
        str(script),
        "--capture",
        str(capture),
        "--final-status",
        str(final_status),
        "--output",
        str(output),
    ]
    preflight = subprocess.run(base, check=True, capture_output=True, text=True)
    preflight_result = json.loads(preflight.stdout)
    assert preflight_result["output_created"] is False
    assert not output.exists()
    executed = subprocess.run(
        base + ["--execute"], check=True, capture_output=True, text=True
    )
    result = json.loads(executed.stdout)
    assert result["output_created"] is True
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["valid_for_training"] is False
    load_case_dataset(output)
