import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


pytest.importorskip("torch")

from scripts.two_wheel_balance.train_riser_residual_bc import (  # noqa: E402
    case_balanced_mse,
    load_dataset,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
)


def _write_dataset(path, split_labels: np.ndarray) -> None:
    row_count = len(split_labels)
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_residual_merged_v2",
        "case_count": 3,
        "row_count": row_count,
        "trajectory_leakage": False,
        "observation_names": list(OBSERVATION_NAMES),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "split_cases": {"train": [1], "validation": [2], "holdout": [3]},
    }
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata)),
        observations=np.zeros(
            (row_count, len(OBSERVATION_NAMES)), dtype=np.float32
        ),
        actions=np.zeros((row_count, len(ACTION_NAMES)), dtype=np.float32),
        case_ids=np.repeat(np.arange(1, 4, dtype=np.int16), 2),
        split_labels=split_labels,
        source_index=np.repeat(np.arange(3, dtype=np.int16), 2),
    )


def _write_learnable_dataset(path: Path) -> np.ndarray:
    rng = np.random.default_rng(20260716)
    rows_per_case = 128
    observations = rng.normal(
        size=(3 * rows_per_case, len(OBSERVATION_NAMES))
    ).astype(np.float32)
    actions = np.column_stack(
        (
            0.4 * np.tanh(observations[:, 0] + 0.3 * observations[:, 1]),
            0.3 * np.tanh(observations[:, 2] - 0.2 * observations[:, 3]),
            0.2 * np.tanh(observations[:, 4] + 0.5 * observations[:, 5]),
        )
    ).astype(np.float32)
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_residual_merged_v2",
        "case_count": 3,
        "row_count": len(observations),
        "trajectory_leakage": False,
        "observation_names": list(OBSERVATION_NAMES),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "split_cases": {"train": [1], "validation": [2], "holdout": [3]},
    }
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata)),
        observations=observations,
        actions=actions,
        case_ids=np.repeat(np.arange(1, 4, dtype=np.int16), rows_per_case),
        split_labels=np.repeat(np.arange(3, dtype=np.int8), rows_per_case),
        source_index=np.repeat(np.arange(3, dtype=np.int16), rows_per_case),
    )
    return observations[:8]


def test_bc_loader_accepts_case_disjoint_splits(tmp_path) -> None:
    path = tmp_path / "accepted.npz"
    _write_dataset(path, np.repeat(np.arange(3, dtype=np.int8), 2))
    metadata, arrays = load_dataset(path)
    assert metadata["case_count"] == 3
    assert arrays["observations"].shape == (6, len(OBSERVATION_NAMES))


def test_case_balanced_metric_does_not_overweight_long_cases() -> None:
    target = np.zeros((4, 1), dtype=np.float32)
    prediction = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    cases = np.array([1, 2, 2, 2], dtype=np.int16)
    np.testing.assert_allclose(case_balanced_mse(target, prediction, cases), [0.5])
    np.testing.assert_allclose(np.mean(np.square(prediction - target), axis=0), [0.25])


def test_bc_loader_rejects_source_leakage(tmp_path) -> None:
    path = tmp_path / "leaking.npz"
    _write_dataset(path, np.array([0, 1, 1, 1, 2, 2], dtype=np.int8))
    with pytest.raises(ValueError, match="source leakage"):
        load_dataset(path)


def test_bc_loader_rejects_old_observation_schema(tmp_path) -> None:
    path = tmp_path / "old_v1.npz"
    _write_dataset(path, np.repeat(np.arange(3, dtype=np.int8), 2))
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["schema"] = "cinebotrl_two_wheel_riser_residual_merged_v1"
    payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="wrong merged dataset schema"):
        load_dataset(path)


def test_bc_loader_rejects_declared_split_mismatch(tmp_path) -> None:
    path = tmp_path / "bad_split.npz"
    _write_dataset(path, np.repeat(np.arange(3, dtype=np.int8), 2))
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["split_cases"]["holdout"] = [2]
    payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="holdout cases mismatch"):
        load_dataset(path)


def test_bc_loader_rejects_source_that_contains_multiple_cases(tmp_path) -> None:
    path = tmp_path / "bad_source.npz"
    _write_dataset(path, np.repeat(np.arange(3, dtype=np.int8), 2))
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["case_ids"] = np.array([1, 4, 2, 2, 3, 3], dtype=np.int16)
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["case_count"] = 4
    metadata["split_cases"]["train"] = [1, 4]
    payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="one-to-one"):
        load_dataset(path)


def test_failed_offline_gate_does_not_emit_policy_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "zero_labels.npz"
    _write_dataset(dataset, np.repeat(np.arange(3, dtype=np.int8), 2))
    output = tmp_path / "policy"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/two_wheel_balance/train_riser_residual_bc.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--source-commit",
            "a" * 40,
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-sizes",
            "8",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert not report["offline_gate_passed"]
    assert report["checkpoint"] is None
    assert report["torchscript"] is None
    assert not (output / "residual_policy.pt").exists()
    assert not (output / "residual_policy.torchscript.pt").exists()
    assert report["deterministic_algorithms_enabled"]
    assert report["source_commit"] == "a" * 40
    assert report["offline_gate_splits"] == ["validation"]
    assert not report["holdout_used_for_model_selection"]
    assert not report["holdout_metrics_computed"]
    assert "holdout" not in report["split_results"]
    assert report["case_balanced_training_loss"]
    assert report["case_balanced_validation_gate"]
    assert report["seed"] == 20260716


def test_admitted_bc_is_reproducible_for_the_same_seed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dataset = tmp_path / "learnable.npz"
    probe = _write_learnable_dataset(dataset)
    predictions = []
    for run in (1, 2):
        output = tmp_path / f"policy_{run}"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/two_wheel_balance/train_riser_residual_bc.py",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
                "--source-commit",
                "a" * 40,
                "--epochs",
                "80",
                "--patience",
                "15",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.003",
                "--hidden-sizes",
                "32,32",
                "--device",
                "cpu",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["offline_gate_passed"]
        assert report["source_commit"] == "a" * 40
        assert report["offline_gate_splits"] == ["validation"]
        assert not report["holdout_used_for_model_selection"]
        assert not report["holdout_metrics_computed"]
        assert "holdout" not in report["split_results"]
        assert report["case_balanced_training_loss"]
        assert report["case_balanced_validation_gate"]
        assert (output / "residual_policy.pt").is_file()
        scripted = torch.jit.load(str(output / "residual_policy.torchscript.pt"))
        with torch.inference_mode():
            predictions.append(scripted(torch.from_numpy(probe)).numpy())
    np.testing.assert_array_equal(predictions[0], predictions[1])
