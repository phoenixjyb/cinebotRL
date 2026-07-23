import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


pytest.importorskip("torch")

from scripts.two_wheel_balance.train_riser_residual_bc import (  # noqa: E402
    build_sequence_windows,
    case_balanced_mse,
    dataset_action_semantics,
    load_dataset,
    predict_recursive_previous_action_windows,
    previous_action_observation_gains,
    scheduled_sampling_probability,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


def _write_dataset(
    path, split_labels: np.ndarray, *, schema: str = "cinebotrl_two_wheel_riser_residual_merged_v2"
) -> None:
    row_count = len(split_labels)
    metadata = {
        "schema": schema,
        "case_count": 3,
        "row_count": row_count,
        "trajectory_leakage": False,
        "observation_names": list(OBSERVATION_NAMES),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "split_cases": {"train": [1], "validation": [2], "holdout": [3]},
    }
    observations = np.zeros((row_count, len(OBSERVATION_NAMES)), dtype=np.float32)
    actions = np.zeros((row_count, len(ACTION_NAMES)), dtype=np.float32)
    extra = {}
    if schema == "cinebotrl_two_wheel_riser_residual_merged_v3":
        metadata.update(
            {
                "dataset_admission_passed": True,
                "valid_for_bc_initialization": True,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
                "action_scales": [0.35, 0.4, 0.1],
                "action_clip_ratio": [0.0, 0.0, 0.0],
                "previous_action_contract": "previous_normalized_teacher_action_v1",
                "previous_action_rebuilt": True,
                "source_action_labels_used": False,
                "physical_gimbal_labels_used_as_actions": False,
            }
        )
        extra["action_valid_mask"] = np.ones_like(actions, dtype=np.float32)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata)),
        observations=observations,
        actions=actions,
        case_ids=np.repeat(np.arange(1, 4, dtype=np.int16), 2),
        split_labels=split_labels,
        source_index=np.repeat(np.arange(3, dtype=np.int16), 2),
        **extra,
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


def test_bc_loader_accepts_admitted_v3_previous_action_dataset(tmp_path) -> None:
    path = tmp_path / "accepted_v3.npz"
    _write_dataset(
        path,
        np.repeat(np.arange(3, dtype=np.int8), 2),
        schema="cinebotrl_two_wheel_riser_residual_merged_v3",
    )
    metadata, arrays = load_dataset(path)
    assert metadata["valid_for_bc_initialization"]
    assert arrays["action_valid_mask"].shape == arrays["actions"].shape


def test_dataset_action_semantics_distinguish_legacy_scales() -> None:
    v2 = dataset_action_semantics(
        {"schema": "cinebotrl_two_wheel_riser_residual_merged_v2"}
    )
    assert v2 == {
        "policy_command_base": "phase_feedforward",
        "policy_residual_contract": (
            "phase_feedforward_plus_bounded_policy_residual_v1"
        ),
        "residual_action_scales": [0.3, 0.4, 0.1],
    }
    v3 = dataset_action_semantics(
        {
            "schema": "cinebotrl_two_wheel_riser_residual_merged_v3",
            "action_scales": [0.35, 0.4, 0.1],
        }
    )
    assert v3["residual_action_scales"] == [0.35, 0.4, 0.1]
    with pytest.raises(ValueError, match="invalid residual action scales"):
        dataset_action_semantics(
            {
                "schema": "cinebotrl_two_wheel_riser_residual_merged_v3",
                "action_scales": [0.0, 0.4, 0.1],
            }
        )


def test_bc_loader_rejects_broken_v3_previous_action_recurrence(tmp_path) -> None:
    path = tmp_path / "broken_v3.npz"
    _write_dataset(
        path,
        np.repeat(np.arange(3, dtype=np.int8), 2),
        schema="cinebotrl_two_wheel_riser_residual_merged_v3",
    )
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    observations = payload["observations"].copy()
    observations[1, PREVIOUS_ACTION_INDICES[0]] = 0.5
    payload["observations"] = observations
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="previous-action recurrence"):
        load_dataset(path)


def test_case_balanced_metric_does_not_overweight_long_cases() -> None:
    target = np.zeros((4, 1), dtype=np.float32)
    prediction = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    cases = np.array([1, 2, 2, 2], dtype=np.int16)
    np.testing.assert_allclose(case_balanced_mse(target, prediction, cases), [0.5])
    np.testing.assert_allclose(np.mean(np.square(prediction - target), axis=0), [0.25])


def test_scheduled_sampling_probability_has_bounded_deterministic_ramp() -> None:
    values = [
        scheduled_sampling_probability(
            epoch, maximum=0.8, warmup_epochs=2, ramp_epochs=4
        )
        for epoch in range(1, 9)
    ]
    np.testing.assert_allclose(values, [0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8])
    with pytest.raises(ValueError, match="maximum"):
        scheduled_sampling_probability(1, maximum=1.1, warmup_epochs=0, ramp_epochs=1)


def test_previous_action_channel_gain_parser_is_fail_closed() -> None:
    args = type(
        "Args",
        (),
        {
            "previous_action_observation_gain": 1.0,
            "previous_action_observation_gains": "0.1,0.0,0.1",
        },
    )()
    assert previous_action_observation_gains(args) == (0.1, 0.0, 0.1)
    args.previous_action_observation_gains = "0.1,0.0"
    with pytest.raises(ValueError, match="three values"):
        previous_action_observation_gains(args)


def test_sequence_windows_never_cross_case_boundaries() -> None:
    cases = np.array([4, 4, 4, 4, 7, 7, 7], dtype=np.int16)
    windows = build_sequence_windows(cases, 3)
    np.testing.assert_array_equal(
        windows,
        np.array([[0, 1, 2], [3, -1, -1], [4, 5, 6]], dtype=np.int64),
    )
    for window in windows:
        active = window[window >= 0]
        assert len(set(cases[active].tolist())) == 1
    with pytest.raises(ValueError, match="not contiguous"):
        build_sequence_windows(np.array([1, 2, 1], dtype=np.int16), 2)


def test_recursive_prediction_is_bounded_by_windows_and_cases() -> None:
    torch = pytest.importorskip("torch")

    class PreviousActionIncrement(torch.nn.Module):
        def forward(self, observations):
            return observations[:, PREVIOUS_ACTION_INDICES] + 0.1

    observations = np.zeros((7, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, PREVIOUS_ACTION_INDICES] = np.array(
        [[0.0, 0.0, 0.0], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8],
         [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.7, 0.7, 0.7],
         [0.7, 0.7, 0.7]],
        dtype=np.float32,
    )
    cases = np.array([4, 4, 4, 4, 7, 7, 7], dtype=np.int16)
    prediction = predict_recursive_previous_action_windows(
        PreviousActionIncrement(),
        observations,
        cases,
        torch.device("cpu"),
        sequence_length=3,
        window_batch_size=2,
    )
    np.testing.assert_allclose(
        prediction[:, 0], [0.1, 0.2, 0.3, 0.6, 0.1, 0.2, 0.3], atol=1e-6
    )


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
            "--state-hidden-sizes",
            "8",
            "--lookahead-hidden-sizes",
            "4",
            "--fusion-hidden-sizes",
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
    assert not report["offline_policy_candidate_ready"]
    assert report["maximum_normalized_prediction_abs"] == 0.95
    assert report["prediction_margin_checks"]["validation"] == [True, True, True]
    assert report["recursive_prediction_margin_checks"] == {}
    assert report["separate_dynamic_authorization_required"]
    assert not report["dynamic_holdout_authorized"]
    assert not report["learned_rollout_authorized"]
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
    assert report["schema"] == "cinebotrl_two_wheel_riser_residual_bc_gate_v2"
    assert report["policy_architecture"] == "state_shared_lookahead_fusion_v1"
    assert report["policy_command_base"] == "phase_feedforward"
    assert report["policy_residual_contract"] == (
        "phase_feedforward_plus_bounded_policy_residual_v1"
    )
    assert report["residual_action_scales"] == [0.3, 0.4, 0.1]


def test_masked_previous_action_contract_is_recorded(tmp_path: Path) -> None:
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
            "--state-hidden-sizes",
            "8",
            "--lookahead-hidden-sizes",
            "4",
            "--fusion-hidden-sizes",
            "8",
            "--device",
            "cpu",
            "--mask-previous-action-observations",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["policy_architecture"] == (
        "state_shared_lookahead_fusion_previous_action_masked_v1"
    )
    assert report["masked_observation_indices"] == list(PREVIOUS_ACTION_INDICES)
    assert report["previous_action_observation_contract"] == (
        "masked_after_normalization_v1"
    )


def test_scheduled_previous_action_contract_is_recorded(tmp_path: Path) -> None:
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
            "--state-hidden-sizes",
            "8",
            "--lookahead-hidden-sizes",
            "4",
            "--fusion-hidden-sizes",
            "8",
            "--device",
            "cpu",
            "--scheduled-previous-action-max-probability",
            "1.0",
            "--scheduled-previous-action-ramp-epochs",
            "1",
            "--scheduled-sequence-length",
            "2",
            "--scheduled-sequence-batch-size",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["training_method"] == (
        "offline_behavior_cloning_deterministic_scheduled_sampling"
    )
    assert report["policy_architecture"] == "state_shared_lookahead_fusion_v1"
    assert report["masked_observation_indices"] == []
    assert report["previous_action_observation_contract"] == (
        "deterministic_scheduled_policy_previous_action_v1"
    )
    assert report["scheduled_previous_action_enabled"]
    assert report["sequence_windows_cross_case_boundaries"] is False
    assert report["scheduled_sampling_detaches_previous_prediction"]
    assert report["recursive_previous_action_split_results"]
    assert report["recursive_improvement_checks"]["validation"] == [False] * 3
    assert report["history"][0]["scheduled_previous_action_probability"] == 1.0
    assert report["history"][0]["scheduled_previous_action_rows"] == 1


def test_mask_and_scheduled_previous_action_modes_are_exclusive(tmp_path: Path) -> None:
    dataset = tmp_path / "zero_labels.npz"
    _write_dataset(dataset, np.repeat(np.arange(3, dtype=np.int8), 2))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/two_wheel_balance/train_riser_residual_bc.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "policy"),
            "--source-commit",
            "a" * 40,
            "--mask-previous-action-observations",
            "--scheduled-previous-action-max-probability",
            "0.5",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "exclusive" in result.stderr


def test_attenuated_previous_action_contract_is_recorded(tmp_path: Path) -> None:
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
            "--state-hidden-sizes",
            "8",
            "--lookahead-hidden-sizes",
            "4",
            "--fusion-hidden-sizes",
            "8",
            "--device",
            "cpu",
            "--previous-action-observation-gain",
            "0.1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["policy_architecture"] == (
        "state_shared_lookahead_fusion_previous_action_attenuated_v1"
    )
    assert report["previous_action_observation_gain"] == 0.1
    assert report["previous_action_observation_gains"] == [0.1, 0.1, 0.1]
    assert report["previous_action_observation_contract"] == (
        "attenuated_after_normalization_v1"
    )


def test_channel_selective_previous_action_contract_is_recorded(tmp_path: Path) -> None:
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
            "--state-hidden-sizes",
            "8",
            "--lookahead-hidden-sizes",
            "4",
            "--fusion-hidden-sizes",
            "8",
            "--device",
            "cpu",
            "--previous-action-observation-gains",
            "0.1,0.0,0.1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["policy_architecture"] == (
        "state_shared_lookahead_fusion_previous_action_attenuated_v1"
    )
    assert report["previous_action_observation_gain"] is None
    assert report["previous_action_observation_gains"] == [0.1, 0.0, 0.1]


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
                "--state-hidden-sizes",
                "32,32",
                "--lookahead-hidden-sizes",
                "16,16",
                "--fusion-hidden-sizes",
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
        assert report["offline_policy_candidate_ready"]
        assert report["maximum_normalized_prediction_abs"] == 0.95
        assert report["prediction_margin_checks"]["validation"] == [
            True,
            True,
            True,
        ]
        assert report["recursive_prediction_margin_checks"] == {}
        assert report["separate_dynamic_authorization_required"]
        assert not report["dynamic_holdout_authorized"]
        assert not report["learned_rollout_authorized"]
        assert report["source_commit"] == "a" * 40
        assert report["offline_gate_splits"] == ["validation"]
        assert not report["holdout_used_for_model_selection"]
        assert not report["holdout_metrics_computed"]
        assert "holdout" not in report["split_results"]
        assert report["case_balanced_training_loss"]
        assert report["case_balanced_validation_gate"]
        assert report["policy_architecture"] == (
            "state_shared_lookahead_fusion_v1"
        )
        assert (output / "residual_policy.pt").is_file()
        checkpoint = torch.load(
            output / "residual_policy.pt", map_location="cpu", weights_only=False
        )
        assert checkpoint["dataset_schema"] == (
            "cinebotrl_two_wheel_riser_residual_merged_v2"
        )
        assert checkpoint["policy_command_base"] == "phase_feedforward"
        assert checkpoint["policy_residual_contract"] == (
            "phase_feedforward_plus_bounded_policy_residual_v1"
        )
        assert checkpoint["residual_action_scales"] == [0.3, 0.4, 0.1]
        scripted = torch.jit.load(str(output / "residual_policy.torchscript.pt"))
        with torch.inference_mode():
            predictions.append(scripted(torch.from_numpy(probe)).numpy())
    np.testing.assert_array_equal(predictions[0], predictions[1])
