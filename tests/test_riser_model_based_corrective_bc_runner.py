import argparse
import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from scripts.two_wheel_balance.train_riser_residual_bc import (  # noqa: E402
    build_projection_aware_residual_policy,
    run_projection_aware_bc,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_bc_contract import (  # noqa: E402
    DEFAULT_BC_TRAINING_CONFIG,
    sha256_file,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_training_dataset import (  # noqa: E402
    MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE,
)


EXECUTION_COMMIT = "a" * 40
TRAIN_CASES = [1, 2, 6, 7]
VALIDATION_CASES = [8, 16]
HOLDOUT_CASES = [3, 5, 13, 19, 24]


def _synthetic_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260723)
    rows_per_case = 24
    cases = np.asarray(TRAIN_CASES + VALIDATION_CASES, dtype=np.int64)
    case_ids = np.repeat(cases, rows_per_case)
    observations = rng.normal(
        size=(len(case_ids), len(OBSERVATION_NAMES))
    ).astype(np.float32)
    actions = np.column_stack(
        (
            0.20 * np.tanh(observations[:, 0]),
            0.15 * np.tanh(observations[:, 1]),
            0.10 * np.tanh(observations[:, 2]),
        )
    ).astype(np.float32)
    split_labels = np.repeat(
        np.asarray([0] * len(TRAIN_CASES) + [1] * len(VALIDATION_CASES)),
        rows_per_case,
    ).astype(np.int8)
    previous = np.arange(len(case_ids), dtype=np.int64) - 1
    transition_valid = np.ones(len(case_ids), dtype=bool)
    transition_valid[::rows_per_case] = False
    previous[~transition_valid] = -1
    model_commands = np.zeros((len(case_ids), len(ACTION_NAMES)), dtype=np.float32)
    model_commands[:, 2] = 0.6
    return {
        "observations": observations,
        "actions": actions,
        "model_based_commands": model_commands,
        "case_ids": case_ids,
        "split_labels": split_labels,
        "previous_row_index": previous,
        "delta_time_s": np.ones(len(case_ids), dtype=np.float32),
        "transition_valid": transition_valid,
        "case_balanced_sample_weights": np.full(
            len(case_ids),
            1.0 / rows_per_case,
            dtype=np.float32,
        ),
    }


def _args(tmp_path: Path, dataset: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "policy",
        dataset=dataset,
        epochs=DEFAULT_BC_TRAINING_CONFIG["epochs_max"],
        patience=DEFAULT_BC_TRAINING_CONFIG["patience"],
        batch_size=DEFAULT_BC_TRAINING_CONFIG["batch_size"],
        learning_rate=DEFAULT_BC_TRAINING_CONFIG["learning_rate"],
        weight_decay=DEFAULT_BC_TRAINING_CONFIG["weight_decay"],
        state_hidden_sizes="128,128",
        lookahead_hidden_sizes="64,64",
        fusion_hidden_sizes="256,128",
        seed=DEFAULT_BC_TRAINING_CONFIG["seed"],
        device=DEFAULT_BC_TRAINING_CONFIG["device"],
        minimum_improvement_fraction=DEFAULT_BC_TRAINING_CONFIG[
            "minimum_improvement_fraction"
        ],
        maximum_normalized_prediction_abs=DEFAULT_BC_TRAINING_CONFIG[
            "maximum_normalized_prediction_abs"
        ],
        mask_previous_action_observations=False,
        scheduled_previous_action_max_probability=0.0,
        previous_action_observation_gain=1.0,
        previous_action_observation_gains=None,
    )


def test_projection_aware_policy_starts_as_exact_zero_residual() -> None:
    mean = np.linspace(-1.0, 1.0, len(OBSERVATION_NAMES), dtype=np.float32)
    std = np.linspace(0.5, 1.5, len(OBSERVATION_NAMES), dtype=np.float32)
    model = build_projection_aware_residual_policy(mean, std).eval()
    observations = torch.from_numpy(
        np.stack((mean, mean + std, mean - 2.0 * std))
    )
    with torch.no_grad():
        eager = model(observations)
        scripted = torch.jit.script(model)(observations)
    assert torch.count_nonzero(model.action_head.weight).item() == 0
    assert torch.count_nonzero(model.action_head.bias).item() == 0
    assert torch.count_nonzero(eager).item() == 0
    assert torch.equal(scripted, eager)


def test_guarded_projection_runner_learns_synthetic_data_and_seals_artifacts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "training.npz"
    dataset.write_bytes(b"synthetic-projection-training")
    metadata = {
        "schema": MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
        "action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
    }
    admission = {
        "dataset": {"path": dataset.as_posix(), "sha256": sha256_file(dataset)},
        "execution_commit": EXECUTION_COMMIT,
        "optimizer_contract": (
            "exact_case_balanced_projection_aware_gradient_accumulation_v1"
        ),
        "validation_contract": (
            "projected_effective_action_case_balanced_validation_v1"
        ),
        "loss_contract": "model_based_projected_effective_action_bc_loss_v1",
        "training_config": DEFAULT_BC_TRAINING_CONFIG,
        "split_cases": {
            "train": TRAIN_CASES,
            "validation": VALIDATION_CASES,
        },
        "reserved_holdout_cases": HOLDOUT_CASES,
        "bc_authorized": True,
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    result = run_projection_aware_bc(
        _args(tmp_path, dataset),
        metadata,
        _synthetic_arrays(),
        {
            "policy_command_base": "model_based_planner",
            "policy_residual_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
            "residual_action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        },
        admission=admission,
        admission_path=admission_path,
        execution_commit=EXECUTION_COMMIT,
        device=torch.device("cpu"),
    )
    assert result == 0
    report = json.loads(
        (tmp_path / "policy/report.json").read_text(encoding="utf-8")
    )
    assert report["offline_gate_passed"] is True
    assert report["passed"] is True
    assert report["valid_for_dynamic_canary"] is True
    assert report["holdout_metrics_computed"] is False
    assert report["learned_rollout_authorized"] is False
    assert report["ppo_authorized"] is False
    assert report["split_metrics"]["validation"][
        "requested_slew_violation_count"
    ] == [0, 0, 0]
    assert (tmp_path / "policy/residual_policy.pt").is_file()
    assert (tmp_path / "policy/residual_policy.torchscript.pt").is_file()
    checkpoint = torch.load(
        tmp_path / "policy/residual_policy.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["policy_architecture"] == (
        MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
    )
    assert checkpoint["observation_dimension"] == len(OBSERVATION_NAMES) == 65
    assert checkpoint["base_observation_dimension"] == 26
    assert checkpoint["lookahead_horizon_count"] == 3
    assert checkpoint["lookahead_channel_count"] == 13
    assert checkpoint["action_dimension"] == len(ACTION_NAMES) == 3
    assert checkpoint["zero_initialized_before_optimization"] is True
