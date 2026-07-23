import copy

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from rl_platform.tasks.two_wheel_balance.riser_model_based_bc_loss import (  # noqa: E402
    ModelBasedProjectedBCLoss,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_bc_adapter import (  # noqa: E402
    build_projection_aware_split,
    evaluate_projection_aware_model,
    train_projection_aware_epoch,
)


def _synthetic_split() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260723)
    rows_per_case = 24
    case_ids = np.repeat(np.arange(4, dtype=np.int64), rows_per_case)
    observations = rng.normal(size=(len(case_ids), 6)).astype(np.float32)
    previous_observations = np.zeros_like(observations)
    transition_valid = np.ones(len(case_ids), dtype=bool)
    transition_valid[::rows_per_case] = False
    previous_observations[transition_valid] = observations[
        np.flatnonzero(transition_valid) - 1
    ]
    teacher_weights = np.array(
        [
            [0.7, -0.2, 0.1, 0.0, 0.2, -0.1],
            [-0.1, 0.3, 0.0, 0.2, -0.2, 0.1],
            [0.0, -0.1, 0.4, -0.2, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    targets = 0.35 * np.tanh(observations @ teacher_weights.T)
    sample_weights = np.full(
        len(case_ids),
        1.0 / rows_per_case,
        dtype=np.float32,
    )
    model_based_commands = np.zeros((len(case_ids), 3), dtype=np.float32)
    model_based_commands[:, 2] = 0.6
    return {
        "observations": observations,
        "previous_observations": previous_observations,
        "effective_target_actions": targets.astype(np.float32),
        "model_based_commands": model_based_commands,
        "delta_time_s": np.ones(len(case_ids), dtype=np.float32),
        "transition_valid": transition_valid,
        "sample_weights": sample_weights,
        "case_ids": case_ids,
        "source_row_index": np.arange(len(case_ids), dtype=np.int64),
    }


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(6, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 3),
        torch.nn.Tanh(),
    )


def test_projection_aware_gradient_accumulation_is_batch_partition_invariant() -> None:
    torch.manual_seed(20260723)
    model_full = _model()
    model_chunked = copy.deepcopy(model_full)
    split = _synthetic_split()
    loss_full = ModelBasedProjectedBCLoss()
    loss_chunked = ModelBasedProjectedBCLoss()
    optimizer_full = torch.optim.SGD(model_full.parameters(), lr=0.01)
    optimizer_chunked = torch.optim.SGD(model_chunked.parameters(), lr=0.01)
    train_projection_aware_epoch(
        model_full,
        optimizer_full,
        split,
        loss_full,
        device=torch.device("cpu"),
        batch_size=len(split["observations"]),
        generator=torch.Generator().manual_seed(11),
    )
    train_projection_aware_epoch(
        model_chunked,
        optimizer_chunked,
        split,
        loss_chunked,
        device=torch.device("cpu"),
        batch_size=7,
        generator=torch.Generator().manual_seed(11),
    )
    for full, chunked in zip(
        model_full.parameters(), model_chunked.parameters(), strict=True
    ):
        torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-6)


def test_projection_aware_optimizer_learns_effective_targets_deterministically() -> None:
    split = _synthetic_split()
    torch.manual_seed(17)
    model = _model()
    loss = ModelBasedProjectedBCLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    before = evaluate_projection_aware_model(
        model,
        split,
        loss,
        device=torch.device("cpu"),
        batch_size=19,
    )
    generator = torch.Generator().manual_seed(23)
    history = []
    for _ in range(100):
        history.append(
            train_projection_aware_epoch(
                model,
                optimizer,
                split,
                loss,
                device=torch.device("cpu"),
                batch_size=13,
                generator=generator,
            )
        )
    after = evaluate_projection_aware_model(
        model,
        split,
        loss,
        device=torch.device("cpu"),
        batch_size=17,
    )
    before_mse = np.asarray(before["case_balanced_mse_per_action"])
    after_mse = np.asarray(after["case_balanced_mse_per_action"])
    assert np.all(after_mse < before_mse * 0.2)
    assert all(after["improves_over_zero_requested"])
    assert np.max(after_mse) < 2e-3
    assert after["loss_total"] < before["loss_total"] * 0.2
    assert history[-1]["loss_total"] < history[0]["loss_total"] * 0.2
    assert after["projection_clipped_rows"] == 0


def test_projection_split_rejects_cross_case_or_split_predecessor() -> None:
    observations = np.arange(24, dtype=np.float32).reshape(4, 6)
    payload = {
        "observations": observations,
        "actions": np.zeros((4, 3), dtype=np.float32),
        "model_based_commands": np.zeros((4, 3), dtype=np.float32),
        "case_ids": np.array([1, 1, 2, 2], dtype=np.int64),
        "split_labels": np.array([0, 0, 1, 1], dtype=np.int8),
        "previous_row_index": np.array([-1, 0, -1, 2], dtype=np.int64),
        "delta_time_s": np.ones(4, dtype=np.float32),
        "transition_valid": np.array([False, True, False, True]),
        "case_balanced_sample_weights": np.full(4, 0.5, dtype=np.float32),
    }
    split = build_projection_aware_split(payload, split_code=0)
    np.testing.assert_array_equal(split["source_row_index"], [0, 1])
    np.testing.assert_array_equal(
        split["previous_observations"][1], observations[0]
    )

    payload["previous_row_index"][1] = 2
    with pytest.raises(ValueError, match="crosses a case or split"):
        build_projection_aware_split(payload, split_code=0)
