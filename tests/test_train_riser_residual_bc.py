import json

import numpy as np
import pytest


pytest.importorskip("torch")

from scripts.two_wheel_balance.train_riser_residual_bc import load_dataset  # noqa: E402
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    OBSERVATION_NAMES,
)


def _write_dataset(path, split_labels: np.ndarray) -> None:
    row_count = len(split_labels)
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_residual_merged_v1",
        "case_count": 3,
        "row_count": row_count,
    }
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata)),
        observations=np.zeros(
            (row_count, len(OBSERVATION_NAMES)), dtype=np.float32
        ),
        actions=np.zeros((row_count, len(ACTION_NAMES)), dtype=np.float32),
        split_labels=split_labels,
        source_index=np.repeat(np.arange(3, dtype=np.int16), 2),
    )


def test_bc_loader_accepts_case_disjoint_splits(tmp_path) -> None:
    path = tmp_path / "accepted.npz"
    _write_dataset(path, np.repeat(np.arange(3, dtype=np.int8), 2))
    metadata, arrays = load_dataset(path)
    assert metadata["case_count"] == 3
    assert arrays["observations"].shape == (6, len(OBSERVATION_NAMES))


def test_bc_loader_rejects_source_leakage(tmp_path) -> None:
    path = tmp_path / "leaking.npz"
    _write_dataset(path, np.array([0, 1, 1, 1, 2, 2], dtype=np.int8))
    with pytest.raises(ValueError, match="source leakage"):
        load_dataset(path)
