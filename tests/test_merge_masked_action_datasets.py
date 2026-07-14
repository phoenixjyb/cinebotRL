"""Regression tests for masked-dataset source-group namespacing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "reinforcement_learning" / "sb3" / "merge_masked_action_datasets.py"
SPEC = importlib.util.spec_from_file_location("merge_masked_action_datasets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_dataset(path: Path, source_index: np.ndarray) -> None:
    count = source_index.shape[0]
    np.savez_compressed(
        path,
        observations=np.zeros((count, 3), dtype=np.float32),
        actions=np.zeros((count, 2), dtype=np.float32),
        action_valid_mask=np.ones((count, 2), dtype=np.float32),
        source_index=source_index,
    )


def test_merge_namespaces_overlapping_local_source_ids(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "merged.npz"
    write_dataset(first, np.asarray([4, 4], dtype=np.int32))
    write_dataset(second, np.asarray([4, 8], dtype=np.int32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--inputs",
            str(first),
            str(second),
            "--output",
            str(output),
            "--no-shuffle",
        ],
    )
    assert MODULE.main() == 0
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["source_index"], np.asarray([0, 0, 1, 2], dtype=np.int32))

