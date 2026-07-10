"""Regression tests for trajectory-disjoint BC splits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reinforcement_learning.bc.pretrain_bc import build_source_grouped_split


def test_source_grouped_split_never_leaks_trajectory_rows():
    source_index = np.repeat(np.arange(20), np.arange(1, 21))
    labels, groups = build_source_grouped_split(
        source_index,
        val_fraction=0.2,
        holdout_fraction=0.2,
        seed=17,
    )
    assert set(groups) == {"train", "validation", "holdout"}
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert set(groups["train"]).isdisjoint(groups["holdout"])
    assert set(groups["validation"]).isdisjoint(groups["holdout"])
    for source in np.unique(source_index):
        assert np.unique(labels[source_index == source]).size == 1


def test_source_grouped_split_is_deterministic():
    source_index = np.repeat(np.arange(10), 3)
    first, _ = build_source_grouped_split(
        source_index,
        val_fraction=0.2,
        holdout_fraction=0.2,
        seed=42,
    )
    second, _ = build_source_grouped_split(
        source_index,
        val_fraction=0.2,
        holdout_fraction=0.2,
        seed=42,
    )
    np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    test_source_grouped_split_never_leaks_trajectory_rows()
    test_source_grouped_split_is_deterministic()
    print("BC grouped split assertions passed")
