"""Regression tests for explicit NPZ sample weights in BC."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.reinforcement_learning.bc.pretrain_bc import build_sample_weights


def test_dataset_sample_weights_are_loaded_without_renormalizing():
    args = SimpleNamespace(sample_weight_mode="dataset")
    expected = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)
    actual = build_sample_weights(3, {}, args, dataset_sample_weights=expected)
    np.testing.assert_array_equal(actual, expected)


def test_dataset_sample_weights_require_positive_finite_values():
    args = SimpleNamespace(sample_weight_mode="dataset")
    with pytest.raises(ValueError, match="finite positive"):
        build_sample_weights(2, {}, args, dataset_sample_weights=np.asarray([1.0, 0.0]))

