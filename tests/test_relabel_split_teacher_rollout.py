import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "imitation"))

from relabel_split_teacher_rollout import build_episode_rows, masked_rmse  # noqa: E402


def test_build_episode_rows_preserves_teacher_group_ids() -> None:
    lookup = build_episode_rows(
        np.asarray([4, 4, 9, 9, 9], dtype=np.int32),
        np.asarray([12, 27], dtype=np.int32),
    )

    assert lookup[12][0] == 4
    assert lookup[12][1].tolist() == [0, 1]
    assert lookup[27][0] == 9
    assert lookup[27][1].tolist() == [2, 3, 4]


def test_build_episode_rows_rejects_interleaved_sources() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        build_episode_rows(
            np.asarray([0, 1, 0], dtype=np.int32),
            np.asarray([1, 2], dtype=np.int32),
        )


def test_masked_rmse_ignores_unowned_gimbal_rows() -> None:
    prediction = np.asarray([[0.0, 5.0, 9.0]], dtype=np.float32)
    target = np.asarray([[1.0, 0.0, 3.0]], dtype=np.float32)
    mask = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)

    assert masked_rmse(prediction, target, mask) == pytest.approx(np.sqrt(18.5))
