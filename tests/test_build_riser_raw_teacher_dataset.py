import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
    save_raw_teacher_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/build_riser_raw_teacher_dataset.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, bad_hash: bool = False) -> list[str]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = []
    for case in range(1, 41):
        count = 2
        observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
        observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
        raw = np.array([[0.1, 0.1, 0.01], [0.2, -0.1, 0.02]], dtype=np.float32)
        path = raw_dir / f"case_{case:04d}_executed_raw_teacher_v1.npz"
        save_raw_teacher_case(
            path,
            case,
            {
                "observations": observations,
                "raw_residual_commands": raw,
                "case_ids": np.full(count, case, dtype=np.int16),
                "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
                "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
                "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
                "teacher_commands": np.column_stack(
                    (0.1 + raw[:, 0], raw[:, 1], raw[:, 2])
                ),
            },
        )
        rows.append(
            {
                "case": case,
                "raw_case": str(path),
                "raw_case_sha256": "0" * 64 if bad_hash and case == 7 else _sha(path),
            }
        )
    audit = tmp_path / "corpus_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_raw_teacher_corpus_audit_v1",
                "capture_admission_passed": True,
                "action_scale_frozen": True,
                "valid_for_bc_initialization": True,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
                "frozen_action_scales": [0.4, 0.4, 0.1],
                "case_count": 40,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return [
        sys.executable,
        str(SCRIPT),
        "--corpus-audit",
        str(audit),
        "--output",
        str(tmp_path / "dataset.npz"),
    ]


def test_builds_exact_30_5_5_dataset_and_previous_actions(tmp_path: Path) -> None:
    result = subprocess.run(
        _fixture(tmp_path), cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    with np.load(tmp_path / "dataset.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        observations = np.asarray(data["observations"])
        actions = np.asarray(data["actions"])
        case_ids = np.asarray(data["case_ids"])
    assert metadata["schema"] == "cinebotrl_two_wheel_riser_residual_merged_v3"
    assert metadata["split_case_counts"] == {
        "train": 30,
        "validation": 5,
        "holdout": 5,
    }
    assert metadata["valid_for_bc_initialization"]
    assert not metadata["bc_authorized"]
    assert not metadata["ppo_authorized"]
    for case in np.unique(case_ids):
        mask = case_ids == case
        previous = observations[mask][:, PREVIOUS_ACTION_INDICES]
        np.testing.assert_array_equal(previous[0], np.zeros(3))
        np.testing.assert_allclose(previous[1:], actions[mask][:-1], atol=1e-7)


def test_rejects_raw_source_hash_mismatch(tmp_path: Path) -> None:
    result = subprocess.run(
        _fixture(tmp_path, bad_hash=True),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "raw source identity mismatch for case 7" in result.stderr
