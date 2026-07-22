import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)
from scripts.two_wheel_balance.build_riser_initial_dataset_v2 import build_dataset


def test_merges_case78_and_applies_only_admitted_split_change() -> None:
    cases = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28, 30, 31, 32, 33, 34, 36, 37, 41, 52, 53, 66, 67, 68, 70, 74]
    count = len(cases) * 2
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
    actions = np.tile(np.asarray([[0.1, 0.0, 0.0], [0.2, 0.1, 0.1]], dtype=np.float32), (len(cases), 1))
    for index in range(len(cases)):
        observations[index * 2 + 1, PREVIOUS_ACTION_INDICES] = actions[index * 2]
    case_ids = np.repeat(np.asarray(cases, dtype=np.int16), 2)
    teacher = np.column_stack((0.1 + 0.35 * actions[:, 0], 0.4 * actions[:, 1], 0.1 * actions[:, 2]))
    base_arrays = {
        "observations": observations,
        "actions": actions,
        "case_ids": case_ids,
        "elapsed_time_s": np.tile(np.asarray([0.0, 0.005]), len(cases)),
        "phase_time_s": np.tile(np.asarray([0.0, 0.004]), len(cases)),
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": teacher,
        "source_index": np.repeat(np.arange(len(cases), dtype=np.int16), 2),
        "split_labels": np.zeros(count, dtype=np.int8),
        "action_valid_mask": np.ones((count, 3), dtype=np.float32),
    }
    previous_split = {
        "train": [2, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 21, 23, 25, 26, 28, 30, 31, 33, 34, 36, 37, 41, 52, 53, 66, 67, 68, 70, 74],
        "validation": [4, 8, 16, 22, 32],
        "holdout": [3, 5, 13, 19, 24],
    }
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_residual_merged_v3",
        "case_count": 40,
        "captured_case_count": 41,
        "row_count": count,
        "action_scales": [0.35, 0.4, 0.1],
        "split_cases": previous_split,
        "source_rows": [{"case": case, "split": next(name for name, members in previous_split.items() if case in members)} for case in cases],
        "corpus_audit": "base_corpus.json",
        "corpus_audit_sha256": "base-corpus-sha",
        "seed": 20260720,
        "dataset_admission_passed": True,
        "valid_for_bc_initialization": True,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    admitted = {
        "train": sorted(previous_split["train"] + [4]),
        "validation": [8, 16, 22, 32, 78],
        "holdout": previous_split["holdout"],
    }
    raw_obs = np.zeros((3, len(OBSERVATION_NAMES)), dtype=np.float32)
    raw_obs[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
    raw = np.asarray([[0.035, 0.0, 0.0], [0.07, 0.04, 0.01], [0.0, -0.04, 0.0]], dtype=np.float32)
    raw_payload = {
        "observations": raw_obs,
        "raw_residual_commands": raw,
        "case_ids": np.full(3, 78, dtype=np.int16),
        "elapsed_time_s": np.asarray([0.0, 0.005, 0.01]),
        "phase_time_s": np.asarray([0.0, 0.004, 0.008]),
        "baseline_wheel_actions": np.zeros((3, 2), dtype=np.float32),
        "teacher_commands": np.column_stack((0.1 + raw[:, 0], raw[:, 1], raw[:, 2])),
    }
    result, arrays = build_dataset(
        base_metadata=metadata,
        base_arrays=base_arrays,
        base_summary={"dataset_sha256": "53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44", "split_cases": previous_split},
        split_admission={"split_admitted": True, "admitted_split_cases": admitted},
        raw_payload=raw_payload,
        raw_summary={"raw_teacher_conversion_passed": True, "offline_dataset_rebuild_authorized": True, "case": 78, "raw_teacher": "case78.npz", "raw_teacher_sha256": "abc", "bc_authorized": False, "ppo_authorized": False},
        label_admission={"label_admission_passed": True, "offline_dataset_rebuild_authorized": True, "case": 78, "bc_authorized": False, "ppo_authorized": False},
        expected_base_rows=count,
        expected_total_rows=count + 3,
    )
    assert result["case_count"] == 41
    assert result["row_count"] == count + 3
    assert result["split_cases"] == admitted
    assert result["holdout_policy_metrics_computed"] is False
    assert result["base_corpus_audit"] == "base_corpus.json"
    assert result["split_assignment_randomized"] is False
    assert "corpus_audit" not in result
    assert "seed" not in result
    assert np.all(arrays["split_labels"][arrays["case_ids"] == 4] == 0)
    assert np.all(arrays["split_labels"][arrays["case_ids"] == 78] == 1)
