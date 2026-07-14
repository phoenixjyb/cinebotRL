"""Regression tests for packaging accepted corrective GIK labels."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "imitation" / "package_corrective_teacher_response.py"
SPEC = importlib.util.spec_from_file_location("package_corrective_teacher_response", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_package_joins_labels_by_sample_id_and_preserves_mask(tmp_path: Path):
    count = 2
    request = tmp_path / "request.npz"
    request_metadata = {
        "schema": MODULE.REQUEST_SCHEMA,
        "observation_contract": "split_reference_v2",
        "target_orientation_contract": MODULE.FRAME_CONTRACT,
    }
    mask = np.broadcast_to(MODULE.LEARNED_ACTION_MASK, (count, MODULE.ACTION_DIM)).copy()
    np.savez_compressed(
        request,
        observations=np.arange(count * 98, dtype=np.float32).reshape(count, 98),
        action_label_mask=mask,
        source_episode_index=np.asarray([7, 7], dtype=np.int32),
        rollout_step=np.asarray([10, 11], dtype=np.int32),
        rollout_waypoint_idx=np.asarray([20, 21], dtype=np.int32),
        trajectory_progress=np.asarray([0.1, 0.2], dtype=np.float32),
        metadata=json.dumps(request_metadata),
    )

    response_csv = tmp_path / "response.csv"
    fieldnames = [
        "sample_id",
        "source_episode_index",
        "rollout_step",
        "rollout_waypoint_idx",
        *(f"teacher_action_{index}" for index in MODULE.LEARNED_ACTION_INDICES),
        "runtime_transition_valid",
        "action_inside_envelope",
        "srdf_collision_free",
        "terminal_position_residual_m",
        "terminal_orientation_residual_deg",
        "accepted",
    ]
    with response_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in (1, 0):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_episode_index": 7,
                    "rollout_step": 10 + sample_id,
                    "rollout_waypoint_idx": 20 + sample_id,
                    **{
                        f"teacher_action_{index}": 0.1 * (sample_id + 1) + 0.01 * index
                        for index in MODULE.LEARNED_ACTION_INDICES
                    },
                    "runtime_transition_valid": 1,
                    "action_inside_envelope": 1,
                    "srdf_collision_free": 1,
                    "terminal_position_residual_m": 0.2,
                    "terminal_orientation_residual_deg": 2.0,
                    "accepted": 1,
                }
            )

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": MODULE.RESPONSE_SCHEMA,
                "frame_contract": MODULE.FRAME_CONTRACT,
                "request_sample_count": count,
                "sample_count": count,
                "all_accepted": True,
                "training_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "corrective.npz"
    metadata = MODULE.build_dataset(request, response_csv, summary, output, sample_weight=0.5)

    assert metadata["effective_weighted_rows"] == 1.0
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action_valid_mask"], mask)
        np.testing.assert_array_equal(data["source_index"], np.zeros(count, dtype=np.int32))
        np.testing.assert_allclose(data["sample_weight"], 0.5)
        for sample_id in range(count):
            expected = np.asarray(
                [0.1 * (sample_id + 1) + 0.01 * index for index in MODULE.LEARNED_ACTION_INDICES]
            )
            np.testing.assert_allclose(data["actions"][sample_id, MODULE.LEARNED_ACTION_INDICES], expected)
            np.testing.assert_array_equal(data["actions"][sample_id, 3:6], 0.0)


def test_package_rejects_unaccepted_response(tmp_path: Path):
    request = tmp_path / "request.npz"
    response = tmp_path / "response.csv"
    request.touch()
    response.touch()
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": MODULE.RESPONSE_SCHEMA,
                "frame_contract": MODULE.FRAME_CONTRACT,
                "all_accepted": False,
                "training_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    try:
        MODULE.build_dataset(
            request,
            response,
            summary,
            tmp_path / "output.npz",
        )
    except ValueError as error:
        assert "not fully accepted" in str(error)
    else:
        raise AssertionError("unaccepted response must be rejected")
