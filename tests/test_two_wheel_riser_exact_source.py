import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_exact_source import (
    audit_exact_source_playback_plan,
    execution_schedule_for_source,
    load_exact_source_package,
    save_exact_source_playback_plan,
)
from rl_platform.tasks.two_wheel_balance.riser_playback import RiserPlaybackPlan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path: Path) -> Path:
    episode = tmp_path / "episode_0001"
    episode.mkdir(parents=True)
    source = episode / "source.json"
    source.write_text(
        json.dumps(
            {
                "first_point_vx_prefer": -0.4,
                "poses": [
                    {"position": [0.0, 0.0, 0.5], "orientation": [0.0, 0.0, 0.0, 1.0], "time": 0.0},
                    {"position": [0.03, 0.0, 0.51], "orientation": [0.0, 0.0, 0.0, 1.0], "time": 0.001},
                    {"position": [0.06, 0.0, 0.52], "orientation": [0.0, 0.0, 0.0, 1.0], "time": 0.02},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gik_exact_source_reference_package_v1",
                "trajectory_integrity_contract": "exact_source_v1",
                "episode_count": 1,
                "integrity_passed": True,
                "quality_qualified_teacher": False,
                "valid_for_training": False,
                "frame_contract": {
                    "pose_target_link": "ee1_tool",
                    "semantic_forward_axis": "+y in ee1_tool",
                },
                "items": [
                    {
                        "episode_index": 1,
                        "bundled_source_json": "episode_0001/source.json",
                        "source_json_sha256": _sha(source),
                        "source_pose_count": 3,
                        "source_duration_s": 0.02,
                        "source_path_length_m": float(2.0 * np.linalg.norm([0.03, 0.0, 0.01])),
                        "trajectory_integrity_contract": "exact_source_v1",
                        "integrity_passed": True,
                        "quality_qualified_teacher": False,
                        "valid_for_training": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _plan(time_s: np.ndarray) -> RiserPlaybackPlan:
    count = len(time_s)
    position = np.column_stack(
        (np.arange(count) * 0.03, np.zeros(count), 0.6 + np.arange(count) * 0.01)
    )
    return RiserPlaybackPlan(
        case=1,
        time_s=time_s,
        target_position_world_m=position,
        target_semantic_dfr_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        base_xy_yaw=np.column_stack((np.arange(count) * 0.01, np.zeros(count), np.zeros(count))),
        riser_q=np.arange(count) * 0.01,
        proxy_gimbal_q=np.zeros((count, 3)),
        feedforward_v_wz=np.zeros((count - 1, 2)),
        feedforward_riser_velocity=np.zeros(count - 1),
        feedforward_proxy_velocity=np.zeros((count - 1, 3)),
        vertical_shift_m=0.1,
        planning_strategy="fixed_path",
    )


def test_exact_source_loader_and_retimer_preserve_reference(tmp_path: Path) -> None:
    manifest = _package(tmp_path)
    references = load_exact_source_package(
        manifest, expected_manifest_sha256=_sha(manifest), expected_count=1
    )
    reference = references[1]
    execution_time = execution_schedule_for_source(reference)
    np.testing.assert_array_equal(reference.source_time_s, [0.0, 0.001, 0.02])
    assert execution_time[1] >= 0.1
    assert reference.initial_base_yaw_rad == pytest.approx(np.pi)
    planning = reference.planning_reference(execution_time)
    np.testing.assert_array_equal(planning.positions_m, reference.source_position_world_m)
    np.testing.assert_array_equal(planning.time_s, execution_time)


def test_exact_source_plan_audit_binds_all_anchors_and_blocks_training(tmp_path: Path) -> None:
    manifest = _package(tmp_path)
    reference = load_exact_source_package(
        manifest, expected_manifest_sha256=_sha(manifest), expected_count=1
    )[1]
    execution_time = execution_schedule_for_source(reference)
    path = tmp_path / "plan.npz"
    save_exact_source_playback_plan(path, _plan(execution_time), reference)
    audit = audit_exact_source_playback_plan(path, reference)
    assert audit["passed"]
    assert audit["source_pose_count"] == 3
    assert audit["retargeted_waypoint_state_count"] == 3
    assert audit["transition_count"] == 2
    assert audit["maximum_mapped_position_error_m"] == 0.0
    assert not audit["quality_gate_passed"]
    assert not audit["valid_for_training"]


def test_exact_source_plan_audit_rejects_missing_reordered_and_initialization_leak(
    tmp_path: Path,
) -> None:
    manifest = _package(tmp_path)
    reference = load_exact_source_package(
        manifest, expected_manifest_sha256=_sha(manifest), expected_count=1
    )[1]
    execution_time = execution_schedule_for_source(reference)
    good = tmp_path / "good.npz"
    save_exact_source_playback_plan(good, _plan(execution_time), reference)
    with np.load(good, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    arrays["source_anchor_execution_index"] = np.array([0, 2, 1])
    reordered = tmp_path / "reordered.npz"
    np.savez_compressed(reordered, **arrays)
    assert not audit_exact_source_playback_plan(reordered, reference)["passed"]

    arrays["source_anchor_execution_index"] = np.array([0, 1])
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, **arrays)
    assert not audit_exact_source_playback_plan(missing, reference)["passed"]

    arrays["source_anchor_execution_index"] = np.arange(3)
    arrays["initialization_time_s"] = np.array([0.0])
    leaked = tmp_path / "leaked.npz"
    np.savez_compressed(leaked, **arrays)
    assert not audit_exact_source_playback_plan(leaked, reference)["passed"]


def test_exact_source_loader_rejects_copied_or_modified_source(tmp_path: Path) -> None:
    manifest = _package(tmp_path)
    source = tmp_path / "episode_0001/source.json"
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_exact_source_package(
            manifest, expected_manifest_sha256=_sha(manifest), expected_count=1
        )
