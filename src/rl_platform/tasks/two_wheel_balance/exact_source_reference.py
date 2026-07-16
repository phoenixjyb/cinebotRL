"""Fail-closed loader for immutable exact-source trajectory references."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


PACKAGE_SCHEMA = "gik_exact_source_reference_package_v1"
INTEGRITY_CONTRACT = "exact_source_v1"
EXPECTED_MANIFEST_SHA256 = (
    "f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
)
QUARANTINED_MANIFEST_SHA256S = frozenset(
    {"af035fb50f17322add90bf008427c9247dbbf08ee0bc38dd6d24172d9e3e14e4"}
)


@dataclass(frozen=True)
class ExactSourceReference:
    episode_index: int
    source_json: Path
    source_json_sha256: str
    manifest_sha256: str
    time_s: np.ndarray
    positions_m: np.ndarray
    attitudes_xyzw: np.ndarray

    @property
    def attitudes_wxyz(self) -> np.ndarray:
        return self.attitudes_xyzw[:, [3, 0, 1, 2]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_item(
    package_dir: Path,
    item: dict[str, object],
    manifest_sha256: str,
) -> ExactSourceReference:
    episode = int(item.get("episode_index", -1))
    source_json = (package_dir / str(item.get("bundled_source_json", ""))).resolve()
    expected_source_sha = str(item.get("source_json_sha256", ""))
    _require(source_json.is_file(), f"missing source JSON for episode {episode}")
    _require(
        sha256(source_json) == expected_source_sha,
        f"source JSON checksum mismatch for episode {episode}",
    )
    _require(
        item.get("trajectory_integrity_contract") == INTEGRITY_CONTRACT,
        f"episode {episode} lacks {INTEGRITY_CONTRACT}",
    )
    _require(item.get("integrity_passed") is True, f"episode {episode} integrity failed")
    _require(
        item.get("quality_qualified_teacher") is False,
        f"reference episode {episode} unexpectedly claims teacher quality",
    )
    _require(
        item.get("valid_for_training") is False,
        f"reference episode {episode} unexpectedly claims training validity",
    )

    payload = json.loads(source_json.read_text(encoding="utf-8"))
    poses = payload.get("poses")
    _require(isinstance(poses, list) and len(poses) >= 2, f"bad poses in {source_json}")
    time_s = np.asarray([pose["time"] for pose in poses], dtype=np.float64)
    positions = np.asarray([pose["position"] for pose in poses], dtype=np.float64)
    attitudes = np.asarray([pose["orientation"] for pose in poses], dtype=np.float64)
    count = len(poses)
    _require(
        int(item.get("source_pose_count", -1)) == count,
        f"source pose count mismatch for episode {episode}",
    )
    _require(time_s.shape == (count,) and np.isfinite(time_s).all(), f"bad source time for episode {episode}")
    _require(
        time_s[0] == 0.0 and bool(np.all(np.diff(time_s) > 0.0)),
        f"non-increasing source time for episode {episode}",
    )
    _require(positions.shape == (count, 3) and np.isfinite(positions).all(), f"bad source positions for episode {episode}")
    _require(attitudes.shape == (count, 4) and np.isfinite(attitudes).all(), f"bad source attitudes for episode {episode}")
    _require(
        bool(np.allclose(np.linalg.norm(attitudes, axis=1), 1.0, atol=1e-9)),
        f"non-unit source attitude for episode {episode}",
    )
    _require(
        abs(float(time_s[-1]) - float(item.get("source_duration_s", -1.0))) <= 1e-9,
        f"source duration mismatch for episode {episode}",
    )
    return ExactSourceReference(
        episode_index=episode,
        source_json=source_json,
        source_json_sha256=expected_source_sha,
        manifest_sha256=manifest_sha256,
        time_s=time_s,
        positions_m=positions,
        attitudes_xyzw=attitudes,
    )


def discover_exact_source_references(
    package_dir: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_episodes: int = 79,
) -> dict[int, ExactSourceReference]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    _require(manifest_path.is_file(), f"missing exact-source manifest: {manifest_path}")
    manifest_sha256 = sha256(manifest_path)
    _require(
        manifest_sha256 not in QUARANTINED_MANIFEST_SHA256S,
        f"source package is quarantined upstream-truncated lineage: {manifest_sha256}",
    )
    _require(
        manifest_sha256 == expected_manifest_sha256,
        f"unexpected exact-source manifest SHA-256: {manifest_sha256}",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == PACKAGE_SCHEMA, "wrong exact-source package schema")
    _require(
        manifest.get("trajectory_integrity_contract") == INTEGRITY_CONTRACT,
        f"package lacks {INTEGRITY_CONTRACT}",
    )
    _require(manifest.get("integrity_passed") is True, "package integrity failed")
    _require(
        manifest.get("quality_qualified_teacher") is False,
        "reference package unexpectedly claims teacher quality",
    )
    _require(
        manifest.get("valid_for_training") is False,
        "reference package unexpectedly claims training validity",
    )
    frame = manifest.get("frame_contract", {})
    _require(frame.get("pose_target_link") == "ee1_tool", "wrong source target link")
    _require(
        frame.get("orientation") == "semantic DFR body quaternion in xyzw order",
        "wrong source attitude contract",
    )
    _require(frame.get("semantic_forward_axis") == "+y in ee1_tool", "wrong source forward axis")
    items = manifest.get("items")
    _require(isinstance(items, list), "exact-source manifest has no items")
    _require(len(items) == expected_episodes, "exact-source episode count changed")
    references: dict[int, ExactSourceReference] = {}
    for item in items:
        _require(isinstance(item, dict), "invalid exact-source manifest item")
        reference = _load_item(package_dir, item, manifest_sha256)
        _require(reference.episode_index not in references, f"duplicate episode {reference.episode_index}")
        references[reference.episode_index] = reference
    expected = set(range(1, expected_episodes + 1))
    _require(set(references) == expected, f"exact-source episodes differ: {sorted(set(references) ^ expected)}")
    _require(
        sum(len(reference.time_s) for reference in references.values())
        == int(manifest.get("source_pose_count_total", -1)),
        "exact-source total pose count mismatch",
    )
    return references


def source_anchor_execution_indices(
    source_positions_m: np.ndarray,
    source_attitudes_wxyz: np.ndarray,
    execution_positions_m: np.ndarray,
    execution_attitudes_wxyz: np.ndarray,
    semantic_start_index: int,
    *,
    position_tolerance_m: float = 1e-10,
    attitude_tolerance_rad: float = 1e-7,
) -> np.ndarray:
    """Map every immutable source anchor to one ordered execution sample."""

    source_positions_m = np.asarray(source_positions_m, dtype=np.float64)
    source_attitudes_wxyz = np.asarray(source_attitudes_wxyz, dtype=np.float64)
    execution_positions_m = np.asarray(execution_positions_m, dtype=np.float64)
    execution_attitudes_wxyz = np.asarray(execution_attitudes_wxyz, dtype=np.float64)
    source_count = len(source_positions_m)
    _require(source_positions_m.shape == (source_count, 3), "bad source position shape")
    _require(source_attitudes_wxyz.shape == (source_count, 4), "bad source attitude shape")
    _require(execution_positions_m.ndim == 2 and execution_positions_m.shape[1] == 3, "bad execution position shape")
    _require(execution_attitudes_wxyz.shape == (len(execution_positions_m), 4), "bad execution attitude shape")
    _require(0 <= semantic_start_index < len(execution_positions_m), "bad semantic start index")
    mapping = []
    search_start = semantic_start_index
    for source_index in range(source_count):
        position_error = np.linalg.norm(
            execution_positions_m[search_start:] - source_positions_m[source_index],
            axis=1,
        )
        dots = np.abs(
            execution_attitudes_wxyz[search_start:]
            @ source_attitudes_wxyz[source_index]
        )
        attitude_error = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
        matches = np.flatnonzero(
            (position_error <= position_tolerance_m)
            & (attitude_error <= attitude_tolerance_rad)
        )
        _require(matches.size > 0, f"missing source anchor {source_index}")
        execution_index = search_start + int(matches[0])
        mapping.append(execution_index)
        search_start = execution_index + 1
    result = np.asarray(mapping, dtype=np.int64)
    _require(result[0] == semantic_start_index, "initialization overlaps source anchors")
    _require(bool(np.all(np.diff(result) > 0)), "source anchor mapping is not strict")
    return result


def validate_exact_source_candidate(
    path: Path,
    *,
    require_offline_quality: bool = True,
    require_dynamic_approval: bool = True,
) -> dict[str, np.ndarray]:
    """Validate a retarget artifact without trusting its directory or summary."""

    path = path.resolve()
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    def scalar(key: str):
        _require(key in arrays, f"candidate is missing {key}: {path}")
        _require(arrays[key].size == 1, f"candidate {key} must be scalar: {path}")
        return arrays[key].reshape(-1)[0].item()

    _require(
        scalar("schema") == "cinebotrl_two_wheel_exact_source_retarget_v1",
        f"candidate has obsolete schema: {path}",
    )
    _require(
        scalar("trajectory_integrity_contract") == INTEGRITY_CONTRACT,
        f"candidate lacks {INTEGRITY_CONTRACT}: {path}",
    )
    _require(
        scalar("source_manifest_sha256") == EXPECTED_MANIFEST_SHA256,
        f"candidate source manifest is not admitted: {path}",
    )
    _require(bool(scalar("source_trajectory_integrity_passed")), f"candidate source integrity failed: {path}")
    _require(not bool(scalar("source_reference_quality_qualified_teacher")), f"reference-only input claims teacher quality: {path}")
    _require(not bool(scalar("source_reference_valid_for_training")), f"reference-only input claims training validity: {path}")
    _require(not bool(scalar("physical_gimbal_joint_labels_included")), f"physical gimbal labels leaked: {path}")
    _require(not bool(scalar("initialization_in_learned_actions")), f"initialization leaked into learned actions: {path}")
    _require(not bool(scalar("valid_for_training")), f"offline candidate claims training validity: {path}")
    if require_offline_quality:
        _require(bool(scalar("offline_executable_quality_passed")), f"offline executable quality failed: {path}")
    if require_dynamic_approval:
        _require(bool(scalar("valid_for_dynamic_evaluation")), f"candidate is not admitted for dynamic evaluation: {path}")

    source_count = int(scalar("source_pose_count"))
    source_time = arrays.get("source_time_s")
    source_position = arrays.get("source_position_world_m")
    source_attitude_xyzw = arrays.get("source_attitude_world_dfr_quat_xyzw")
    execution_time = arrays.get("execution_time_s")
    execution_dt = arrays.get("execution_transition_dt_s")
    target_position = arrays.get("target_position_world_m")
    target_attitude = arrays.get("target_attitude_world_dfr_quat_wxyz")
    states = arrays.get("base_arm_q")
    controls = arrays.get("control_v_wz_darm")
    mapping = arrays.get("source_anchor_execution_index")
    _require(source_time is not None and source_time.shape == (source_count,), f"bad immutable source time: {path}")
    _require(source_position is not None and source_position.shape == (source_count, 3), f"bad immutable source positions: {path}")
    _require(source_attitude_xyzw is not None and source_attitude_xyzw.shape == (source_count, 4), f"bad immutable source attitudes: {path}")
    _require(source_time[0] == 0.0 and bool(np.all(np.diff(source_time) > 0.0)), f"source timestamps changed: {path}")
    _require(execution_time is not None and execution_time.ndim == 1, f"bad execution time: {path}")
    execution_count = len(execution_time)
    _require(execution_time[0] == 0.0 and bool(np.all(np.diff(execution_time) > 0.0)), f"bad execution schedule: {path}")
    _require(
        execution_dt is not None
        and execution_dt.shape == (execution_count - 1,)
        and np.array_equal(execution_dt, np.diff(execution_time)),
        f"execution transition dt is not aligned: {path}",
    )
    _require(target_position is not None and target_position.shape == (execution_count, 3), f"bad execution positions: {path}")
    _require(target_attitude is not None and target_attitude.shape == (execution_count, 4), f"bad execution attitudes: {path}")
    _require(states is not None and states.shape == (execution_count, 6), f"bad execution states: {path}")
    _require(controls is not None and controls.shape == (execution_count - 1, 5), f"execution does not have M-1 transitions: {path}")
    _require(mapping is not None and mapping.shape == (source_count,), f"bad source anchor mapping: {path}")
    _require(bool(np.all(np.diff(mapping) > 0)), f"source anchor mapping reordered: {path}")
    semantic_start = int(scalar("semantic_start_index"))
    _require(int(scalar("initialization_sample_count")) == semantic_start, f"initialization count mismatch: {path}")
    _require(int(mapping[0]) == semantic_start, f"initialization overlaps semantic anchors: {path}")
    _require(int(mapping[-1]) < execution_count, f"source anchor mapping out of range: {path}")
    anchor_time = arrays.get("source_anchor_execution_time_s")
    interval_steps = arrays.get("source_interval_execution_step_count")
    interval_duration = arrays.get("source_interval_execution_duration_s")
    _require(
        anchor_time is not None
        and anchor_time.shape == (source_count,)
        and np.array_equal(anchor_time, execution_time[mapping]),
        f"source anchor execution time mismatch: {path}",
    )
    _require(
        interval_steps is not None
        and interval_steps.shape == (source_count - 1,)
        and np.array_equal(interval_steps, np.diff(mapping)),
        f"source interval densification mismatch: {path}",
    )
    _require(
        interval_duration is not None
        and interval_duration.shape == (source_count - 1,)
        and np.array_equal(interval_duration, np.diff(anchor_time)),
        f"source interval execution duration mismatch: {path}",
    )
    mapped_position = target_position[mapping]
    mapped_attitude = target_attitude[mapping]
    source_attitude_wxyz = source_attitude_xyzw[:, [3, 0, 1, 2]]
    _require(bool(np.allclose(mapped_position, source_position, atol=1e-10, rtol=0.0)), f"source position anchors were replaced: {path}")
    dots = np.abs(np.sum(mapped_attitude * source_attitude_wxyz, axis=1))
    _require(bool(np.all(dots >= 1.0 - 1e-10)), f"source attitude anchors were replaced: {path}")
    return arrays
