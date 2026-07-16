import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.exact_source_reference import (
    ExactSourceReference,
    discover_exact_source_references,
    source_anchor_execution_indices,
    validate_exact_source_candidate,
    validate_execution_plan_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/two_wheel_balance"))

import retarget_exact_source_v1_nonholonomic as exact_retarget  # noqa: E402
from seal_exact_source_candidate_metadata import seal  # noqa: E402


def write_package(root: Path) -> str:
    episode_dir = root / "episode_0001"
    episode_dir.mkdir(parents=True)
    source = {
        "poses": [
            {"time": 0.0, "position": [0.0, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"time": 0.2, "position": [0.2, 0.1, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        ]
    }
    source_path = episode_dir / "source.json"
    source_path.write_text(json.dumps(source, separators=(",", ":")), encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "gik_exact_source_reference_package_v1",
        "trajectory_integrity_contract": "exact_source_v1",
        "frame_contract": {
            "orientation": "semantic DFR body quaternion in xyzw order",
            "pose_target_link": "ee1_tool",
            "semantic_forward_axis": "+y in ee1_tool",
        },
        "episode_count": 1,
        "source_pose_count_total": 2,
        "integrity_passed": True,
        "quality_qualified_teacher": False,
        "valid_for_training": False,
        "items": [
            {
                "episode_index": 1,
                "bundled_source_json": "episode_0001/source.json",
                "source_json_sha256": source_sha,
                "source_pose_count": 2,
                "source_duration_s": 0.2,
                "trajectory_integrity_contract": "exact_source_v1",
                "integrity_passed": True,
                "quality_qualified_teacher": False,
                "valid_for_training": False,
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_reference_only_package_is_admitted_for_retargeting(tmp_path: Path) -> None:
    digest = write_package(tmp_path)
    references = discover_exact_source_references(
        tmp_path, expected_manifest_sha256=digest, expected_episodes=1
    )
    reference = references[1]
    np.testing.assert_allclose(reference.time_s, [0.0, 0.2])
    np.testing.assert_allclose(reference.attitudes_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 2)


def test_reference_loader_rejects_copied_package_by_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = write_package(tmp_path)
    monkeypatch.setattr(
        "rl_platform.tasks.two_wheel_balance.exact_source_reference.QUARANTINED_MANIFEST_SHA256S",
        frozenset({digest}),
    )
    renamed = tmp_path.parent / "renamed_old_package"
    tmp_path.rename(renamed)
    with pytest.raises(ValueError, match="quarantined upstream-truncated"):
        discover_exact_source_references(
            renamed, expected_manifest_sha256=digest, expected_episodes=1
        )


def test_reference_loader_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    digest = write_package(tmp_path)
    source_path = tmp_path / "episode_0001/source.json"
    source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        discover_exact_source_references(
            tmp_path, expected_manifest_sha256=digest, expected_episodes=1
        )


def test_anchor_mapping_preserves_every_ordered_source_anchor() -> None:
    source_position = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    source_attitude = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    execution_position = np.array(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
    )
    execution_attitude = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
    np.testing.assert_array_equal(
        source_anchor_execution_indices(
            source_position, source_attitude, execution_position, execution_attitude, 1
        ),
        [1, 3, 4],
    )


def test_anchor_mapping_rejects_missing_or_initialization_leaked_anchor() -> None:
    attitude = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    with pytest.raises(ValueError, match="missing source anchor"):
        source_anchor_execution_indices(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            attitude,
            np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]),
            attitude,
            0,
        )
    with pytest.raises(ValueError, match="initialization overlaps"):
        source_anchor_execution_indices(
            np.array([[0.0, 0.0, 1.0]]),
            attitude[:1],
            np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            attitude,
            0,
        )


def write_candidate(path: Path, **overrides) -> None:
    values = {
        "schema": np.asarray("cinebotrl_two_wheel_exact_source_retarget_v1"),
        "trajectory_integrity_contract": np.asarray("exact_source_v1"),
        "source_manifest_sha256": np.asarray(
            "f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
        ),
        "source_trajectory_integrity_passed": np.bool_(True),
        "source_reference_quality_qualified_teacher": np.bool_(False),
        "source_reference_valid_for_training": np.bool_(False),
        "physical_gimbal_joint_labels_included": np.bool_(False),
        "initialization_in_learned_actions": np.bool_(False),
        "valid_for_training": np.bool_(False),
        "execution_schedule_metadata_sealed": np.bool_(True),
        "acquisition_route_contract": np.asarray(
            "minimum_total_yaw_forward_or_reverse_v1"
        ),
        "base_acquisition_route": np.asarray("reverse"),
        "base_acquisition_total_yaw_travel_deg": np.float64(120.0),
        "offline_executable_quality_passed": np.bool_(True),
        "valid_for_dynamic_evaluation": np.bool_(True),
        "source_pose_count": np.int32(2),
        "source_time_s": np.array([0.0, 0.2]),
        "source_position_world_m": np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        "source_attitude_world_dfr_quat_xyzw": np.tile(
            [0.0, 0.0, 0.0, 1.0], (2, 1)
        ),
        "execution_time_s": np.array([0.0, 0.1, 0.2]),
        "execution_transition_dt_s": np.array([0.1, 0.1]),
        "time_s": np.array([0.0, 0.1, 0.2]),
        "target_position_world_m": np.array(
            [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        "target_attitude_world_dfr_quat_wxyz": np.tile(
            [1.0, 0.0, 0.0, 0.0], (3, 1)
        ),
        "base_arm_q": np.zeros((3, 6)),
        "control_v_wz_darm": np.zeros((2, 5)),
        "source_anchor_execution_index": np.array([1, 2]),
        "source_anchor_execution_time_s": np.array([0.1, 0.2]),
        "source_interval_execution_step_count": np.array([1]),
        "source_interval_execution_duration_s": np.array([0.1]),
        "semantic_start_index": np.int32(1),
        "initialization_sample_count": np.int32(1),
    }
    values.update(overrides)
    np.savez_compressed(path, **values)


def test_exact_source_candidate_contract_accepts_complete_mapping(tmp_path: Path) -> None:
    path = tmp_path / "candidate.npz"
    write_candidate(path)
    validate_exact_source_candidate(path)


def test_execution_plan_identity_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "candidate.npz"
    write_candidate(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_execution_plan_sha256(path, actual) == actual
    with pytest.raises(ValueError, match="execution-plan SHA-256 mismatch"):
        validate_execution_plan_sha256(path, "0" * 64)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"source_anchor_execution_index": np.array([2, 1])}, "mapping reordered"),
        ({"initialization_in_learned_actions": np.bool_(True)}, "initialization leaked"),
        (
            {"execution_schedule_metadata_sealed": np.bool_(False)},
            "schedule is not sealed",
        ),
        (
            {"acquisition_route_contract": np.asarray("forward_only_v0")},
            "route contract is not admitted",
        ),
        ({"base_acquisition_route": np.asarray("sideways")}, "route is invalid"),
        ({"control_v_wz_darm": np.zeros((3, 5))}, "M-1 transitions"),
        ({"execution_transition_dt_s": np.array([0.1, 0.2])}, "dt is not aligned"),
    ],
)
def test_exact_source_candidate_contract_rejects_bypass(
    tmp_path: Path, overrides: dict[str, np.ndarray], match: str
) -> None:
    path = tmp_path / "candidate.npz"
    write_candidate(path, **overrides)
    with pytest.raises(ValueError, match=match):
        validate_exact_source_candidate(path)


def write_upstream_seed(root: Path) -> tuple[ExactSourceReference, str]:
    source = {
        "poses": [
            {"time": 0.0, "position": [0.0, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"time": 0.2, "position": [0.2, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        ]
    }
    source_path = root / "source.json"
    source_path.write_text(json.dumps(source, separators=(",", ":")), encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_manifest_sha = "f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
    npz_path = root / "episode_0077_exact_source_upstream_seed_v1.npz"
    np.savez_compressed(
        npz_path,
        schema="gik9dof_exact_source_upstream_teacher_seed_v1",
        trajectory_integrity_contract="exact_source_v1",
        trajectory_integrity_passed=np.bool_(True),
        upstream_teacher_quality_passed=np.bool_(True),
        valid_for_two_wheel_retarget_input=np.bool_(True),
        two_wheel_dynamic_quality_passed=np.bool_(False),
        valid_for_training=np.bool_(False),
        source_package_manifest_sha256=source_manifest_sha,
        source_json_sha256=source_sha,
        action_provenance="no_policy_actions_in_this_package",
        seed_plant_contract="holonomic_base_arm_retarget_seed",
        source_time_s=np.array([0.0, 0.2]),
        source_position_m=np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]),
        source_attitude_world_dfr_quat_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        seed_time_s=np.array([0.0, 0.2]),
        seed_q_base_arm_6=np.zeros((2, 6)),
        source_anchor_to_seed_state_index=np.array([0, 1]),
        seed_state_count=np.int32(2),
        seed_transition_count=np.int32(1),
        initialization_state_count=np.int32(0),
    )
    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "gik9dof_exact_source_upstream_teacher_seed_manifest_v1",
        "episode_index": 77,
        "trajectory_integrity_contract": "exact_source_v1",
        "trajectory_integrity_passed": True,
        "upstream_teacher_quality_passed": True,
        "valid_for_two_wheel_retarget_input": True,
        "two_wheel_dynamic_quality_passed": False,
        "valid_for_training": False,
        "source_package_manifest_sha256": source_manifest_sha,
        "source_json_sha256": source_sha,
        "output_npz": npz_path.name,
        "output_npz_sha256": npz_sha,
        "bundled_source_json": source_path.name,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    reference = ExactSourceReference(
        episode_index=77,
        source_json=source_path,
        source_json_sha256=source_sha,
        manifest_sha256=source_manifest_sha,
        time_s=np.array([0.0, 0.2]),
        positions_m=np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]),
        attitudes_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
    )
    return reference, manifest_sha


def test_upstream_seed_is_solver_input_not_training_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, manifest_sha = write_upstream_seed(tmp_path)
    monkeypatch.setattr(
        exact_retarget,
        "UPSTREAM_SEED_MANIFEST_SHA256_BY_EPISODE",
        {77: manifest_sha},
    )
    teacher, diagnostics = exact_retarget.load_integrity_seed(tmp_path, reference)
    assert teacher.base_arm_q.shape == (2, 6)
    assert diagnostics["seed_quality_qualified"] is True
    assert diagnostics["seed_valid_for_training"] is False
    assert diagnostics["seed_two_wheel_dynamic_quality_passed"] is False


def test_initialization_endpoint_exports_exact_source_anchor_zero(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    reference = ExactSourceReference(
        episode_index=1,
        source_json=source_path,
        source_json_sha256="a" * 64,
        manifest_sha256="b" * 64,
        time_s=np.array([0.0, 0.2]),
        positions_m=np.array([[1.0, 2.0, 3.0], [1.2, 2.0, 3.0]]),
        attitudes_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
    )
    arrays = {
        "semantic_start_index": np.int32(2),
        "target_position_world_m": np.zeros((4, 3)),
        "target_attitude_world_dfr_quat_wxyz": np.zeros((4, 4)),
    }
    semantic_start = exact_retarget.install_exact_source_anchor_zero(arrays, reference)
    assert semantic_start == 2
    np.testing.assert_array_equal(arrays["target_position_world_m"][2], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        arrays["target_attitude_world_dfr_quat_wxyz"][2], [1.0, 0.0, 0.0, 0.0]
    )


def test_semantic_reference_builder_returns_reference_and_seed_diagnostic(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    reference = ExactSourceReference(
        episode_index=1,
        source_json=source_path,
        source_json_sha256="a" * 64,
        manifest_sha256="b" * 64,
        time_s=np.array([0.0, 0.2]),
        positions_m=np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]),
        attitudes_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
    )
    seed = exact_retarget.SparseTeacher(
        case=1,
        path=source_path,
        base_arm_q=np.zeros((2, 6)),
        time_s=reference.time_s,
        dfr_attitudes_wxyz=reference.attitudes_wxyz[1:],
        desired_positions_m=reference.positions_m,
        desired_attitudes_wxyz=reference.attitudes_wxyz,
        action_order=exact_retarget.EXPECTED_ACTION_ORDER,
    )

    class StubKinematics:
        @staticmethod
        def position(state: np.ndarray) -> np.ndarray:
            return np.array([state[0], state[1], 1.0])

    semantic, seed_error = exact_retarget.build_semantic_reference(
        reference, seed, StubKinematics()
    )
    assert semantic.case == 1
    assert seed_error == pytest.approx(0.2)


def test_seal_adds_deterministic_execution_schedule_metadata(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.npz"
    write_candidate(candidate)
    with np.load(candidate, allow_pickle=False) as data:
        arrays = {
            key: data[key]
            for key in data.files
            if key
            not in {
                "execution_transition_dt_s",
                "source_anchor_execution_time_s",
                "source_interval_execution_step_count",
                "source_interval_execution_duration_s",
            }
        }
    np.savez_compressed(candidate, **arrays)
    result = tmp_path / "candidate.result.json"
    result.write_text(
        json.dumps(
            {
                "case": 1,
                "passed": True,
                "base_acquisition_route": "reverse",
                "base_acquisition_total_yaw_travel_deg": 120.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = seal(candidate, result)
    assert len(digest) == 64
    validate_exact_source_candidate(candidate)
    sealed_result = json.loads(result.read_text())
    assert sealed_result["execution_schedule_metadata_sealed"] is True
    assert sealed_result["execution_plan_sha256"] == digest
    with np.load(candidate, allow_pickle=False) as data:
        assert bool(data["execution_schedule_metadata_sealed"].item())
        assert data["base_acquisition_route"].item() == "reverse"


def test_whole_body_runtime_separates_source_and_execution_clocks() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/smoke_all79_whole_body_playback.py"
    ).read_text(encoding="utf-8")
    assert 'candidate["source_time_s"][-1]' in source
    assert 'candidate["execution_time_s"][-1]' in source
    assert "np.array_equal(time_s, execution_time_s)" in source
    assert '"source_duration_s": source_duration_s' in source
    assert '"execution_duration_s": execution_duration_s' in source
    assert '"execution_plan_sha256": execution_plan_sha256' in source
    assert (
        '"execution_schedule_metadata_sealed": execution_schedule_metadata_sealed'
        in source
    )
    assert '"acquisition_route_contract": acquisition_route_contract' in source
    assert '"base_acquisition_route": base_acquisition_route' in source
    assert "--expected-execution-plan-sha256" in source
    assert "phase_time_s >= execution_duration_s" in source
    assert "phase_time_s >= source_duration_s" not in source
