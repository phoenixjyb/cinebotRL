from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/two_wheel_balance"))

from rl_platform.tasks.two_wheel_balance.exact_source_checkpoint import (
    ExactSourceRetargetPrefix,
    canonical_json_sha256,
    load_exact_source_checkpoint,
    save_exact_source_checkpoint,
    source_time_sha256,
)

import retarget_corrected_teacher_v3_nonholonomic as retarget  # noqa: E402
import retarget_exact_source_v1_nonholonomic as exact_retarget  # noqa: E402


def checkpoint_identity(source_time_s: np.ndarray) -> dict[str, object]:
    config = {
        "maximum_linear_velocity": 0.4,
        "maximum_semantic_gimbal_reserve_search_scale": 24,
    }
    return {
        "git_commit": "a" * 40,
        "code_contract_sha256": "b" * 64,
        "case": 7,
        "retarget_cli_config": config,
        "retarget_cli_config_sha256": canonical_json_sha256(config),
        "reference_manifest_sha256": "c" * 64,
        "reference_episode_sha256": "d" * 64,
        "integrity_seed_sha256": "e" * 64,
        "target_urdf_sha256": "f" * 64,
        "source_pose_count": len(source_time_s),
        "source_time_sha256": source_time_sha256(source_time_s),
    }


def checkpoint_fixture() -> tuple[
    ExactSourceRetargetPrefix, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    source_time = np.array([0.0, 0.2, 0.4])
    source_positions = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    source_attitudes = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    anchor = np.zeros(9)
    prefix = ExactSourceRetargetPrefix(
        states=np.zeros((3, 9)),
        controls=np.zeros((2, 5)),
        target_positions=np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
        ),
        target_attitudes=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        execution_time_s=np.array([0.0, 0.2, 0.4]),
        position_errors_m=np.zeros(3),
        attitude_errors_deg=np.zeros(3),
        previous_control=np.zeros(8),
        source_anchor_execution_index_prefix=np.array([0, 2]),
        retimed_interval_count=1,
        next_source_interval=2,
    )
    return prefix, source_time, source_positions, source_attitudes, anchor


def save_fixture(path: Path) -> tuple[dict[str, object], tuple[np.ndarray, ...]]:
    prefix, source_time, source_positions, source_attitudes, anchor = (
        checkpoint_fixture()
    )
    identity = checkpoint_identity(source_time)
    save_exact_source_checkpoint(
        path,
        identity,
        prefix,
        source_time_s=source_time,
        source_positions_m=source_positions,
        source_attitudes_wxyz=source_attitudes,
        expected_anchor=anchor,
    )
    return identity, (source_time, source_positions, source_attitudes, anchor)


def test_checkpoint_atomic_round_trip_preserves_complete_prefix(tmp_path: Path) -> None:
    path = tmp_path / "ep7.checkpoint.npz"
    identity, source = save_fixture(path)
    loaded = load_exact_source_checkpoint(
        path,
        identity,
        source_time_s=source[0],
        source_positions_m=source[1],
        source_attitudes_wxyz=source[2],
        expected_anchor=source[3],
    )
    expected = checkpoint_fixture()[0]
    assert loaded.next_source_interval == 2
    assert loaded.retimed_interval_count == 1
    for key, value in expected.arrays().items():
        np.testing.assert_array_equal(loaded.arrays()[key], value)
    assert not list(tmp_path.glob(".ep7.checkpoint.npz.*.tmp"))


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("git_commit", "1" * 40),
        ("code_contract_sha256", "1" * 64),
        ("reference_manifest_sha256", "1" * 64),
        ("reference_episode_sha256", "1" * 64),
        ("integrity_seed_sha256", "1" * 64),
        ("target_urdf_sha256", "1" * 64),
        ("source_time_sha256", "1" * 64),
    ],
)
def test_checkpoint_rejects_identity_hash_mismatch(
    tmp_path: Path, field: str, replacement: str
) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, source = save_fixture(path)
    changed = {**identity, field: replacement}
    with pytest.raises(ValueError, match="identity mismatch|source time hash differs"):
        load_exact_source_checkpoint(
            path,
            changed,
            source_time_s=source[0],
            source_positions_m=source[1],
            source_attitudes_wxyz=source[2],
            expected_anchor=source[3],
        )


def test_checkpoint_rejects_cli_config_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, source = save_fixture(path)
    config = {**identity["retarget_cli_config"], "maximum_linear_velocity": 0.41}
    changed = {
        **identity,
        "retarget_cli_config": config,
        "retarget_cli_config_sha256": canonical_json_sha256(config),
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        load_exact_source_checkpoint(
            path,
            changed,
            source_time_s=source[0],
            source_positions_m=source[1],
            source_attitudes_wxyz=source[2],
            expected_anchor=source[3],
        )


def test_checkpoint_rejects_actual_source_time_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, source = save_fixture(path)
    changed_time = source[0].copy()
    changed_time[-1] += 0.01
    with pytest.raises(ValueError, match="source time hash differs"):
        load_exact_source_checkpoint(
            path,
            identity,
            source_time_s=changed_time,
            source_positions_m=source[1],
            source_attitudes_wxyz=source[2],
            expected_anchor=source[3],
        )


def test_checkpoint_rejects_truncated_archive(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, source = save_fixture(path)
    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(ValueError, match="invalid exact-source checkpoint"):
        load_exact_source_checkpoint(
            path,
            identity,
            source_time_s=source[0],
            source_positions_m=source[1],
            source_attitudes_wxyz=source[2],
            expected_anchor=source[3],
        )


def test_checkpoint_rejects_execution_clock_not_implied_by_source_map(
    tmp_path: Path,
) -> None:
    prefix, source_time, positions, attitudes, anchor = checkpoint_fixture()
    changed = replace(prefix, execution_time_s=np.array([0.0, 0.1, 0.2]))
    with pytest.raises(ValueError, match="transition dt differs"):
        save_exact_source_checkpoint(
            tmp_path / "bad-clock.npz",
            checkpoint_identity(source_time),
            changed,
            source_time_s=source_time,
            source_positions_m=positions,
            source_attitudes_wxyz=attitudes,
            expected_anchor=anchor,
        )


def test_checkpoint_rejects_retimed_count_not_implied_by_source_map(
    tmp_path: Path,
) -> None:
    prefix, source_time, positions, attitudes, anchor = checkpoint_fixture()
    changed = replace(prefix, retimed_interval_count=0)
    with pytest.raises(ValueError, match="retimed interval count differs"):
        save_exact_source_checkpoint(
            tmp_path / "bad-retimed-count.npz",
            checkpoint_identity(source_time),
            changed,
            source_time_s=source_time,
            source_positions_m=positions,
            source_attitudes_wxyz=attitudes,
            expected_anchor=anchor,
        )


class StubPositionKinematics:
    arm_lower = np.full(3, -2.0)
    arm_upper = np.full(3, 2.0)

    @staticmethod
    def position(state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], 0.0])

    @staticmethod
    def gravitational_effort_nm(state: np.ndarray) -> np.ndarray:
        return np.zeros(3)

    @staticmethod
    def equilibrium_pitch_rad(state: np.ndarray, wheel_height: float) -> float:
        return 0.0


class StubCameraKinematics:
    gimbal_lower = np.full(3, -2.0)
    gimbal_upper = np.full(3, 2.0)

    def __init__(self, target_rotation: np.ndarray):
        self.target_rotation = target_rotation

    def world_rotation(
        self, root_quaternion: np.ndarray, arm: np.ndarray, gimbal: np.ndarray
    ) -> np.ndarray:
        return self.target_rotation


def solver_args(**overrides: object) -> SimpleNamespace:
    values = {
        "integrity_seed_prior_only": True,
        "rebuild_com_safe_seed_prior": False,
        "maximum_linear_velocity": 0.4,
        "maximum_yaw_rate": 0.4,
        "maximum_arm_rate": 0.5,
        "maximum_gimbal_rate": 0.25,
        "maximum_ik_error_deg": 0.1,
        "maximum_arm_gravity_effort_nm": 29.5,
        "gravity_effort_tolerance_nm": 0.01,
        "maximum_equilibrium_pitch_deg": 10.0,
        "wheel_axle_height_m": 0.1016,
        "camera_solve_root_model": "balanced",
        "minimum_semantic_gimbal_limit_margin_ratio": 0.005,
        "minimum_semantic_gimbal_reserve_margin_ratio": 0.01,
        "maximum_semantic_gimbal_reserve_search_scale": 24,
        "semantic_gimbal_center_regularization": 0.1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def stationary_problem() -> tuple[object, np.ndarray, np.ndarray, object, object]:
    attitude = np.array([1.0, 0.0, 0.0, 0.0])
    reference = retarget.SemanticReference(
        case=7,
        source_mat=Path("source.json"),
        time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        positions_m=np.zeros((4, 3)),
        attitudes_wxyz=np.tile(attitude, (4, 1)),
        source_fk_max_error_m=0.0,
        package_position_max_error_m=0.0,
        package_q_max_error_rad=0.0,
        package_time_max_error_s=0.0,
        package_attitude_max_error_deg=0.0,
    )
    target_rotation = retarget.quaternion_matrix_wxyz(
        retarget.semantic_dfr_to_physical_cam_quat_wxyz(attitude)
    )
    return (
        reference,
        np.zeros(9),
        np.zeros((4, 6)),
        StubPositionKinematics(),
        StubCameraKinematics(target_rotation),
    )


def test_resumed_solver_is_identical_to_uninterrupted_solver(tmp_path: Path) -> None:
    problem = stationary_problem()
    uninterrupted = retarget.retarget_semantic_full_pose(
        *problem, solver_args()
    )
    source_time = problem[0].time_s
    identity = checkpoint_identity(source_time)
    checkpoint = tmp_path / "resume.npz"
    save_exact_source_checkpoint(
        checkpoint,
        identity,
        ExactSourceRetargetPrefix(
            states=uninterrupted[1][:2],
            controls=uninterrupted[2][:1],
            target_positions=uninterrupted[3][:2],
            target_attitudes=uninterrupted[4][:2],
            execution_time_s=uninterrupted[0][:2],
            position_errors_m=uninterrupted[5][:2],
            attitude_errors_deg=uninterrupted[6][:2],
            previous_control=np.zeros(8),
            source_anchor_execution_index_prefix=np.array([0, 1]),
            retimed_interval_count=0,
            next_source_interval=2,
        ),
        source_time_s=source_time,
        source_positions_m=problem[0].positions_m,
        source_attitudes_wxyz=problem[0].attitudes_wxyz,
        expected_anchor=problem[1],
    )
    resumed = retarget.retarget_semantic_full_pose(
        *problem,
        solver_args(
            exact_source_checkpoint_path=checkpoint,
            exact_source_checkpoint_identity=identity,
            exact_source_resume_checkpoint=True,
        ),
    )
    for expected, actual in zip(uninterrupted[:-1], resumed[:-1], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert resumed[-1] == uninterrupted[-1]


def test_final_interval_checkpoint_is_written_off_cadence(tmp_path: Path) -> None:
    problem = stationary_problem()
    source_time = problem[0].time_s
    identity = checkpoint_identity(source_time)
    checkpoint = tmp_path / "final-off-cadence.npz"
    retarget.retarget_semantic_full_pose(
        *problem,
        solver_args(
            exact_source_checkpoint_path=checkpoint,
            exact_source_checkpoint_identity=identity,
            exact_source_resume_checkpoint=False,
            checkpoint_cadence_source_intervals=10,
        ),
    )
    loaded = load_exact_source_checkpoint(
        checkpoint,
        identity,
        source_time_s=source_time,
        source_positions_m=problem[0].positions_m,
        source_attitudes_wxyz=problem[0].attitudes_wxyz,
        expected_anchor=problem[1],
    )
    assert loaded.next_source_interval == len(source_time)
    np.testing.assert_array_equal(
        loaded.source_anchor_execution_index_prefix,
        np.arange(len(source_time)),
    )


def test_successful_finalization_removes_checkpoint_after_both_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"recoverable")
    output = tmp_path / "case.npz"
    result = tmp_path / "case.result.json"
    summary: dict[str, object] = {"valid_for_training": False}
    exact_retarget.finalize_case_artifacts(
        output,
        result,
        {"valid_for_training": np.bool_(False), "values": np.arange(3)},
        summary,
        checkpoint,
    )
    assert output.is_file()
    assert result.is_file()
    assert not checkpoint.exists()
    assert len(str(summary["candidate_sha256"])) == 64


def test_json_finalization_failure_preserves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"recoverable")
    output = tmp_path / "case.npz"

    def fail_json(path: Path, value: object) -> None:
        raise OSError("simulated JSON persistence failure")

    monkeypatch.setattr(exact_retarget, "_atomic_write_json", fail_json)
    with pytest.raises(OSError, match="simulated JSON persistence failure"):
        exact_retarget.finalize_case_artifacts(
            output,
            tmp_path / "case.result.json",
            {"valid_for_training": np.bool_(False), "values": np.arange(3)},
            {"valid_for_training": False},
            checkpoint,
        )
    assert output.is_file()
    assert checkpoint.read_bytes() == b"recoverable"


def test_legacy_solver_without_checkpoint_remains_unchanged() -> None:
    problem = stationary_problem()
    first = retarget.retarget_semantic_full_pose(*problem, solver_args())
    second = retarget.retarget_semantic_full_pose(*problem, solver_args())
    for expected, actual in zip(first[:-1], second[:-1], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert first[-1] == second[-1]
