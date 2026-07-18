from dataclasses import replace
import json
from pathlib import Path
import subprocess
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


def test_semantic_branch_lookback_truncates_complete_prefix_and_control() -> None:
    states = np.arange(63, dtype=np.float64).reshape(7, 9) / 100.0
    controls = np.arange(30, dtype=np.float64).reshape(6, 5) / 10.0
    prefix = ExactSourceRetargetPrefix(
        states=states,
        controls=controls,
        target_positions=np.arange(21, dtype=np.float64).reshape(7, 3),
        target_attitudes=np.tile([1.0, 0.0, 0.0, 0.0], (7, 1)),
        execution_time_s=np.arange(7, dtype=np.float64) * 0.2,
        position_errors_m=np.arange(7, dtype=np.float64) / 100.0,
        attitude_errors_deg=np.arange(7, dtype=np.float64) / 10.0,
        previous_control=np.full(8, 99.0),
        source_anchor_execution_index_prefix=np.array([0, 2, 4, 6]),
        retimed_interval_count=3,
        next_source_interval=4,
    )

    rewound = retarget.truncate_exact_source_prefix_for_semantic_lookback(
        prefix, lookback_source_intervals=2
    )

    assert rewound.next_source_interval == 2
    assert rewound.retimed_interval_count == 1
    np.testing.assert_array_equal(
        rewound.source_anchor_execution_index_prefix, [0, 2]
    )
    np.testing.assert_array_equal(rewound.states, states[:3])
    np.testing.assert_array_equal(rewound.controls, controls[:2])
    np.testing.assert_array_equal(rewound.previous_control[:5], controls[1])
    np.testing.assert_array_equal(
        rewound.previous_control[5:8], states[2, 6:9] - states[1, 6:9]
    )


def test_semantic_branch_lookback_rejects_anchor_zero_and_bad_map() -> None:
    prefix = checkpoint_fixture()[0]
    with pytest.raises(ValueError, match="anchor zero"):
        retarget.truncate_exact_source_prefix_for_semantic_lookback(prefix, 1)
    bad = replace(
        prefix,
        source_anchor_execution_index_prefix=np.array([0]),
    )
    with pytest.raises(ValueError, match="map is incomplete"):
        retarget.truncate_exact_source_prefix_for_semantic_lookback(bad, 1)


def test_windows_project_root_uses_wsl_git_path() -> None:
    assert (
        exact_retarget._windows_project_root_to_wsl(
            Path(r"G:\wSpace\cinebotRL-two-wheel-balance")
        )
        == "/mnt/g/wSpace/cinebotRL-two-wheel-balance"
    )


def test_git_output_uses_wsl_backend_for_windows_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], object, bool, bool, bool]] = []
    outputs = iter(["a" * 40 + "\n", " M tracked.py\n"])

    def fake_run(
        command: list[str],
        *,
        cwd: object,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        calls.append((command, cwd, check, capture_output, text))
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(
        exact_retarget, "PROJECT_ROOT", Path(r"G:\wSpace\cinebotRL-two-wheel-balance")
    )
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert exact_retarget._git_output("rev-parse", "HEAD") == "a" * 40
    assert (
        exact_retarget._git_output(
            "status", "--porcelain", "--untracked-files=no"
        )
        == "M tracked.py"
    )
    executable = str(Path(r"C:\Windows") / "System32" / "wsl.exe")
    common = [executable, "git", "-C", "/mnt/g/wSpace/cinebotRL-two-wheel-balance"]
    assert calls == [
        ([*common, "rev-parse", "HEAD"], None, True, True, True),
        (
            [*common, "status", "--porcelain", "--untracked-files=no"],
            None,
            True,
            True,
            True,
        ),
    ]


def test_gravity_recovery_settings_are_disabled_and_identity_bound_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retarget_exact_source_v1_nonholonomic.py",
            "--reference-package",
            "reference",
            "--integrity-seed-package",
            "seed",
            "--target-urdf",
            "robot.urdf",
            "--output-dir",
            "output",
        ],
    )
    args = exact_retarget.parse_args()
    config = exact_retarget.retarget_cli_config(args)

    assert config["enable_gravity_aware_base_arm_recovery_seed"] is False
    assert config["gravity_aware_base_arm_finite_difference_step"] == 1e-5
    assert config["gravity_aware_base_arm_maximum_step_fraction"] == 0.25
    assert config["enable_semantic_branch_lookback"] is False
    assert config["semantic_branch_lookback_source_intervals"] == 6
    assert config["semantic_branch_beam_width"] == 4
    changed = {**config, "enable_gravity_aware_base_arm_recovery_seed": True}
    assert canonical_json_sha256(changed) != canonical_json_sha256(config)
    branch_changed = {**config, "enable_semantic_branch_lookback": True}
    assert canonical_json_sha256(branch_changed) != canonical_json_sha256(config)


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


def stationary_checkpoint_prefix() -> ExactSourceRetargetPrefix:
    return ExactSourceRetargetPrefix(
        states=np.zeros((3, 9)),
        controls=np.zeros((2, 5)),
        target_positions=np.zeros((3, 3)),
        target_attitudes=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        execution_time_s=np.array([0.0, 0.1, 0.2]),
        position_errors_m=np.zeros(3),
        attitude_errors_deg=np.zeros(3),
        previous_control=np.zeros(8),
        source_anchor_execution_index_prefix=np.array([0, 1, 2]),
        retimed_interval_count=0,
        next_source_interval=3,
    )


def _seed_diagnostics(start_interval: int) -> dict[str, object]:
    return {
        "contract": retarget.GRAVITY_AWARE_BASE_ARM_RECOVERY_CONTRACT,
        "enabled": False,
        "scope": "resumed_suffix_only",
        "scope_start_source_interval": start_interval,
        "generated_count": 0,
        "deduplicated_count": 0,
        "unavailable_count": 0,
        "skipped_existing_feasible_count": 0,
        "selected_count": 0,
    }


def test_semantic_branch_lookback_disabled_calls_single_branch_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel = tuple(range(9))
    calls: list[object] = []

    def fake_single(*arguments: object) -> tuple[int, ...]:
        calls.append(arguments[-1])
        return sentinel

    monkeypatch.setattr(
        retarget, "_retarget_semantic_full_pose_single_branch", fake_single
    )
    checkpoint_path = tmp_path / "disabled.npz"
    args = solver_args(
        enable_semantic_branch_lookback=False,
        exact_source_checkpoint_path=checkpoint_path,
    )

    assert retarget.retarget_semantic_full_pose(*stationary_problem(), args) == sentinel
    assert calls == [args]
    assert args.semantic_branch_lookback_diagnostics == {
        "contract": retarget.SEMANTIC_BRANCH_LOOKBACK_CONTRACT,
        "enabled": False,
    }
    assert not retarget.semantic_branch_journal_path(checkpoint_path).exists()


def test_semantic_branch_journal_atomic_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "main.npz"
    checkpoint.write_bytes(b"main")
    identity = {"bound": True}
    journal = retarget._new_semantic_branch_journal(
        checkpoint_identity=identity,
        original_checkpoint_sha256=retarget.sha256(checkpoint),
        rejection_source_interval=50,
        replay_start_source_interval=44,
        lookback_source_intervals=6,
        beam_width=4,
    )
    path = retarget.semantic_branch_journal_path(checkpoint)

    retarget._atomic_write_semantic_branch_json(path, journal)
    loaded = retarget._load_semantic_branch_journal(
        path,
        checkpoint_identity=identity,
        original_checkpoint_sha256=retarget.sha256(checkpoint),
        rejection_source_interval=50,
        replay_start_source_interval=44,
        lookback_source_intervals=6,
        beam_width=4,
    )

    assert loaded == journal
    assert loaded["valid_for_training"] is False
    assert loaded["training_started"] is False
    assert not list(tmp_path.glob(".main.npz.semantic_branch_journal.json.*.tmp"))


def test_semantic_branch_journal_rejects_identity_hash_and_corruption(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "main.npz"
    checkpoint.write_bytes(b"main")
    identity = {"bound": True}
    path = retarget.semantic_branch_journal_path(checkpoint)
    retarget._atomic_write_semantic_branch_json(
        path,
        retarget._new_semantic_branch_journal(
            checkpoint_identity=identity,
            original_checkpoint_sha256=retarget.sha256(checkpoint),
            rejection_source_interval=50,
            replay_start_source_interval=44,
            lookback_source_intervals=6,
            beam_width=4,
        ),
    )
    expected = {
        "original_checkpoint_sha256": retarget.sha256(checkpoint),
        "rejection_source_interval": 50,
        "replay_start_source_interval": 44,
        "lookback_source_intervals": 6,
        "beam_width": 4,
    }

    with pytest.raises(ValueError, match="identity mismatch"):
        retarget._load_semantic_branch_journal(
            path, checkpoint_identity={"bound": False}, **expected
        )
    with pytest.raises(ValueError, match="original checkpoint hash mismatch"):
        retarget._load_semantic_branch_journal(
            path,
            checkpoint_identity=identity,
            **{**expected, "original_checkpoint_sha256": "0" * 64},
        )

    path.write_text("{truncated", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid semantic branch journal"):
        retarget._load_semantic_branch_journal(
            path, checkpoint_identity=identity, **expected
        )


def test_semantic_branch_lookback_persists_histories_and_bypasses_replay_after_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    problem = stationary_problem()
    original = ExactSourceRetargetPrefix(
        states=np.zeros((3, 9)),
        controls=np.zeros((2, 5)),
        target_positions=np.zeros((3, 3)),
        target_attitudes=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        execution_time_s=np.array([0.0, 0.1, 0.2]),
        position_errors_m=np.zeros(3),
        attitude_errors_deg=np.zeros(3),
        previous_control=np.zeros(8),
        source_anchor_execution_index_prefix=np.array([0, 1, 2]),
        retimed_interval_count=0,
        next_source_interval=3,
    )
    rewound = replace(
        original,
        states=original.states[:2],
        controls=original.controls[:1],
        target_positions=original.target_positions[:2],
        target_attitudes=original.target_attitudes[:2],
        execution_time_s=original.execution_time_s[:2],
        position_errors_m=original.position_errors_m[:2],
        attitude_errors_deg=original.attitude_errors_deg[:2],
        source_anchor_execution_index_prefix=np.array([0, 1]),
        next_source_interval=2,
    )
    saved: dict[Path, ExactSourceRetargetPrefix] = {}
    continuation_prefixes: list[ExactSourceRetargetPrefix] = []
    speculative_variants: list[int] = []
    sentinel = tuple(range(9))

    checkpoint_path = tmp_path / "checkpoint.npz"
    checkpoint_path.write_bytes(b"original-checkpoint")

    def fake_load(path: Path, *unused: object, **keywords: object) -> ExactSourceRetargetPrefix:
        path = Path(path).resolve()
        if path in saved:
            return saved[path]
        if path == checkpoint_path.resolve() and path.read_bytes().startswith(b"variant-"):
            marker = int(path.read_bytes().decode().split("-")[1])
            return next(
                prefix
                for prefix in saved.values()
                if int(prefix.states[-1, 0]) == marker
            )
        return original

    monkeypatch.setattr(retarget, "load_exact_source_checkpoint", fake_load)
    monkeypatch.setattr(
        retarget,
        "truncate_exact_source_prefix_for_semantic_lookback",
        lambda prefix, lookback: rewound,
    )

    def fake_single(*arguments: object) -> tuple[int, ...]:
        branch_args = arguments[-1]
        stop_after = getattr(
            branch_args, "_semantic_branch_stop_after_source_interval", None
        )
        if stop_after is None:
            continuation_prefixes.append(
                getattr(branch_args, "_semantic_branch_prefix_override", original)
            )
            branch_args.semantic_seed_family_diagnostics = _seed_diagnostics(4)
            return sentinel
        variant = branch_args._semantic_branch_rank_variant
        speculative_variants.append(variant)
        assert branch_args.exact_source_checkpoint_path is None
        if variant == 0:
            raise ValueError("greedy history rejected")
        marker = float(variant)
        branch_args._semantic_branch_prefix_result = replace(
            original,
            states=np.vstack((original.states, np.full((1, 9), marker))),
            controls=np.vstack((original.controls, np.zeros((1, 5)))),
            target_positions=np.vstack((original.target_positions, np.zeros((1, 3)))),
            target_attitudes=np.vstack(
                (original.target_attitudes, [[1.0, 0.0, 0.0, 0.0]])
            ),
            execution_time_s=np.append(original.execution_time_s, 0.3),
            position_errors_m=np.append(original.position_errors_m, 0.01),
            attitude_errors_deg=np.append(original.attitude_errors_deg, 0.01),
            source_anchor_execution_index_prefix=np.array([0, 1, 2, 3]),
            next_source_interval=4,
        )
        branch_args.semantic_seed_family_diagnostics = _seed_diagnostics(2)
        return sentinel

    monkeypatch.setattr(
        retarget, "_retarget_semantic_full_pose_single_branch", fake_single
    )
    monkeypatch.setattr(
        retarget,
        "_semantic_branch_prefix_scores",
        lambda prefix, *unused: (float(prefix.states[-1, 0]), -float(prefix.states[-1, 0])),
    )

    def fake_save(*arguments: object, **keywords: object) -> None:
        path = Path(arguments[0]).resolve()
        prefix = arguments[2]
        saved[path] = prefix
        marker = int(prefix.states[-1, 0])
        path.write_bytes(f"variant-{marker}".encode())

    monkeypatch.setattr(retarget, "save_exact_source_checkpoint", fake_save)
    args = solver_args(
        enable_semantic_branch_lookback=True,
        semantic_branch_lookback_source_intervals=1,
        semantic_branch_beam_width=3,
        exact_source_checkpoint_path=checkpoint_path,
        exact_source_checkpoint_identity={"bound": True},
        exact_source_resume_checkpoint=True,
    )

    result = retarget.retarget_semantic_full_pose(*problem, args)

    assert result == sentinel
    assert len(saved) == 2
    assert checkpoint_path.read_bytes() == b"variant-2"
    assert len(continuation_prefixes) == 1
    assert continuation_prefixes[0].states[-1, 0] == 2.0
    assert speculative_variants == [0, 1, 2]
    diagnostics = args.semantic_branch_lookback_diagnostics
    assert diagnostics["selected_rank_variant"] == 2
    assert diagnostics["hard_feasible_distinct_history_count"] == 2
    assert [row["status"] for row in diagnostics["history_diagnostics"]] == [
        "rejected",
        "crossed_prior_rejection",
        "crossed_prior_rejection",
    ]

    result = retarget.retarget_semantic_full_pose(*problem, args)

    assert result == sentinel
    assert speculative_variants == [0, 1, 2]
    assert len(continuation_prefixes) == 2
    assert args.semantic_branch_lookback_diagnostics[
        "resumed_after_persisted_selection"
    ] is True
    journal_path = retarget.semantic_branch_journal_path(checkpoint_path)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["selection"]["status"] == "selected"
    assert journal["selection"]["rank_variant"] == 2
    assert journal["valid_for_training"] is False


def test_semantic_branch_lookback_skips_completed_rejected_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    problem = stationary_problem()
    original = ExactSourceRetargetPrefix(
        states=np.zeros((3, 9)),
        controls=np.zeros((2, 5)),
        target_positions=np.zeros((3, 3)),
        target_attitudes=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        execution_time_s=np.array([0.0, 0.1, 0.2]),
        position_errors_m=np.zeros(3),
        attitude_errors_deg=np.zeros(3),
        previous_control=np.zeros(8),
        source_anchor_execution_index_prefix=np.array([0, 1, 2]),
        retimed_interval_count=0,
        next_source_interval=3,
    )
    rewound = replace(
        original,
        states=original.states[:2],
        controls=original.controls[:1],
        target_positions=original.target_positions[:2],
        target_attitudes=original.target_attitudes[:2],
        execution_time_s=original.execution_time_s[:2],
        position_errors_m=original.position_errors_m[:2],
        attitude_errors_deg=original.attitude_errors_deg[:2],
        source_anchor_execution_index_prefix=np.array([0, 1]),
        next_source_interval=2,
    )
    checkpoint_path = tmp_path / "checkpoint.npz"
    checkpoint_path.write_bytes(b"original")
    saved: dict[Path, ExactSourceRetargetPrefix] = {}
    calls: list[int] = []
    phase = {"value": 1}
    sentinel = tuple(range(9))

    def successful_prefix(marker: int) -> ExactSourceRetargetPrefix:
        return replace(
            original,
            states=np.vstack((original.states, np.full((1, 9), marker))),
            controls=np.vstack((original.controls, np.zeros((1, 5)))),
            target_positions=np.vstack((original.target_positions, np.zeros((1, 3)))),
            target_attitudes=np.vstack(
                (original.target_attitudes, [[1.0, 0.0, 0.0, 0.0]])
            ),
            execution_time_s=np.append(original.execution_time_s, 0.3),
            position_errors_m=np.append(original.position_errors_m, 0.01),
            attitude_errors_deg=np.append(original.attitude_errors_deg, 0.01),
            source_anchor_execution_index_prefix=np.array([0, 1, 2, 3]),
            next_source_interval=4,
        )

    def fake_load(path: Path, *unused: object, **keywords: object) -> ExactSourceRetargetPrefix:
        path = Path(path).resolve()
        if path in saved:
            return saved[path]
        if path == checkpoint_path.resolve() and path.read_bytes() == b"variant-1":
            return successful_prefix(1)
        return original

    def fake_single(*arguments: object) -> tuple[int, ...]:
        branch_args = arguments[-1]
        stop_after = getattr(
            branch_args, "_semantic_branch_stop_after_source_interval", None
        )
        if stop_after is None:
            branch_args.semantic_seed_family_diagnostics = _seed_diagnostics(4)
            return sentinel
        variant = branch_args._semantic_branch_rank_variant
        calls.append(variant)
        if variant == 0:
            raise ValueError("rank zero rejected")
        if variant == 1 and phase["value"] == 1:
            raise RuntimeError("bounded interruption")
        if variant == 2:
            raise ValueError("rank two rejected")
        branch_args._semantic_branch_prefix_result = successful_prefix(variant)
        branch_args.semantic_seed_family_diagnostics = _seed_diagnostics(2)
        return sentinel

    def fake_save(*arguments: object, **keywords: object) -> None:
        path = Path(arguments[0]).resolve()
        prefix = arguments[2]
        saved[path] = prefix
        path.write_bytes(f"variant-{int(prefix.states[-1, 0])}".encode())

    monkeypatch.setattr(retarget, "load_exact_source_checkpoint", fake_load)
    monkeypatch.setattr(
        retarget,
        "truncate_exact_source_prefix_for_semantic_lookback",
        lambda prefix, lookback: rewound,
    )
    monkeypatch.setattr(
        retarget, "_retarget_semantic_full_pose_single_branch", fake_single
    )
    monkeypatch.setattr(
        retarget,
        "_semantic_branch_prefix_scores",
        lambda prefix, *unused: (1.0, 1.0),
    )
    monkeypatch.setattr(retarget, "save_exact_source_checkpoint", fake_save)
    args = solver_args(
        enable_semantic_branch_lookback=True,
        semantic_branch_lookback_source_intervals=1,
        semantic_branch_beam_width=3,
        exact_source_checkpoint_path=checkpoint_path,
        exact_source_checkpoint_identity={"bound": True},
        exact_source_resume_checkpoint=True,
    )

    with pytest.raises(RuntimeError, match="bounded interruption"):
        retarget.retarget_semantic_full_pose(*problem, args)
    journal = json.loads(
        retarget.semantic_branch_journal_path(checkpoint_path).read_text(
            encoding="utf-8"
        )
    )
    assert [row["rank_variant"] for row in journal["histories"]] == [0]
    assert journal["histories"][0]["status"] == "rejected"

    phase["value"] = 2
    assert retarget.retarget_semantic_full_pose(*problem, args) == sentinel
    assert calls == [0, 1, 1, 2]
    assert args.semantic_branch_lookback_diagnostics["selected_rank_variant"] == 1


def test_semantic_branch_pending_selection_recovers_with_valid_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    problem = stationary_problem()
    source_time = problem[0].time_s
    identity = checkpoint_identity(source_time)
    checkpoint_path = tmp_path / "checkpoint.npz"
    save_exact_source_checkpoint(
        checkpoint_path,
        identity,
        stationary_checkpoint_prefix(),
        source_time_s=source_time,
        source_positions_m=problem[0].positions_m,
        source_attitudes_wxyz=problem[0].attitudes_wxyz,
        expected_anchor=problem[1],
    )
    original_sha = retarget.sha256(checkpoint_path)
    args = solver_args(
        enable_semantic_branch_lookback=True,
        semantic_branch_lookback_source_intervals=1,
        semantic_branch_beam_width=2,
        exact_source_checkpoint_path=checkpoint_path,
        exact_source_checkpoint_identity=identity,
        exact_source_resume_checkpoint=True,
    )
    real_copy = retarget._atomic_copy_semantic_branch_checkpoint

    def fail_copy(source: Path, target: Path) -> None:
        raise OSError("simulated selection install interruption")

    monkeypatch.setattr(
        retarget, "_atomic_copy_semantic_branch_checkpoint", fail_copy
    )
    with pytest.raises(OSError, match="selection install interruption"):
        retarget.retarget_semantic_full_pose(*problem, args)

    journal_path = retarget.semantic_branch_journal_path(checkpoint_path)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["selection"]["status"] == "pending"
    assert retarget.sha256(checkpoint_path) == original_sha
    selected_variant_path = retarget.semantic_branch_variant_checkpoint_path(
        checkpoint_path, journal["selection"]["rank_variant"]
    )
    assert selected_variant_path.is_file()
    assert retarget.sha256(selected_variant_path) == journal["selection"][
        "prefix_checkpoint_sha256"
    ]

    monkeypatch.setattr(
        retarget, "_atomic_copy_semantic_branch_checkpoint", real_copy
    )
    result = retarget.retarget_semantic_full_pose(*problem, args)

    assert result[1].shape[0] == 4
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["selection"]["status"] == "selected"
    assert retarget.sha256(checkpoint_path) == retarget.sha256(
        selected_variant_path
    )
    loaded = load_exact_source_checkpoint(
        checkpoint_path,
        identity,
        source_time_s=source_time,
        source_positions_m=problem[0].positions_m,
        source_attitudes_wxyz=problem[0].attitudes_wxyz,
        expected_anchor=problem[1],
    )
    assert loaded.next_source_interval == len(source_time)


def test_semantic_branch_rejects_orphan_checkpoint_without_journal(
    tmp_path: Path,
) -> None:
    problem = stationary_problem()
    source_time = problem[0].time_s
    identity = checkpoint_identity(source_time)
    checkpoint_path = tmp_path / "checkpoint.npz"
    save_exact_source_checkpoint(
        checkpoint_path,
        identity,
        stationary_checkpoint_prefix(),
        source_time_s=source_time,
        source_positions_m=problem[0].positions_m,
        source_attitudes_wxyz=problem[0].attitudes_wxyz,
        expected_anchor=problem[1],
    )
    retarget.semantic_branch_variant_checkpoint_path(
        checkpoint_path, 0
    ).write_bytes(b"orphan")
    args = solver_args(
        enable_semantic_branch_lookback=True,
        semantic_branch_lookback_source_intervals=1,
        semantic_branch_beam_width=2,
        exact_source_checkpoint_path=checkpoint_path,
        exact_source_checkpoint_identity=identity,
        exact_source_resume_checkpoint=True,
    )

    with pytest.raises(ValueError, match="without a bound journal"):
        retarget.retarget_semantic_full_pose(*problem, args)


def test_semantic_branch_lookback_requires_bound_resume() -> None:
    args = solver_args(
        enable_semantic_branch_lookback=True,
        semantic_branch_lookback_source_intervals=1,
        semantic_branch_beam_width=2,
    )
    with pytest.raises(ValueError, match="requires checkpoint resume"):
        retarget.retarget_semantic_full_pose(*stationary_problem(), args)


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


def test_disabled_gravity_recovery_matches_legacy_solver_outputs() -> None:
    problem = stationary_problem()
    legacy = retarget.retarget_semantic_full_pose(*problem, solver_args())
    explicitly_disabled = retarget.retarget_semantic_full_pose(
        *problem,
        solver_args(
            enable_gravity_aware_base_arm_recovery_seed=False,
            gravity_aware_base_arm_finite_difference_step=3e-5,
            gravity_aware_base_arm_maximum_step_fraction=0.75,
        ),
    )

    for expected, actual in zip(legacy[:-1], explicitly_disabled[:-1], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert explicitly_disabled[-1] == legacy[-1]
