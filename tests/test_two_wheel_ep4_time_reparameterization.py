from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.ep4_time_reparameterization import (
    TimeReparameterizationConfig,
    array_sha256,
    derive_ep4_time_warp_package,
    derive_time_reparameterization,
    load_ep4_time_warp_reference,
    verify_ep4_time_warp_package,
)
from rl_platform.tasks.two_wheel_balance.exact_source_checkpoint import (
    _validate_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/two_wheel_balance"))

import retarget_exact_source_v1_nonholonomic as exact_retarget  # noqa: E402


def sample_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.linspace(0.0, 2.0, 11)
    positions = np.zeros((11, 3), dtype=np.float64)
    increments = np.array([0.02, 0.02, 0.02, 0.08, 0.10, 0.10, 0.08, 0.02, 0.02, 0.02])
    positions[1:, 0] = np.cumsum(increments)
    angles = np.linspace(0.0, 0.35, 11)
    attitudes = np.column_stack(
        (np.zeros(11), np.zeros(11), np.sin(angles / 2.0), np.cos(angles / 2.0))
    )
    return time_s, positions, attitudes


def config() -> TimeReparameterizationConfig:
    return TimeReparameterizationConfig(
        episode_index=1,
        translation_speed_cap_mps=0.4,
        angular_speed_cap_radps=0.35,
        diagnostic_transition_start_1based=4,
        diagnostic_transition_end_1based=7,
    )


def test_projection_is_deterministic_fixed_duration_and_strict() -> None:
    time_s, positions, attitudes = sample_trajectory()
    first = derive_time_reparameterization(time_s, positions, attitudes, config())
    second = derive_time_reparameterization(time_s, positions, attitudes, config())
    np.testing.assert_array_equal(first.derived_time_s, second.derived_time_s)
    assert first.derived_time_s[0] == time_s[0]
    assert first.derived_time_s[-1] == time_s[-1]
    assert np.all(np.diff(first.derived_time_s) > 0.0)
    assert np.max(first.derived_translation_speed_mps) <= 0.4 + 1.0e-12
    assert np.max(first.derived_angular_speed_radps) <= 0.35 + 1.0e-12
    assert np.all(first.derived_dt_s[3:7] >= first.source_dt_s[3:7])
    assert first.slowdown_regions
    assert first.recovery_regions


def test_projection_fails_closed_when_duration_cannot_meet_bounds() -> None:
    time_s, positions, attitudes = sample_trajectory()
    with pytest.raises(ValueError, match="shorter than the sum"):
        derive_time_reparameterization(
            time_s,
            positions,
            attitudes,
            TimeReparameterizationConfig(
                translation_speed_cap_mps=0.05,
                angular_speed_cap_radps=0.05,
                diagnostic_transition_start_1based=4,
                diagnostic_transition_end_1based=7,
            ),
        )


def test_projection_does_not_create_a_fast_micro_interval() -> None:
    time_s = np.array([0.0, 0.02, 0.04, 0.0412])
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.006, 0.0, 0.0], [0.012, 0.0, 0.0], [0.012, 0.0, 0.0]]
    )
    attitudes = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
    result = derive_time_reparameterization(
        time_s,
        positions,
        attitudes,
        TimeReparameterizationConfig(
            episode_index=1,
            diagnostic_transition_start_1based=1,
            diagnostic_transition_end_1based=3,
        ),
    )
    assert np.min(result.derived_dt_s) >= 1.0e-3 - 1.0e-12


def test_projection_applies_a_localized_translation_cap_with_fixed_duration() -> None:
    time_s, positions, attitudes = sample_trajectory()
    result = derive_time_reparameterization(
        time_s,
        positions,
        attitudes,
        TimeReparameterizationConfig(
            episode_index=1,
            translation_speed_cap_mps=0.4,
            localized_transition_start_1based=4,
            localized_transition_end_1based=7,
            localized_translation_speed_cap_mps=0.27,
            diagnostic_transition_start_1based=4,
            diagnostic_transition_end_1based=7,
        ),
    )
    assert result.derived_time_s[-1] == time_s[-1]
    assert np.max(result.derived_translation_speed_mps[3:7]) <= 0.27 + 1.0e-12
    assert np.max(result.derived_translation_speed_mps[:3]) <= 0.4 + 1.0e-12
    assert np.max(result.derived_translation_speed_mps[7:]) <= 0.4 + 1.0e-12
    assert np.all(result.translation_speed_cap_mps[3:7] == 0.27)


def test_projection_rejects_an_incomplete_localized_cap_contract() -> None:
    time_s, positions, attitudes = sample_trajectory()
    with pytest.raises(ValueError, match="requires start, end, and speed"):
        derive_time_reparameterization(
            time_s,
            positions,
            attitudes,
            TimeReparameterizationConfig(
                episode_index=1,
                localized_transition_start_1based=4,
                diagnostic_transition_start_1based=4,
                diagnostic_transition_end_1based=7,
            ),
        )


def test_proportional_lower_bounds_distributes_slack_without_migration() -> None:
    time_s, positions, attitudes = sample_trajectory()
    result = derive_time_reparameterization(
        time_s,
        positions,
        attitudes,
        TimeReparameterizationConfig(
            episode_index=1,
            time_allocation_strategy="proportional_lower_bounds",
            diagnostic_transition_start_1based=4,
            diagnostic_transition_end_1based=7,
        ),
    )
    scale = result.derived_dt_s / result.lower_dt_s
    assert result.derived_time_s[-1] == time_s[-1]
    assert np.max(scale) - np.min(scale) <= 1.0e-12
    assert np.min(result.derived_dt_s) >= 1.0e-3
    assert np.max(result.derived_translation_speed_mps) < 0.4
    assert np.max(result.derived_angular_speed_radps) < 0.35


def test_projection_rejects_unknown_time_allocation_strategy() -> None:
    time_s, positions, attitudes = sample_trajectory()
    with pytest.raises(ValueError, match="unsupported time allocation strategy"):
        derive_time_reparameterization(
            time_s,
            positions,
            attitudes,
            TimeReparameterizationConfig(
                episode_index=1,
                time_allocation_strategy="unknown",
                diagnostic_transition_start_1based=4,
                diagnostic_transition_end_1based=7,
            ),
        )


def write_reference_package(root: Path) -> str:
    time_s, positions, attitudes = sample_trajectory()
    episode_dir = root / "episode_0001"
    episode_dir.mkdir(parents=True)
    payload = {
        "duration_sec": float(time_s[-1]),
        "poses": [
            {
                "time": float(time_s[index]),
                "position": positions[index].tolist(),
                "orientation": attitudes[index].tolist(),
            }
            for index in range(len(time_s))
        ],
    }
    source_path = episode_dir / "source.json"
    source_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
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
        "source_pose_count_total": len(time_s),
        "integrity_passed": True,
        "quality_qualified_teacher": False,
        "valid_for_training": False,
        "items": [
            {
                "episode_index": 1,
                "bundled_source_json": "episode_0001/source.json",
                "source_json_sha256": source_sha,
                "source_pose_count": len(time_s),
                "source_duration_s": float(time_s[-1]),
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


def write_raw_seed_package(root: Path, reference_dir: Path) -> tuple[str, str]:
    document = json.loads((reference_dir / "episode_0001/source.json").read_text())
    time_s = np.asarray([pose["time"] for pose in document["poses"]], dtype=np.float64)
    positions = np.asarray([pose["position"] for pose in document["poses"]], dtype=np.float64)
    attitudes_xyzw = np.asarray(
        [pose["orientation"] for pose in document["poses"]], dtype=np.float64
    )
    count = len(time_s)
    base_arm_q = np.zeros((count, 6), dtype=np.float32)
    base_arm_q[:, 0] = np.linspace(0.0, 0.4, count)
    base_arm_q[:, 1] = np.linspace(0.0, 0.1, count)
    base_arm_q[:, 2] = np.linspace(0.0, 0.2, count)
    base_arm_q[:, 3:] = np.linspace(0.0, 0.3, count)[:, None]
    actions = np.zeros((count - 1, 6), dtype=np.float32)
    actions[:, :3] = np.linspace(-0.5, 0.5, count - 1)[:, None]
    episode_dir = root / "episode_0001"
    episode_dir.mkdir(parents=True)
    seed_path = episode_dir / "episode_0001_exact_source_integrity_v2.npz"
    np.savez_compressed(
        seed_path,
        schema=np.asarray("cinebotrl_gik_monorepo_ee1_split_teacher_v2"),
        valid_for_training=np.bool_(False),
        valid_for_candidate_training=np.bool_(False),
        teacher_quality_passed=np.bool_(False),
        trajectory_integrity_contract=np.asarray("exact_source_v1"),
        trajectory_integrity_passed=np.bool_(True),
        episode_index=np.int32(1),
        base_arm_actions=actions,
        q_current_base_arm_6=base_arm_q[:-1],
        q_next_base_arm_6=base_arm_q[1:],
        desired_time_full_s=time_s,
        desired_position_full_m=positions,
        desired_attitude_full_world_dfr_quat_wxyz=attitudes_xyzw[:, [3, 0, 1, 2]],
        time_s=time_s[1:],
        dt_s=np.diff(time_s).astype(np.float32),
        source_pose_count=np.int32(count),
        reference_pose_count=np.int32(count),
        state_count=np.int32(count),
        action_count=np.int32(count - 1),
        max_timestamp_error_s=np.float64(0.0),
        max_linear_velocity=np.float32(1.5),
        max_angular_velocity=np.float32(2.0),
        max_abs_base_action_unclipped=np.float32(0.0),
        source_json_sha256=np.asarray(
            hashlib.sha256((reference_dir / "episode_0001/source.json").read_bytes()).hexdigest()
        ),
    )
    seed_sha = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "gik_exact_source_teacher_integrity_canaries_v1",
        "trajectory_integrity_contract": "exact_source_v1",
        "valid_for_training": False,
        "case_count": 1,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest(), seed_sha


def build_package(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    reference_dir = tmp_path / "reference"
    reference_digest = write_reference_package(reference_dir)
    raw_seed_dir = tmp_path / "raw_seed"
    raw_seed_manifest_digest, raw_seed_digest = write_raw_seed_package(
        raw_seed_dir, reference_dir
    )
    output_dir = tmp_path / "derived"
    manifest = derive_ep4_time_warp_package(
        reference_dir,
        raw_seed_dir,
        output_dir,
        config=config(),
        expected_manifest_sha256=reference_digest,
        expected_episodes=1,
        expected_raw_seed_manifest_sha256=raw_seed_manifest_digest,
        expected_raw_seed_sha256=raw_seed_digest,
    )
    return reference_dir, raw_seed_dir, output_dir, manifest


def test_package_preserves_pose_and_seed_state_bytes_and_binds_provenance(
    tmp_path: Path,
) -> None:
    reference_dir, raw_seed_dir, output_dir, manifest = build_package(tmp_path)
    verified = verify_ep4_time_warp_package(
        reference_dir, raw_seed_dir, output_dir, expected_episodes=1
    )
    assert verified == manifest
    assert manifest["valid_for_training"] is False
    assert manifest["constraints"]["positions_byte_identical"] is True
    assert manifest["constraints"]["orientations_byte_identical"] is True
    assert manifest["constraints"]["solver_or_admission_gates_modified"] is False
    seed_rates = manifest["metrics"]["integrity_seed_rate_diagnostics"]
    assert seed_rates["seed_state_rate_bounds_used_in_projection"] is False
    assert seed_rates["counterfactual_all_seed_rate_bounds"]["rejected_for_this_derivation"]

    raw = json.loads((reference_dir / "episode_0001/source.json").read_text())
    derived = json.loads((output_dir / "source.json").read_text())
    raw_positions = np.asarray([pose["position"] for pose in raw["poses"]], dtype=np.float64)
    new_positions = np.asarray([pose["position"] for pose in derived["poses"]], dtype=np.float64)
    raw_attitudes = np.asarray([pose["orientation"] for pose in raw["poses"]], dtype=np.float64)
    new_attitudes = np.asarray([pose["orientation"] for pose in derived["poses"]], dtype=np.float64)
    assert raw_positions.tobytes() == new_positions.tobytes()
    assert raw_attitudes.tobytes() == new_attitudes.tobytes()
    assert array_sha256(raw_positions) == array_sha256(new_positions)
    assert array_sha256(raw_attitudes) == array_sha256(new_attitudes)
    assert derived["poses"][0]["time"] == raw["poses"][0]["time"]
    assert derived["poses"][-1]["time"] == raw["poses"][-1]["time"]


def test_derived_reference_and_seed_load_together_but_raw_seed_is_rejected(
    tmp_path: Path,
) -> None:
    reference_dir, raw_seed_dir, output_dir, _ = build_package(tmp_path)
    derived_reference = load_ep4_time_warp_reference(
        reference_dir,
        raw_seed_dir,
        output_dir,
        expected_episodes=1,
    )
    teacher, metadata = exact_retarget.load_integrity_seed(
        output_dir / "paired_integrity_seed", derived_reference
    )
    np.testing.assert_array_equal(teacher.time_s, derived_reference.time_s)
    assert metadata["seed_valid_for_training"] is False
    with pytest.raises(ValueError, match="timestamps replace source"):
        exact_retarget.load_integrity_seed(raw_seed_dir, derived_reference)


def test_package_rejects_mutated_source_and_existing_output(tmp_path: Path) -> None:
    reference_dir, raw_seed_dir, output_dir, _ = build_package(tmp_path)
    reference_digest = hashlib.sha256((reference_dir / "manifest.json").read_bytes()).hexdigest()
    raw_manifest_digest = hashlib.sha256((raw_seed_dir / "manifest.json").read_bytes()).hexdigest()
    raw_seed_path = next((raw_seed_dir / "episode_0001").glob("*.npz"))
    raw_seed_digest = hashlib.sha256(raw_seed_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already exists"):
        derive_ep4_time_warp_package(
            reference_dir,
            raw_seed_dir,
            output_dir,
            config=config(),
            expected_manifest_sha256=reference_digest,
            expected_episodes=1,
            expected_raw_seed_manifest_sha256=raw_manifest_digest,
            expected_raw_seed_sha256=raw_seed_digest,
        )

    source_path = reference_dir / "episode_0001/source.json"
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_ep4_time_warp_package(
            reference_dir, raw_seed_dir, output_dir, expected_episodes=1
        )


def test_runner_time_warp_mode_preflights_without_entering_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_dir, raw_seed_dir, output_dir, manifest = build_package(tmp_path)
    real_verify = verify_ep4_time_warp_package
    real_load = load_ep4_time_warp_reference
    monkeypatch.setattr(
        exact_retarget,
        "verify_ep4_time_warp_package",
        lambda reference, raw_seed, derived: real_verify(
            reference, raw_seed, derived, expected_episodes=1
        ),
    )
    monkeypatch.setattr(
        exact_retarget,
        "load_ep4_time_warp_reference",
        lambda reference, raw_seed, derived: replace(
            real_load(reference, raw_seed, derived, expected_episodes=1),
            episode_index=4,
        ),
    )
    args = SimpleNamespace(
        cases="4",
        reference_package=reference_dir,
        integrity_seed_package=output_dir / "paired_integrity_seed",
        ep4_time_warp_package=output_dir,
        ep4_time_warp_raw_integrity_seed_package=raw_seed_dir,
    )
    references, cases = exact_retarget.load_retarget_inputs(args)
    assert cases == [4]
    assert references[4].source_json == (output_dir / "source.json").resolve()
    assert args.verified_ep4_time_warp_identity["time_warp_sha256"] == manifest[
        "time_warp_sha256"
    ]
    assert args.verified_ep4_time_warp_identity["valid_for_training"] is False


def test_runner_time_warp_mode_rejects_raw_or_mismatched_seed_package(
    tmp_path: Path,
) -> None:
    reference_dir, raw_seed_dir, output_dir, _ = build_package(tmp_path)
    base = {
        "cases": "4",
        "reference_package": reference_dir,
        "ep4_time_warp_package": output_dir,
        "ep4_time_warp_raw_integrity_seed_package": raw_seed_dir,
    }
    with pytest.raises(ValueError, match="must equal the verified ep4 paired"):
        exact_retarget.load_retarget_inputs(
            SimpleNamespace(**base, integrity_seed_package=raw_seed_dir)
        )
    with pytest.raises(ValueError, match="must equal the verified ep4 paired"):
        exact_retarget.load_retarget_inputs(
            SimpleNamespace(
                **{**base, "ep4_time_warp_package": tmp_path / "other"},
                integrity_seed_package=output_dir / "paired_integrity_seed",
            )
        )


def test_runner_default_cli_keeps_time_warp_disabled_and_identity_bound(
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
    cli_config = exact_retarget.retarget_cli_config(args)
    assert cli_config["ep4_time_warp_package"] is None
    assert cli_config["ep4_time_warp_raw_integrity_seed_package"] is None
    assert Path(
        "src/rl_platform/tasks/two_wheel_balance/ep4_time_reparameterization.py"
    ) in exact_retarget.CHECKPOINT_CODE_CONTRACT_PATHS


def test_time_warp_checkpoint_identity_uses_existing_fail_closed_schema(
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
            "derived/paired_integrity_seed",
            "--ep4-time-warp-package",
            "derived",
            "--ep4-time-warp-raw-integrity-seed-package",
            "raw-seed",
            "--target-urdf",
            "robot.urdf",
            "--output-dir",
            "output",
            "--cases",
            "4",
        ],
    )
    args = exact_retarget.parse_args()
    args.verified_ep4_time_warp_identity = {"time_warp_sha256": "d" * 64}
    monkeypatch.setattr(
        exact_retarget,
        "checkpoint_code_contract",
        lambda: ("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(exact_retarget, "sha256", lambda _path: "c" * 64)
    reference = SimpleNamespace(
        episode_index=4,
        manifest_sha256="e" * 64,
        source_json_sha256="f" * 64,
        time_s=np.asarray([0.0, 0.1], dtype=np.float64),
    )

    identity = exact_retarget.build_checkpoint_identity(
        reference,
        args,
        "1" * 64,
    )

    assert _validate_identity(identity) == identity
    assert "ep4_time_warp" not in identity
