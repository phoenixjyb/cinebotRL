#!/usr/bin/env python3
"""Retarget immutable exact-source references using integrity-only solver seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from retarget_corrected_teacher_v3_nonholonomic import (  # noqa: E402
    SemanticReference,
    retarget_case,
    sha256,
)
from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    EXPECTED_ACTION_ORDER,
    SparseTeacher,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (  # noqa: E402
    UrdfPhysicalCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.exact_source_reference import (  # noqa: E402
    ExactSourceReference,
    discover_exact_source_references,
    source_anchor_execution_indices,
)
from rl_platform.tasks.two_wheel_balance.exact_source_checkpoint import (  # noqa: E402
    canonical_json_sha256,
    source_time_sha256,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (  # noqa: E402
    UrdfPositionKinematics,
)


CANDIDATE_SCHEMA = "cinebotrl_two_wheel_exact_source_retarget_v1"
UPSTREAM_SEED_MANIFEST_SHA256_BY_EPISODE = {
    1: "8d10e4c2093c44cc2e5cf9bc1b12b760da2375e24c083f47bf31b954cdc9f272",
    77: "2dc31d86325155fafb0dc3afe8870f9ae32ea3c58d5ed7d2671f43aa2d4d7404",
}
CHECKPOINT_CODE_CONTRACT_PATHS = (
    Path("scripts/two_wheel_balance/retarget_exact_source_v1_nonholonomic.py"),
    Path("scripts/two_wheel_balance/retarget_corrected_teacher_v3_nonholonomic.py"),
    Path("src/rl_platform/tasks/two_wheel_balance/exact_source_checkpoint.py"),
)
RETARGET_CLI_CONFIG_FIELDS = (
    "reference_package",
    "integrity_seed_package",
    "target_urdf",
    "cases",
    "acquisition_dt_s",
    "minimum_acquisition_duration_s",
    "maximum_acquisition_position_rate_mps",
    "maximum_acquisition_attitude_rate_radps",
    "maximum_linear_velocity",
    "maximum_yaw_rate",
    "maximum_arm_rate",
    "maximum_gimbal_rate",
    "maximum_acquisition_linear_velocity",
    "maximum_acquisition_yaw_rate",
    "maximum_acquisition_arm_rate",
    "maximum_acquisition_gimbal_rate",
    "maximum_arm_gravity_effort_nm",
    "gravity_effort_tolerance_nm",
    "maximum_equilibrium_pitch_deg",
    "wheel_axle_height_m",
    "camera_solve_root_model",
    "minimum_anchor_gimbal_limit_margin_ratio",
    "minimum_semantic_gimbal_limit_margin_ratio",
    "minimum_semantic_gimbal_reserve_margin_ratio",
    "maximum_semantic_gimbal_reserve_search_scale",
    "semantic_gimbal_center_regularization",
    "enable_gravity_aware_base_arm_recovery_seed",
    "gravity_aware_base_arm_finite_difference_step",
    "gravity_aware_base_arm_maximum_step_fraction",
    "position_scale_m",
    "control_regularization",
    "maximum_position_p95_m",
    "maximum_position_error_m",
    "maximum_ik_error_deg",
    "maximum_gimbal_interpolation_error_deg",
    "rebuild_com_safe_seed_prior",
    "checkpoint_cadence_source_intervals",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--integrity-seed-package", type=Path, required=True)
    parser.add_argument("--target-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="1,4,7")
    parser.add_argument("--acquisition-dt-s", type=float, default=0.1)
    parser.add_argument("--minimum-acquisition-duration-s", type=float, default=3.0)
    parser.add_argument("--maximum-acquisition-position-rate-mps", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-attitude-rate-radps", type=float, default=0.35)
    parser.add_argument("--maximum-linear-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-yaw-rate", type=float, default=0.4)
    parser.add_argument("--maximum-arm-rate", type=float, default=0.5)
    parser.add_argument("--maximum-gimbal-rate", type=float, default=0.25)
    parser.add_argument("--maximum-acquisition-linear-velocity", type=float, default=0.15)
    parser.add_argument("--maximum-acquisition-yaw-rate", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-arm-rate", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-gimbal-rate", type=float, default=0.2)
    parser.add_argument("--maximum-arm-gravity-effort-nm", type=float, default=29.5)
    parser.add_argument("--gravity-effort-tolerance-nm", type=float, default=0.01)
    parser.add_argument("--maximum-equilibrium-pitch-deg", type=float, default=10.0)
    parser.add_argument("--wheel-axle-height-m", type=float, default=0.1016)
    parser.add_argument(
        "--camera-solve-root-model", choices=("balanced", "upright"), default="balanced"
    )
    parser.add_argument("--minimum-anchor-gimbal-limit-margin-ratio", type=float, default=0.10)
    parser.add_argument("--minimum-semantic-gimbal-limit-margin-ratio", type=float, default=0.005)
    parser.add_argument(
        "--minimum-semantic-gimbal-reserve-margin-ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--maximum-semantic-gimbal-reserve-search-scale",
        type=int,
        default=24,
    )
    parser.add_argument("--semantic-gimbal-center-regularization", type=float, default=0.10)
    parser.add_argument(
        "--enable-gravity-aware-base-arm-recovery-seed",
        action="store_true",
        help="Append one bounded gravity-descent EE-nullspace seed after existing seeds.",
    )
    parser.add_argument(
        "--gravity-aware-base-arm-finite-difference-step",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--gravity-aware-base-arm-maximum-step-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument("--position-scale-m", type=float, default=0.01)
    parser.add_argument("--control-regularization", type=float, default=0.01)
    parser.add_argument("--maximum-position-p95-m", type=float, default=0.10)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.20)
    parser.add_argument("--maximum-ik-error-deg", type=float, default=0.1)
    parser.add_argument("--maximum-gimbal-interpolation-error-deg", type=float, default=0.25)
    parser.add_argument("--rebuild-com-safe-seed-prior", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Atomic exact-source interval-prefix checkpoint (single case only).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        action="store_true",
        help="Resume from --checkpoint-path after validating every bound identity.",
    )
    parser.add_argument(
        "--checkpoint-cadence-source-intervals",
        type=int,
        default=10,
        help="Persist each N completed source intervals and always at completion.",
    )
    return parser.parse_args()


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_parent_directory(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_parent_directory(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_parent_directory(path: Path) -> None:
    if os.name != "posix":
        return
    directory_fd = os.open(path.resolve().parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def finalize_case_artifacts(
    output_path: Path,
    result_path: Path,
    arrays: dict[str, np.ndarray],
    summary: dict[str, object],
    checkpoint_path: Path | None,
) -> None:
    _atomic_save_npz(output_path, arrays)
    summary["candidate_sha256"] = sha256(output_path)
    _atomic_write_json(result_path, summary)
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint_path.unlink()
        _fsync_parent_directory(checkpoint_path)


def _json_cli_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported CLI configuration value: {type(value).__name__}")


def retarget_cli_config(args: argparse.Namespace) -> dict[str, object]:
    missing = [field for field in RETARGET_CLI_CONFIG_FIELDS if not hasattr(args, field)]
    if missing:
        raise ValueError(f"checkpoint CLI configuration fields missing: {missing}")
    return {
        field: _json_cli_value(getattr(args, field))
        for field in RETARGET_CLI_CONFIG_FIELDS
    }


def _windows_project_root_to_wsl(path: Path) -> str | None:
    value = str(path).replace("\\", "/")
    if len(value) < 3 or value[1:3] != ":/":
        value = str(path.resolve()).replace("\\", "/")
    if len(value) < 3 or value[1:3] != ":/" or not value[0].isalpha():
        return None
    return f"/mnt/{value[0].lower()}/{value[3:]}"


def _git_output(*arguments: str) -> str:
    wsl_root = _windows_project_root_to_wsl(PROJECT_ROOT)
    if wsl_root is None:
        command = ["git", *arguments]
        cwd = PROJECT_ROOT
    else:
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        command = [
            str(system_root / "System32" / "wsl.exe"),
            "git",
            "-C",
            wsl_root,
            *arguments,
        ]
        cwd = None
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkpoint_code_contract() -> tuple[str, str]:
    git_commit = _git_output("rev-parse", "HEAD")
    dirty = _git_output("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("refusing exact-source checkpoint from a dirty code contract")
    files = {
        str(path): sha256(PROJECT_ROOT / path)
        for path in CHECKPOINT_CODE_CONTRACT_PATHS
    }
    return git_commit, canonical_json_sha256(files)


def build_checkpoint_identity(
    reference: ExactSourceReference,
    args: argparse.Namespace,
    seed_sha256: str,
) -> dict[str, object]:
    git_commit, code_contract_sha256 = checkpoint_code_contract()
    config = retarget_cli_config(args)
    return {
        "git_commit": git_commit,
        "code_contract_sha256": code_contract_sha256,
        "case": reference.episode_index,
        "retarget_cli_config": config,
        "retarget_cli_config_sha256": canonical_json_sha256(config),
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_episode_sha256": reference.source_json_sha256,
        "integrity_seed_sha256": seed_sha256,
        "target_urdf_sha256": sha256(args.target_urdf.resolve()),
        "source_pose_count": len(reference.time_s),
        "source_time_sha256": source_time_sha256(reference.time_s),
    }


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return value.reshape(-1)[0].item()


def load_integrity_seed(
    seed_package: Path,
    reference: ExactSourceReference,
) -> tuple[SparseTeacher, dict[str, object]]:
    manifest_path = seed_package / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") == "gik9dof_exact_source_upstream_teacher_seed_manifest_v1":
            return load_upstream_retarget_seed(seed_package, reference, manifest)
    episode_dir = seed_package / f"episode_{reference.episode_index:04d}"
    paths = sorted(episode_dir.glob("*.npz"))
    if len(paths) != 1:
        raise ValueError(f"expected one integrity seed for episode {reference.episode_index}")
    path = paths[0]
    with np.load(path, allow_pickle=False) as data:
        if _scalar(data, "trajectory_integrity_contract") != "exact_source_v1":
            raise ValueError(f"seed lacks exact_source_v1: {path}")
        if not bool(_scalar(data, "trajectory_integrity_passed")):
            raise ValueError(f"seed integrity failed: {path}")
        if bool(_scalar(data, "valid_for_training")):
            raise ValueError(f"integrity-only seed unexpectedly training-valid: {path}")
        if bool(_scalar(data, "teacher_quality_passed")):
            raise ValueError(f"integrity-only seed unexpectedly quality-qualified: {path}")
        current = np.asarray(data["q_current_base_arm_6"], dtype=np.float64)
        next_q = np.asarray(data["q_next_base_arm_6"], dtype=np.float64)
        seed_time = np.asarray(data["desired_time_full_s"], dtype=np.float64)
        seed_position = np.asarray(data["desired_position_full_m"], dtype=np.float64)
        seed_attitude = np.asarray(
            data["desired_attitude_full_world_dfr_quat_wxyz"], dtype=np.float64
        )
        source_pose_count = int(_scalar(data, "source_pose_count"))
        state_count = int(_scalar(data, "state_count"))
        action_count = int(_scalar(data, "action_count"))
    count = len(reference.time_s)
    if current.shape != (count - 1, 6) or next_q.shape != current.shape:
        raise ValueError(f"seed state shape mismatch for episode {reference.episode_index}")
    if source_pose_count != count or state_count != count or action_count != count - 1:
        raise ValueError(f"seed N/N-1 contract mismatch for episode {reference.episode_index}")
    if not np.array_equal(seed_time, reference.time_s):
        raise ValueError(f"seed timestamps replace source for episode {reference.episode_index}")
    if not np.allclose(seed_position, reference.positions_m, atol=1e-12, rtol=0.0):
        raise ValueError(f"seed positions replace source for episode {reference.episode_index}")
    attitude_dots = np.abs(np.sum(seed_attitude * reference.attitudes_wxyz, axis=1))
    if not bool(np.all(attitude_dots >= 1.0 - 1e-10)):
        raise ValueError(f"seed attitudes replace source for episode {reference.episode_index}")
    if not np.allclose(current[1:], next_q[:-1], atol=2e-6, rtol=0.0):
        raise ValueError(f"seed state transitions are discontinuous for episode {reference.episode_index}")
    base_arm_q = np.vstack((current, next_q[-1]))
    teacher = SparseTeacher(
        case=reference.episode_index,
        path=path,
        base_arm_q=base_arm_q,
        time_s=reference.time_s,
        dfr_attitudes_wxyz=reference.attitudes_wxyz[1:],
        desired_positions_m=reference.positions_m,
        desired_attitudes_wxyz=reference.attitudes_wxyz,
        action_order=EXPECTED_ACTION_ORDER,
    )
    return teacher, {
        "seed_path": str(path.resolve()),
        "seed_sha256": sha256(path),
        "seed_contract": "integrity_only_free_gik_not_teacher_labels",
        "seed_quality_qualified": False,
        "seed_valid_for_training": False,
    }


def load_upstream_retarget_seed(
    seed_package: Path,
    reference: ExactSourceReference,
    manifest: dict[str, object],
) -> tuple[SparseTeacher, dict[str, object]]:
    """Load a quality-checked holonomic state seed, never policy actions."""

    manifest_path = seed_package / "manifest.json"
    manifest_sha = sha256(manifest_path)
    expected_manifest_sha = UPSTREAM_SEED_MANIFEST_SHA256_BY_EPISODE.get(
        reference.episode_index
    )
    if manifest_sha != expected_manifest_sha:
        raise ValueError(f"unexpected upstream seed manifest SHA-256: {manifest_sha}")
    episode = reference.episode_index
    if int(manifest.get("episode_index", -1)) != episode:
        raise ValueError(f"upstream seed episode mismatch for {episode}")
    required_manifest = {
        "trajectory_integrity_contract": "exact_source_v1",
        "trajectory_integrity_passed": True,
        "upstream_teacher_quality_passed": True,
        "valid_for_two_wheel_retarget_input": True,
        "two_wheel_dynamic_quality_passed": False,
        "valid_for_training": False,
        "source_package_manifest_sha256": reference.manifest_sha256,
        "source_json_sha256": reference.source_json_sha256,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"upstream seed manifest {key} mismatch for {episode}")
    output_path = seed_package / str(manifest.get("output_npz", ""))
    if not output_path.is_file():
        raise ValueError(f"missing upstream seed NPZ for {episode}")
    if sha256(output_path) != manifest.get("output_npz_sha256"):
        raise ValueError(f"upstream seed NPZ checksum mismatch for {episode}")
    bundled_source = seed_package / str(manifest.get("bundled_source_json", ""))
    if sha256(bundled_source) != reference.source_json_sha256:
        raise ValueError(f"upstream seed source checksum mismatch for {episode}")

    with np.load(output_path, allow_pickle=False) as data:
        scalar_expectations = {
            "schema": "gik9dof_exact_source_upstream_teacher_seed_v1",
            "trajectory_integrity_contract": "exact_source_v1",
            "trajectory_integrity_passed": True,
            "upstream_teacher_quality_passed": True,
            "valid_for_two_wheel_retarget_input": True,
            "two_wheel_dynamic_quality_passed": False,
            "valid_for_training": False,
            "source_package_manifest_sha256": reference.manifest_sha256,
            "source_json_sha256": reference.source_json_sha256,
            "action_provenance": "no_policy_actions_in_this_package",
            "seed_plant_contract": "holonomic_base_arm_retarget_seed",
        }
        for key, expected in scalar_expectations.items():
            if _scalar(data, key) != expected:
                raise ValueError(f"upstream seed {key} mismatch for {episode}")
        source_time = np.asarray(data["source_time_s"], dtype=np.float64)
        source_position = np.asarray(data["source_position_m"], dtype=np.float64)
        source_attitude_xyzw = np.asarray(
            data["source_attitude_world_dfr_quat_xyzw"], dtype=np.float64
        )
        seed_time = np.asarray(data["seed_time_s"], dtype=np.float64)
        seed_q = np.asarray(data["seed_q_base_arm_6"], dtype=np.float64)
        mapping = np.asarray(data["source_anchor_to_seed_state_index"], dtype=np.int64)
        state_count = int(_scalar(data, "seed_state_count"))
        transition_count = int(_scalar(data, "seed_transition_count"))
        initialization_count = int(_scalar(data, "initialization_state_count"))
    count = len(reference.time_s)
    if not np.array_equal(source_time, reference.time_s):
        raise ValueError(f"upstream seed timestamps replace source for {episode}")
    if not np.allclose(source_position, reference.positions_m, atol=1e-12, rtol=0.0):
        raise ValueError(f"upstream seed positions replace source for {episode}")
    if not np.allclose(source_attitude_xyzw, reference.attitudes_xyzw, atol=1e-12, rtol=0.0):
        raise ValueError(f"upstream seed attitudes replace source for {episode}")
    if seed_q.shape != (count, 6) or state_count != count or transition_count != count - 1:
        raise ValueError(f"upstream seed N/N-1 contract mismatch for {episode}")
    if not np.array_equal(seed_time, reference.time_s):
        raise ValueError(f"upstream seed execution timestamps changed for {episode}")
    if not np.array_equal(mapping, np.arange(count)):
        raise ValueError(f"upstream seed anchor mapping is not identity for {episode}")
    if initialization_count != 0:
        raise ValueError(f"upstream seed initialization leaked for {episode}")
    teacher = SparseTeacher(
        case=episode,
        path=output_path,
        base_arm_q=seed_q,
        time_s=reference.time_s,
        dfr_attitudes_wxyz=reference.attitudes_wxyz[1:],
        desired_positions_m=reference.positions_m,
        desired_attitudes_wxyz=reference.attitudes_wxyz,
        action_order=EXPECTED_ACTION_ORDER,
    )
    return teacher, {
        "seed_path": str(output_path.resolve()),
        "seed_sha256": sha256(output_path),
        "seed_manifest_sha256": manifest_sha,
        "seed_contract": "upstream_holonomic_base_arm_seed_no_policy_actions",
        "seed_quality_qualified": True,
        "seed_valid_for_training": False,
        "seed_two_wheel_dynamic_quality_passed": False,
    }


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_semantic_reference(
    reference: ExactSourceReference,
    seed: SparseTeacher,
    kinematics: UrdfPositionKinematics,
) -> tuple[SemanticReference, float]:
    seed_fk = np.asarray([kinematics.position(state) for state in seed.base_arm_q])
    seed_fk_error = float(
        np.max(np.linalg.norm(seed_fk - reference.positions_m, axis=1))
    )
    return (
        SemanticReference(
            case=reference.episode_index,
            source_mat=reference.source_json,
            time_s=reference.time_s,
            positions_m=reference.positions_m,
            attitudes_wxyz=reference.attitudes_wxyz,
            source_fk_max_error_m=0.0,
            package_position_max_error_m=0.0,
            package_q_max_error_rad=0.0,
            package_time_max_error_s=0.0,
            package_attitude_max_error_deg=0.0,
        ),
        seed_fk_error,
    )


def install_exact_source_anchor_zero(
    arrays: dict[str, np.ndarray],
    reference: ExactSourceReference,
) -> int:
    """End initialization at immutable source anchor zero."""

    semantic_start = int(arrays["semantic_start_index"])
    arrays["target_position_world_m"][semantic_start] = reference.positions_m[0]
    arrays["target_attitude_world_dfr_quat_wxyz"][semantic_start] = (
        reference.attitudes_wxyz[0]
    )
    return semantic_start


def process_case(
    reference: ExactSourceReference,
    args: argparse.Namespace,
) -> dict[str, object]:
    teacher, seed_diagnostics = load_integrity_seed(
        args.integrity_seed_package.resolve(), reference
    )
    if args.checkpoint_path is not None:
        args.exact_source_checkpoint_path = args.checkpoint_path.resolve()
        args.exact_source_resume_checkpoint = args.resume_checkpoint
        args.exact_source_checkpoint_identity = build_checkpoint_identity(
            reference, args, str(seed_diagnostics["seed_sha256"])
        )
    position_kinematics = UrdfPositionKinematics(args.target_urdf.resolve())
    camera_kinematics = UrdfPhysicalCameraKinematics(args.target_urdf.resolve())
    semantic_reference, seed_fk_error = build_semantic_reference(
        reference, teacher, position_kinematics
    )
    summary, arrays = retarget_case(
        teacher,
        semantic_reference,
        position_kinematics,
        camera_kinematics,
        args,
    )
    # The legacy acquisition builder exports achieved FK at its endpoint.
    # Exact-source artifacts must expose immutable anchor zero as the first
    # semantic desired target while keeping initialization states separate.
    semantic_start = install_exact_source_anchor_zero(arrays, reference)
    anchor_indices = source_anchor_execution_indices(
        reference.positions_m,
        reference.attitudes_wxyz,
        arrays["target_position_world_m"],
        arrays["target_attitude_world_dfr_quat_wxyz"],
        semantic_start,
    )
    mapped_position = arrays["target_position_world_m"][anchor_indices]
    mapped_attitude = arrays["target_attitude_world_dfr_quat_wxyz"][anchor_indices]
    mapped_attitude_error = 2.0 * np.arccos(
        np.clip(
            np.abs(np.sum(mapped_attitude * reference.attitudes_wxyz, axis=1)),
            -1.0,
            1.0,
        )
    )
    arrays.update(
        {
            "schema": np.asarray(CANDIDATE_SCHEMA),
            "source_manifest_sha256": np.asarray(reference.manifest_sha256),
            "source_json_sha256": np.asarray(reference.source_json_sha256),
            "source_pose_count": np.int32(len(reference.time_s)),
            "source_time_s": reference.time_s.copy(),
            "source_position_world_m": reference.positions_m.copy(),
            "source_attitude_world_dfr_quat_xyzw": reference.attitudes_xyzw.copy(),
            "source_quaternion_order": np.asarray("xyzw"),
            "source_pose_target_link": np.asarray("ee1_tool"),
            "source_semantic_forward_axis": np.asarray("+y in ee1_tool"),
            "source_time_sha256": np.asarray(_array_sha256(reference.time_s)),
            "source_position_sha256": np.asarray(_array_sha256(reference.positions_m)),
            "source_attitude_xyzw_sha256": np.asarray(
                _array_sha256(reference.attitudes_xyzw)
            ),
            "execution_time_s": arrays["time_s"].copy(),
            "execution_transition_dt_s": np.diff(arrays["time_s"]),
            "source_anchor_execution_index": anchor_indices,
            "source_anchor_execution_time_s": arrays["time_s"][anchor_indices],
            "source_interval_execution_step_count": np.diff(anchor_indices),
            "source_interval_execution_duration_s": np.diff(
                arrays["time_s"][anchor_indices]
            ),
            "initialization_sample_count": np.int32(semantic_start),
            "initialization_in_learned_actions": np.bool_(False),
            "source_reference_quality_qualified_teacher": np.bool_(False),
            "source_reference_valid_for_training": np.bool_(False),
            "offline_executable_quality_passed": np.bool_(summary["passed"]),
            "valid_for_dynamic_evaluation": np.bool_(summary["passed"]),
            "valid_for_candidate_training": np.bool_(False),
            "valid_for_training": np.bool_(False),
            "acquisition_route_contract": np.asarray(
                "minimum_total_yaw_forward_or_reverse_v1"
            ),
            "base_acquisition_route": np.asarray(
                summary["base_acquisition_route"]
            ),
            "base_acquisition_total_yaw_travel_deg": np.float64(
                summary["base_acquisition_total_yaw_travel_deg"]
            ),
            "execution_schedule_metadata_sealed": np.bool_(False),
        }
    )
    arrays.pop("source_teacher_quality_passed", None)
    arrays.pop("source_teacher_sha256", None)
    checks = dict(summary["checks"])
    checks.pop("source_ee1_tool_fk_verified", None)
    checks["immutable_source_arrays_preserved"] = True
    checks["complete_strict_source_anchor_mapping"] = bool(
        len(anchor_indices) == len(reference.time_s)
        and np.all(np.diff(anchor_indices) > 0)
    )
    checks["initialization_separate_from_source_anchors"] = bool(
        anchor_indices[0] == semantic_start
    )
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    summary.update(
        {
            "schema": CANDIDATE_SCHEMA,
            "trajectory_integrity_contract": "exact_source_v1",
            "source_manifest_sha256": reference.manifest_sha256,
            "source_json_sha256": reference.source_json_sha256,
            "source_pose_count": len(reference.time_s),
            "source_anchor_count": len(anchor_indices),
            "source_anchor_mapping_strict": bool(np.all(np.diff(anchor_indices) > 0)),
            "source_anchor_position_max_error_m": float(
                np.max(np.linalg.norm(mapped_position - reference.positions_m, axis=1))
            ),
            "source_anchor_attitude_max_error_rad": float(np.max(mapped_attitude_error)),
            "source_time_sha256": _array_sha256(reference.time_s),
            "source_position_sha256": _array_sha256(reference.positions_m),
            "source_attitude_xyzw_sha256": _array_sha256(reference.attitudes_xyzw),
            "initialization_sample_count": semantic_start,
            "initialization_in_learned_actions": False,
            "execution_sample_count": len(arrays["time_s"]),
            "execution_transition_count": len(arrays["control_v_wz_darm"]),
            "integrity_seed_fk_max_error_m": seed_fk_error,
            "source_reference_quality_qualified_teacher": False,
            "source_reference_valid_for_training": False,
            "offline_executable_quality_passed": bool(summary["passed"]),
            "valid_for_dynamic_evaluation": bool(summary["passed"]),
            "valid_for_training": False,
            **seed_diagnostics,
        }
    )
    output_path = args.output_dir / f"case_{reference.episode_index:04d}.npz"
    finalize_case_artifacts(
        output_path,
        args.output_dir / f"case_{reference.episode_index:04d}.result.json",
        arrays,
        summary,
        args.checkpoint_path,
    )
    return summary


def main() -> int:
    args = parse_args()
    args.integrity_seed_prior_only = True
    args.report_exact_source_prior_progress = args.rebuild_com_safe_seed_prior
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = discover_exact_source_references(args.reference_package)
    cases = [int(value) for value in args.cases.split(",") if value.strip()]
    if not cases or any(case not in references for case in cases):
        raise ValueError(f"invalid exact-source cases: {cases}")
    if args.resume_checkpoint and args.checkpoint_path is None:
        raise ValueError("--resume-checkpoint requires --checkpoint-path")
    if args.checkpoint_cadence_source_intervals <= 0:
        raise ValueError("checkpoint cadence must be positive")
    if args.checkpoint_path is not None:
        if len(cases) != 1:
            raise ValueError("checkpointing requires exactly one requested case")
        checkpoint_exists = args.checkpoint_path.resolve().is_file()
        if args.resume_checkpoint and not checkpoint_exists:
            raise FileNotFoundError(args.checkpoint_path)
        if not args.resume_checkpoint and checkpoint_exists:
            raise FileExistsError(
                f"refusing to overwrite existing checkpoint {args.checkpoint_path}"
            )
    results = []
    for case in cases:
        try:
            result = process_case(references[case], args)
        except Exception as error:
            result = {
                "case": case,
                "passed": False,
                "valid_for_dynamic_evaluation": False,
                "valid_for_training": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            seed_diagnostics = getattr(
                error, "retarget_seed_family_diagnostics", None
            )
            if seed_diagnostics is not None:
                result["semantic_seed_family_diagnostics"] = seed_diagnostics
            (args.output_dir / f"case_{case:04d}.result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    summary = {
        "schema": "cinebotrl_two_wheel_exact_source_retarget_batch_v1",
        "trajectory_integrity_contract": "exact_source_v1",
        "source_manifest_sha256": next(iter(references.values())).manifest_sha256,
        "source_reference_quality_qualified_teacher": False,
        "source_reference_valid_for_training": False,
        "requested_cases": cases,
        "passed_cases": [int(row["case"]) for row in results if row.get("passed")],
        "rejected_cases": [int(row["case"]) for row in results if not row.get("passed")],
        "valid_for_training": False,
        "training_started": False,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0 if not summary["rejected_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
