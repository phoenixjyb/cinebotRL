#!/usr/bin/env python3
"""Build an experimental rs4_attitude_rate_v1 imitation dataset from GIK demos.

This reuses the enriched GIK .npz files and the existing offline observation
composer, but rewrites the action labels to the proposed deployment-oriented
contract:

    [arm_yaw, arm_pitch, arm_elbow,
     rs4_yaw_rate, rs4_pitch_rate, rs4_roll_rate,
     base_vx, base_vy, base_wz]

The RS4 attitude-rate labels are experimental.  They are derived from the
deployment-style target camera/gimbal attitude quaternion when available, then
finite-differenced as SciPy ZYX Euler angles in degrees.  This is a
simulator/dataset bridge, not hardware-equivalence proof.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rotation
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_gik_obs_dataset import (  # noqa: E402
    build_action_history,
    build_lookahead,
    build_components,
    chain_to_link,
    compose_batches,
    get_observation_dimensions,
    interp_quats_wxyz,
    apply_env_state_observation_sources,
    fk_ee_from_q,
    parse_urdf,
    require,
    resolve_npz_path,
)
from rl_platform.tasks.mobile_mm.action_envelopes import normalize_arm_targets as normalize_arm_targets_for_profile  # noqa: E402
from rl_platform.tasks.mobile_mm.observations import build_directional_obstacle_features  # noqa: E402
from rl_platform.tasks.mobile_mm.trajectories import semantic_dfr_to_physical_cam_quat_wxyz  # noqa: E402
from rl_platform.tasks.mobile_mm.rs4_adapter import (  # noqa: E402
    Rs4RateAdapterConfig,
    quaternion_tracking_policy_rates_deg_s,
    slew_limit_policy_rate_sequence_deg_s,
)


ACTION_DIM = 9
MAX_LINEAR_VELOCITY = 1.5
MAX_ANGULAR_VELOCITY = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=Path("data/gik_ik_demos"))
    parser.add_argument("--manifest", type=str, default="manifest_strict.json")
    parser.add_argument("--output", type=Path, default=Path("data/gik_rs4_attitude_demos/obs_dataset_strict_rs4.npz"))
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--action-history-length", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--safety-radius", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--urdf", type=Path, default=Path("assets_own/recomoProto2-1190_moveit.urdf"))
    parser.add_argument(
        "--arm-envelope-profile",
        choices=["proto2_safe_v1", "teacher_wide_v1"],
        default="teacher_wide_v1",
    )
    parser.add_argument(
        "--resample-dt",
        type=float,
        default=0.1,
        help="Progress-retime every accepted teacher to this deliberate execution clock.",
    )
    parser.add_argument(
        "--obstacle-observation-mode",
        choices=["scalar_clearance_v1", "relative_two_v2"],
        default="relative_two_v2",
    )
    parser.add_argument("--max-obstacles", type=int, default=2)
    parser.add_argument(
        "--allow-incomplete-gimbal-solve-diagnostic",
        action="store_true",
        help="Write a diagnostic-only dataset when some physical-gimbal rows exceed 2 deg.",
    )
    parser.add_argument("--enable-roll", action="store_true", help="Include roll-rate labels. Default masks roll.")
    parser.add_argument(
        "--attitude-response-horizon-s",
        type=float,
        default=0.5,
        help="Time horizon used to convert camera-attitude residual into a corrective rate.",
    )
    parser.add_argument(
        "--control-dt-s",
        type=float,
        default=0.05,
        help="Fixed runtime control period used for RS4 acceleration slew limiting.",
    )
    parser.set_defaults(
        ee_state_source="fk_current",
        velocity_source="lagged_q",
        flip_fk_ee_quat=False,
        target_shift_steps=0,
        use_obstacles=True,
    )
    return parser.parse_args()


def require_corrected_physical_export(data: np.lib.npyio.NpzFile, path: Path) -> None:
    """Reject the quarantined exports that selected virtual joints as physical state."""

    require("q_selection_meta" in data.files, f"quarantined legacy export lacks q_selection_meta: {path}")
    raw = data["q_selection_meta"]
    meta = json.loads(str(raw.item() if raw.ndim == 0 else raw))
    require(
        meta.get("selected_contract") == "base3_arm3_physical_gimbal3",
        f"export does not declare physical gimbal selection: {path}",
    )
    if int(meta.get("source_q_dim", 9)) == 13:
        require(
            meta.get("selected_q_indices_0based") == [0, 1, 2, 3, 4, 5, 10, 11, 12],
            f"13D export has wrong physical gimbal indices: {path}",
        )


def convert_target_quaternion(target_quat: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        converted = semantic_dfr_to_physical_cam_quat_wxyz(torch.from_numpy(target_quat).float())
    return converted.numpy().astype(np.float32)


def trim_ramp_prefix(
    components: dict[str, np.ndarray],
    item: dict,
) -> tuple[dict[str, np.ndarray], int]:
    """Drop exporter ramp rows, whose poses are physical setup poses, not semantic DFR targets."""

    total_rows = int(components["actions"].shape[0])
    completed = item.get("completed_waypoints")
    if completed is None:
        raise ValueError("corrected frame export must declare completed_waypoints for ramp quarantine")
    ramp_rows = total_rows - int(completed)
    require(0 <= ramp_rows < total_rows, f"invalid ramp row count {ramp_rows}/{total_rows}")
    trimmed: dict[str, np.ndarray] = {}
    for key, value in components.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_rows:
            trimmed[key] = value[ramp_rows:].copy()
        else:
            trimmed[key] = value
    return trimmed, ramp_rows


def solve_physical_gimbal_trajectory(
    joint_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    fk_chain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve physical gimbal joints with base/arm fixed for each target attitude."""

    solved = np.asarray(joint_pos, dtype=np.float32).copy()
    errors_deg = np.empty(solved.shape[0], dtype=np.float32)
    success = np.zeros(solved.shape[0], dtype=bool)
    lower = np.asarray([-np.pi, -3.2, -3.2], dtype=np.float64)
    upper = np.asarray([np.pi, np.pi / 2, np.pi / 2], dtype=np.float64)
    seed = np.clip(solved[0, 6:9].astype(np.float64), lower, upper)

    for index in range(solved.shape[0]):
        fixed = solved[index].copy()
        target_xyzw = target_quat_wxyz[index, [1, 2, 3, 0]].astype(np.float64)
        target_rotation = Rotation.from_quat(target_xyzw)

        def residual(gimbal: np.ndarray) -> np.ndarray:
            candidate = fixed.copy()
            candidate[6:9] = gimbal.astype(np.float32)
            _, current_wxyz = fk_ee_from_q(candidate[None, :], fk_chain)
            current_rotation = Rotation.from_quat(current_wxyz[0, [1, 2, 3, 0]].astype(np.float64))
            return (current_rotation.inv() * target_rotation).as_rotvec()

        source_seed = np.clip(fixed[6:9].astype(np.float64), lower, upper)
        starts = [
            seed,
            source_seed,
            np.zeros(3),
            np.asarray([np.pi / 2, 0.0, 0.0]),
            np.asarray([-np.pi / 2, 0.0, 0.0]),
            np.asarray([0.0, -np.pi / 2, 0.0]),
            np.asarray([0.0, 0.0, -np.pi / 2]),
            np.asarray([0.0, 1.3, 0.5]),
            np.asarray([0.7, 1.1, 0.4]),
            np.asarray([2.0, 1.5, -1.2]),
            np.asarray([2.15, -1.45, 0.05]),
            np.asarray([1.65, -1.7, -0.65]),
            np.asarray([1.2, -1.5, -1.0]),
            np.asarray([0.0, -3.0, -0.5]),
            np.asarray([-0.1, -3.0, -0.5]),
            np.asarray([1.5, -np.pi + 1e-3, -1.25]),
            np.asarray([np.pi - 1e-3, 0.0, 0.0]),
            np.asarray([-np.pi + 1e-3, 0.0, 0.0]),
        ]
        best_result = None
        best_error_deg = float("inf")
        for start in starts:
            result = least_squares(
                residual,
                np.clip(start, lower, upper),
                bounds=(lower, upper),
                method="trf",
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                max_nfev=100,
                diff_step=1e-3,
            )
            error_deg = float(np.rad2deg(np.linalg.norm(residual(result.x))))
            if error_deg < best_error_deg:
                best_result = result
                best_error_deg = error_deg
            if best_error_deg <= 0.05:
                break
        assert best_result is not None
        solved[index, 6:9] = best_result.x.astype(np.float32)
        errors_deg[index] = np.float32(best_error_deg)
        success[index] = bool(
            best_result.success
            and np.isfinite(errors_deg[index])
            and errors_deg[index] <= 2.0
        )
        seed = best_result.x
    return solved, errors_deg, success


def normalize_real_arm_targets(q_arm: np.ndarray, profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.zeros((q_arm.shape[0], 6), dtype=np.float32)
    padded[:, :3] = q_arm
    actions, raw, valid = normalize_arm_targets_for_profile(padded, profile=profile)
    return actions[:, :3], raw[:, :3], valid[:, :3]


def fixed_rate_base_actions(joint_pos: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    next_pos = np.empty_like(joint_pos)
    next_pos[:-1] = joint_pos[1:]
    next_pos[-1] = joint_pos[-1]
    dxy_world = (next_pos[:, :2] - joint_pos[:, :2]) / float(dt_s)
    dyaw = (next_pos[:, 2] - joint_pos[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    dyaw /= float(dt_s)
    yaw = joint_pos[:, 2]
    c = np.cos(yaw)
    s = np.sin(yaw)
    raw = np.stack(
        [
            (c * dxy_world[:, 0] + s * dxy_world[:, 1]) / MAX_LINEAR_VELOCITY,
            (-s * dxy_world[:, 0] + c * dxy_world[:, 1]) / MAX_LINEAR_VELOCITY,
            dyaw / MAX_ANGULAR_VELOCITY,
        ],
        axis=1,
    )
    return np.clip(raw, -1.0, 1.0).astype(np.float32), (np.abs(raw) <= 1.0) & np.isfinite(raw)


def retime_quaternion_by_progress(quat: np.ndarray, output_count: int) -> np.ndarray:
    src_t = np.linspace(0.0, 1.0, quat.shape[0], dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, output_count, dtype=np.float64)
    return interp_quats_wxyz(quat, src_t, dst_t)


def build_obstacle_features_from_data(
    data: np.lib.npyio.NpzFile,
    components: dict[str, np.ndarray],
    max_obstacles: int,
) -> np.ndarray:
    require(max_obstacles > 0, "--max-obstacles must be positive")
    centers = (
        data["obstacle_centers_xy"].astype(np.float32)
        if "obstacle_centers_xy" in data.files
        else (
            data["obstacle_center_xy"].astype(np.float32).reshape(1, 2)
            if "obstacle_center_xy" in data.files
            and np.isfinite(data["obstacle_center_xy"]).all()
            else np.zeros((0, 2), dtype=np.float32)
        )
    )
    radii = (
        data["obstacle_radii"].astype(np.float32)
        if "obstacle_radii" in data.files
        else (
            np.asarray([float(data["obstacle_radius"])], dtype=np.float32)
            if "obstacle_radius" in data.files and np.isfinite(float(data["obstacle_radius"]))
            else np.zeros((0,), dtype=np.float32)
        )
    )
    robot_footprint_radius = float(data["robot_footprint_radius"]) if "robot_footprint_radius" in data.files else 0.35
    count = min(max_obstacles, centers.shape[0])
    batch = components["base_pos"].shape[0]
    centers_padded = np.zeros((batch, max_obstacles, 2), dtype=np.float32)
    radii_padded = np.zeros((batch, max_obstacles), dtype=np.float32)
    clearance = np.zeros((batch, max_obstacles), dtype=np.float32)
    valid = np.zeros((batch, max_obstacles), dtype=bool)
    if count:
        centers_padded[:, :count] = centers[None, :count]
        radii_padded[:, :count] = radii[None, :count]
        delta = components["base_pos"][:, None, :2] - centers[None, :count]
        clearance[:, :count] = np.linalg.norm(delta, axis=2) - radii[None, :count] - robot_footprint_radius
        valid[:, :count] = True
    with torch.no_grad():
        features = build_directional_obstacle_features(
            torch.from_numpy(components["base_pos"]).float(),
            torch.from_numpy(components["base_quat"]).float(),
            torch.from_numpy(centers_padded).float(),
            torch.from_numpy(radii_padded).float(),
            torch.from_numpy(clearance).float(),
            torch.from_numpy(valid),
        )
    return features.numpy().astype(np.float32)


def build_rs4_actions_from_components(
    components: dict[str, np.ndarray],
    target_quat: np.ndarray,
    *,
    enable_roll: bool,
    arm_envelope_profile: str,
    response_horizon_s: float,
    teacher_dt_s: float,
    control_dt_s: float,
) -> dict[str, np.ndarray]:
    joint_pos = components["joint_pos"].astype(np.float32)
    q_next = np.empty_like(joint_pos)
    q_next[:-1] = joint_pos[1:]
    q_next[-1] = joint_pos[-1]

    arm_actions, arm_unclipped, arm_valid = normalize_real_arm_targets(
        q_next[:, 3:6],
        arm_envelope_profile,
    )

    current_quat = components["ee_quat"].astype(np.float32)

    adapter_cfg = Rs4RateAdapterConfig(enable_roll=enable_roll)
    max_rates = adapter_cfg.max_policy_order_rates
    desired_rates, feedforward_rates, attitude_residual_deg = quaternion_tracking_policy_rates_deg_s(
        current_quat,
        target_quat,
        dt_s=teacher_dt_s,
        response_horizon_s=response_horizon_s,
        config=adapter_cfg,
    )
    attitude_rate_deg_s = slew_limit_policy_rate_sequence_deg_s(
        desired_rates,
        control_dt_s,
        adapter_cfg,
    )
    rate_actions = (attitude_rate_deg_s / max_rates[None, :]).astype(np.float32)
    rate_valid = np.isfinite(rate_actions)
    if not enable_roll:
        rate_actions[:, 2] = 0.0
        rate_valid[:, 2] = False

    actions = np.zeros((q_next.shape[0], ACTION_DIM), dtype=np.float32)
    actions[:, 0:3] = arm_actions
    actions[:, 3:6] = rate_actions
    base_actions, base_valid = fixed_rate_base_actions(joint_pos, teacher_dt_s)
    actions[:, 6:9] = base_actions

    mask = np.zeros_like(actions, dtype=bool)
    mask[:, 0:3] = arm_valid
    mask[:, 3:6] = rate_valid
    mask[:, 6:9] = base_valid

    return {
        "actions": actions,
        "action_valid_mask": mask,
        "arm_action_unclipped": arm_unclipped,
        "camera_attitude_residual_deg": attitude_residual_deg.astype(np.float32),
        "camera_attitude_feedforward_rate_deg_s": feedforward_rates.astype(np.float32),
        "camera_attitude_rate_deg_s": attitude_rate_deg_s.astype(np.float32),
        "max_rs4_rate_deg_s": max_rates.astype(np.float32),
        "attitude_label_source": np.asarray("gimbal_attitude_target_quat_wxyz"),
    }


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    manifest_path = demo_dir / args.manifest
    require(manifest_path.exists(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    require(items, "manifest contains no items")
    require(
        args.resample_dt == 0.0,
        "option-B corrected exports must preserve native rows; retiming is quarantined until ramp-aware joint interpolation is implemented",
    )
    urdf_path = args.urdf.resolve()
    args.fk_chain = chain_to_link(parse_urdf(urdf_path), "base_root", "cam_link")

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_source_index: list[np.ndarray] = []
    all_current_attitude: list[np.ndarray] = []
    all_attitude_rates: list[np.ndarray] = []
    all_feedforward_rates: list[np.ndarray] = []
    all_solved_gimbal: list[np.ndarray] = []
    all_gimbal_solve_error: list[np.ndarray] = []
    all_gimbal_solve_success: list[np.ndarray] = []
    attitude_label_sources: set[str] = set()
    source_files: list[str] = []
    source_scenarios: list[str] = []
    source_ramp_rows_excluded: list[int] = []

    for idx, item in enumerate(items):
        npz_path = resolve_npz_path(item, demo_dir)
        with np.load(npz_path) as data:
            require_corrected_physical_export(data, npz_path)
            components = build_components(data, args, duration_s=item.get("duration_s"))
            components, ramp_rows = trim_ramp_prefix(components, item)
            teacher_dt_s = (
                float(args.resample_dt)
                if args.resample_dt > 0.0
                else float(np.median(data["dt"].astype(np.float32)))
            )
            target_quat_key = (
                "gimbal_attitude_target_quat_wxyz"
                if "gimbal_attitude_target_quat_wxyz" in data.files
                else "target_quat_wxyz"
            )
            semantic_target_quat = retime_quaternion_by_progress(
                data[target_quat_key].astype(np.float32)[ramp_rows:],
                components["joint_pos"].shape[0],
            )
            target_quat = convert_target_quaternion(semantic_target_quat)
            components["target_quat"] = target_quat
            components["lookahead_pos"] = build_lookahead(
                components["target_pos"], args.lookahead_steps
            ).astype(np.float32)
            solved_joint_pos, solve_error_deg, solve_success = solve_physical_gimbal_trajectory(
                components["joint_pos"],
                target_quat,
                args.fk_chain,
            )
            components["joint_pos"] = solved_joint_pos
            apply_env_state_observation_sources(components, args, teacher_dt_s)
            rs4 = build_rs4_actions_from_components(
                components,
                target_quat,
                enable_roll=args.enable_roll,
                arm_envelope_profile=args.arm_envelope_profile,
                response_horizon_s=args.attitude_response_horizon_s,
                teacher_dt_s=teacher_dt_s,
                control_dt_s=args.control_dt_s,
            )
            components["actions"] = rs4["actions"]
            components["action_valid_mask"] = rs4["action_valid_mask"]
            components["action_history"] = build_action_history(rs4["actions"], args.action_history_length).astype(np.float32)
            if args.obstacle_observation_mode == "relative_two_v2":
                components.pop("min_obstacle_dist", None)
                components["obstacle_features"] = build_obstacle_features_from_data(
                    data,
                    components,
                    args.max_obstacles,
                )
            obs = compose_batches(components, args.batch_size)
            actions = rs4["actions"]

        all_obs.append(obs)
        all_actions.append(actions)
        all_masks.append(rs4["action_valid_mask"])
        all_source_index.append(np.full(actions.shape[0], idx, dtype=np.int32))
        all_current_attitude.append(rs4["camera_attitude_residual_deg"])
        all_attitude_rates.append(rs4["camera_attitude_rate_deg_s"])
        all_feedforward_rates.append(rs4["camera_attitude_feedforward_rate_deg_s"])
        all_solved_gimbal.append(solved_joint_pos[:, 6:9].astype(np.float32))
        all_gimbal_solve_error.append(solve_error_deg)
        all_gimbal_solve_success.append(solve_success)
        attitude_label_sources.add(str(rs4["attitude_label_source"]))
        source_files.append(npz_path.name)
        source_scenarios.append(str(item.get("obstacle_case") or "unknown"))
        source_ramp_rows_excluded.append(ramp_rows)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    action_valid_mask = np.concatenate(all_masks, axis=0)
    source_index = np.concatenate(all_source_index, axis=0)
    attitude_residual = np.concatenate(all_current_attitude, axis=0)
    attitude_rates = np.concatenate(all_attitude_rates, axis=0)
    feedforward_rates = np.concatenate(all_feedforward_rates, axis=0)
    solved_gimbal = np.concatenate(all_solved_gimbal, axis=0)
    gimbal_solve_error = np.concatenate(all_gimbal_solve_error, axis=0)
    gimbal_solve_success = np.concatenate(all_gimbal_solve_success, axis=0)
    training_eligible = bool(np.all(gimbal_solve_success))
    if not training_eligible and not args.allow_incomplete_gimbal_solve_diagnostic:
        raise ValueError(
            "physical-gimbal solve gate failed: "
            f"{int(gimbal_solve_success.sum())}/{gimbal_solve_success.size} rows <= 2 deg; "
            "use --allow-incomplete-gimbal-solve-diagnostic only for a quarantined smoke artifact"
        )
    action_valid_mask[~gimbal_solve_success, 3:6] = False

    expected_dim = get_observation_dimensions(
        num_joints=6,
        num_contacts=args.num_contacts,
        use_lookahead=True,
        lookahead_steps=args.lookahead_steps,
        use_action_history=True,
        action_history_length=args.action_history_length,
        action_dim=ACTION_DIM,
        use_obstacles=True,
        obstacle_feature_dim=(args.max_obstacles * 5 if args.obstacle_observation_mode == "relative_two_v2" else 1),
    )
    require(observations.shape[1] == expected_dim, f"obs dim {observations.shape[1]} != expected {expected_dim}")
    require(np.isfinite(observations).all(), "observations contain non-finite values")
    require(np.isfinite(actions).all(), "actions contain non-finite values")
    require(np.max(np.abs(actions)) <= 1.000001, "actions outside [-1,1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations,
        actions=actions,
        action_valid_mask=action_valid_mask,
        source_index=source_index,
        source_files=np.asarray(source_files),
        source_scenarios=np.asarray(source_scenarios),
        source_ramp_rows_excluded=np.asarray(source_ramp_rows_excluded, dtype=np.int32),
        manifest=str(manifest_path),
        schema=np.asarray("cinebotrl_gik_rs4_attitude_rate_demo_v2"),
        action_contract=np.asarray("rs4_attitude_rate_v1"),
        action_names=np.asarray([
            "arm_yaw",
            "arm_pitch",
            "arm_elbow",
            "rs4_yaw_rate",
            "rs4_pitch_rate",
            "rs4_roll_rate",
            "base_vx",
            "base_vy",
            "base_wz",
        ]),
        camera_attitude_residual_deg=attitude_residual,
        camera_attitude_feedforward_rate_deg_s=feedforward_rates,
        camera_attitude_rate_deg_s=attitude_rates,
        physical_gimbal_solved_rad=solved_gimbal,
        physical_gimbal_solve_error_deg=gimbal_solve_error,
        physical_gimbal_solve_success=gimbal_solve_success,
        attitude_frame_convention=np.asarray("local_camera_rotation_vector_zyx_from_" + ",".join(sorted(attitude_label_sources))),
        attitude_label_mode=np.asarray("quaternion_tracking_slew_limited_v1"),
        target_orientation_contract=np.asarray("semantic_dfr_to_physical_cam_v1"),
        observation_ee_frame=np.asarray("physical_cam_link_fk"),
        source_export_contract=np.asarray("base3_arm3_physical_gimbal3"),
        training_eligible=np.asarray(training_eligible),
        incomplete_gimbal_rows=np.asarray(int((~gimbal_solve_success).sum()), dtype=np.int32),
        attitude_response_horizon_s=np.asarray(args.attitude_response_horizon_s, dtype=np.float32),
        control_dt_s=np.asarray(args.control_dt_s, dtype=np.float32),
        resample_dt_s=np.asarray(args.resample_dt, dtype=np.float32),
        arm_envelope_profile=np.asarray(args.arm_envelope_profile),
        obstacle_observation_mode=np.asarray(args.obstacle_observation_mode),
        max_obstacles=np.asarray(args.max_obstacles, dtype=np.int32),
        attitude_label_source=np.asarray("option_b_physical_cam_link_target_from_semantic_dfr"),
        rs4_axis_order=np.asarray("[yaw, roll, pitch] via local [roll, pitch, yaw] map [2,0,1]"),
        roll_enabled=np.asarray(args.enable_roll),
        max_rs4_rate_deg_s=Rs4RateAdapterConfig(enable_roll=args.enable_roll).max_policy_order_rates,
        observation_dim=np.asarray(observations.shape[1]),
    )

    print(f"Manifest:      {manifest_path}")
    print(f"Output:        {args.output}")
    print(f"Trajectories:  {len(items)}")
    print(f"Samples:       {observations.shape[0]}")
    print(f"Obs dim:       {observations.shape[1]}")
    print(f"Roll enabled:  {args.enable_roll}")
    print("Valid labels:  " + " ".join(f"{v:.3f}" for v in action_valid_mask.mean(axis=0)))
    print("Max |action|:  " + " ".join(f"{v:.3f}" for v in np.max(np.abs(actions), axis=0)))
    print("Rate deg/s p95:" + " ".join(f" {v:.1f}" for v in np.percentile(np.abs(attitude_rates), 95, axis=0)))
    print(
        "Gimbal solve:   "
        f"{int(gimbal_solve_success.sum())}/{gimbal_solve_success.size} success, "
        f"mean/p95/max={np.mean(gimbal_solve_error):.3f}/"
        f"{np.percentile(gimbal_solve_error, 95):.3f}/{np.max(gimbal_solve_error):.3f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
