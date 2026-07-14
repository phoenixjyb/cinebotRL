#!/usr/bin/env python3
"""Build accepted-only CineBotRL observations from split GIK teachers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_gik_obs_dataset import (  # noqa: E402
    MAX_ANGULAR_VELOCITY,
    MAX_LINEAR_VELOCITY,
    angular_velocity_from_quats,
    backward_difference,
    build_action_history,
    chain_to_link,
    compose_batches,
    fk_ee_from_q,
    get_observation_dimensions,
    interp_quats_wxyz,
    parse_urdf,
    require,
    yaw_to_quat_wxyz,
)


EXPECTED_SCHEMA = "cinebotrl_gik_split_teacher_v1"
EXPECTED_LEARNED_CONTRACT = "base_arm_6 plus separate world DFR gimbal attitude target"
EXPECTED_GIMBAL_CONTRACT = "diagnostic only; DJI/runtime adapter performs attitude IK"
ACTION_DIM = 9
ACTION_MASK = np.asarray([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
ACTION_NAMES = np.asarray(
    [
        "arm_yaw",
        "arm_pitch",
        "arm_elbow",
        "attitude_adapter_reserved_yaw",
        "attitude_adapter_reserved_pitch",
        "attitude_adapter_reserved_roll",
        "base_vx",
        "base_vy",
        "base_wz",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--summary", default="COMPLETED.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage-output",
        type=Path,
        help="Optional accepted-only recorded stage generated from the same aligned rows.",
    )
    parser.add_argument(
        "--duration-manifest",
        type=Path,
        help="Recorded-stage manifest supplying original per-episode durations for synchronized retiming.",
    )
    parser.add_argument("--retime-dt", type=float, default=0.1)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "assets_own/recomoProto2-1190_moveit.urdf",
    )
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--lookahead-dt", type=float, default=0.1)
    parser.add_argument("--action-history-length", type=int, default=2)
    parser.add_argument("--num-contacts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--observation-contract",
        choices=["split_reference_v1", "split_reference_v2"],
        default="split_reference_v1",
    )
    parser.add_argument("--reference-time-scale-s", type=float, default=30.0)
    return parser.parse_args()


def scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    require(key in data.files, f"split teacher is missing {key}")
    value = data[key]
    require(value.ndim == 0, f"{key} must be a scalar")
    return str(value.item())


def resolve_episode_npz(root: Path, episode_index: int) -> Path:
    candidates = sorted((root / f"episode_{episode_index:04d}").glob("*_split_teacher_v1.npz"))
    require(len(candidates) == 1, f"episode {episode_index}: expected one split-teacher NPZ, got {len(candidates)}")
    return candidates[0]


def load_duration_map(manifest_path: Path) -> dict[int, float]:
    require(manifest_path.exists(), f"missing duration manifest: {manifest_path}")
    durations: dict[int, float] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.exists():
            path = manifest_path.parent / path.name
        require(path.exists(), f"duration-stage trajectory is missing: {line}")
        document = json.loads(path.read_text(encoding="utf-8"))
        metadata = document.get("metadata", {})
        episode_index = int(metadata.get("episode_index", path.name[:4]))
        duration_s = float(metadata["duration_s"])
        require(duration_s >= 5.0, f"episode {episode_index}: source duration {duration_s:.3f}s < 5s")
        durations[episode_index] = duration_s
    return durations


def quaternion_error_deg(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = lhs / np.maximum(np.linalg.norm(lhs, axis=1, keepdims=True), 1e-12)
    rhs = rhs / np.maximum(np.linalg.norm(rhs, axis=1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(lhs * rhs, axis=1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def build_strided_lookahead(values: np.ndarray, steps: int, stride: int) -> np.ndarray:
    require(steps > 0, f"lookahead steps must be positive, got {steps}")
    require(stride > 0, f"lookahead stride must be positive, got {stride}")
    idx = np.arange(values.shape[0])[:, None] + stride * np.arange(1, steps + 1)[None, :]
    return values[np.clip(idx, 0, values.shape[0] - 1)]


def semantic_dfr_to_physical_cam_quat(semantic_wxyz: np.ndarray) -> np.ndarray:
    """Apply Option B by right-multiplying the semantic attitude by Rz(+pi/2)."""

    c = np.float32(np.sqrt(0.5))
    w, x, y, z = semantic_wxyz.T
    converted = np.stack(
        [c * (w - z), c * (x + y), c * (y - x), c * (w + z)],
        axis=1,
    )
    return converted / np.maximum(np.linalg.norm(converted, axis=1, keepdims=True), 1e-12)


def retime_split_teacher(
    *,
    q_current: np.ndarray,
    q_next: np.ndarray,
    base_arm_actions: np.ndarray,
    semantic_quat: np.ndarray,
    duration_s: float,
    retime_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synchronously retime state, attitude, and labels over normalized progress."""

    require(retime_dt > 0.0, f"retime_dt must be positive, got {retime_dt}")
    source_states = np.concatenate([q_current[:1], q_next], axis=0).astype(np.float64)
    source_states[:, 2:] = np.unwrap(source_states[:, 2:], axis=0)
    source_progress = np.linspace(0.0, 1.0, source_states.shape[0])
    output_steps = max(1, int(np.ceil(duration_s / retime_dt)))
    output_time = np.linspace(0.0, duration_s, output_steps + 1)
    output_progress = output_time / duration_s
    output_states = np.stack(
        [np.interp(output_progress, source_progress, source_states[:, column]) for column in range(9)],
        axis=1,
    ).astype(np.float32)
    retimed_current = output_states[:-1]
    retimed_next = output_states[1:]
    dt = np.diff(output_time).astype(np.float32)

    source_action_progress = np.linspace(1.0 / base_arm_actions.shape[0], 1.0, base_arm_actions.shape[0])
    output_action_progress = output_progress[1:]
    retimed_arm = np.stack(
        [
            np.interp(output_action_progress, source_action_progress, base_arm_actions[:, column])
            for column in range(3)
        ],
        axis=1,
    ).astype(np.float32)

    delta_world = (retimed_next[:, :2] - retimed_current[:, :2]) / dt[:, None]
    yaw = retimed_current[:, 2]
    c = np.cos(yaw)
    s = np.sin(yaw)
    delta_yaw = (retimed_next[:, 2] - retimed_current[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    base_actions = np.stack(
        [
            (c * delta_world[:, 0] + s * delta_world[:, 1]) / MAX_LINEAR_VELOCITY,
            (-s * delta_world[:, 0] + c * delta_world[:, 1]) / MAX_LINEAR_VELOCITY,
            (delta_yaw / dt) / MAX_ANGULAR_VELOCITY,
        ],
        axis=1,
    ).astype(np.float32)
    require(float(np.max(np.abs(base_actions))) <= 1.0 + 1e-4, "retimed base actions exceed runtime bounds")

    source_attitude_progress = np.linspace(1.0 / semantic_quat.shape[0], 1.0, semantic_quat.shape[0])
    source_attitude = np.concatenate([semantic_quat[:1], semantic_quat], axis=0)
    source_attitude_progress = np.concatenate([[0.0], source_attitude_progress])
    retimed_semantic = interp_quats_wxyz(
        source_attitude,
        source_attitude_progress,
        output_action_progress,
    ).astype(np.float32)
    retimed_actions = np.concatenate([retimed_arm, base_actions], axis=1)
    return retimed_current, retimed_next, dt, retimed_actions, retimed_semantic


def build_components(
    data: np.lib.npyio.NpzFile,
    *,
    fk_chain,
    lookahead_steps: int,
    action_history_length: int,
    num_contacts: int,
    observation_contract: str,
    lookahead_dt: float,
    reference_time_scale_s: float,
    duration_s: float | None = None,
    retime_dt: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    require(scalar_text(data, "schema") == EXPECTED_SCHEMA, "unsupported or quarantined split-teacher schema")
    require(bool(data["valid_for_training"].item()), "split teacher is not approved for training")
    require(scalar_text(data, "quaternion_order") == "wxyz", "split teacher quaternion order must be wxyz")
    require(scalar_text(data, "learned_contract") == EXPECTED_LEARNED_CONTRACT, "wrong learned contract")
    require(scalar_text(data, "physical_gimbal_contract") == EXPECTED_GIMBAL_CONTRACT, "wrong gimbal ownership contract")

    base_arm_actions = data["base_arm_actions"].astype(np.float32)
    q_current = data["q_current_physical_9"].astype(np.float32)
    q_next = data["q_next_physical_9"].astype(np.float32)
    dt = data["dt_s"].astype(np.float32)
    target_quat = data["target_cam_link_quat_wxyz"].astype(np.float32)
    actual_next_quat = data["actual_cam_link_quat_wxyz"].astype(np.float32)
    semantic_quat = data["gimbal_attitude_target_world_dfr_quat_wxyz"].astype(np.float32)

    count = base_arm_actions.shape[0]
    for name, value, shape in (
        ("base_arm_actions", base_arm_actions, (count, 6)),
        ("q_current_physical_9", q_current, (count, 9)),
        ("q_next_physical_9", q_next, (count, 9)),
        ("dt_s", dt, (count,)),
        ("target_cam_link_quat_wxyz", target_quat, (count, 4)),
        ("actual_cam_link_quat_wxyz", actual_next_quat, (count, 4)),
        ("gimbal_attitude_target_world_dfr_quat_wxyz", semantic_quat, (count, 4)),
    ):
        require(value.shape == shape, f"{name} shape {value.shape} != {shape}")
        require(np.isfinite(value).all(), f"{name} contains non-finite values")
    require(np.all(dt > 0.0), "dt_s must be positive")

    _, source_fk_next_quat = fk_ee_from_q(q_next, fk_chain)
    fk_error_deg = quaternion_error_deg(source_fk_next_quat, actual_next_quat)
    option_b_error_deg = quaternion_error_deg(
        semantic_dfr_to_physical_cam_quat(semantic_quat),
        target_quat,
    )
    source_target_error_deg = quaternion_error_deg(actual_next_quat, target_quat)
    require(
        float(np.max(fk_error_deg)) <= 0.25,
        f"physical cam_link FK disagrees with teacher by {float(np.max(fk_error_deg)):.3f} deg",
    )
    require(
        float(np.max(option_b_error_deg)) <= 0.1,
        f"teacher target violates Option B by {float(np.max(option_b_error_deg)):.3f} deg",
    )

    if duration_s is not None:
        q_current, q_next, dt, base_arm_actions, semantic_quat = retime_split_teacher(
            q_current=q_current,
            q_next=q_next,
            base_arm_actions=base_arm_actions,
            semantic_quat=semantic_quat,
            duration_s=duration_s,
            retime_dt=retime_dt,
        )
        target_quat = semantic_dfr_to_physical_cam_quat(semantic_quat).astype(np.float32)
        count = base_arm_actions.shape[0]

    actions = np.zeros((count, ACTION_DIM), dtype=np.float32)
    actions[:, :3] = base_arm_actions[:, :3]
    actions[:, 6:9] = base_arm_actions[:, 3:6]
    action_mask = np.broadcast_to(ACTION_MASK, actions.shape).copy()

    ee_pos, ee_quat = fk_ee_from_q(q_current, fk_chain)
    target_pos, _ = fk_ee_from_q(q_next, fk_chain)

    use_reference_v2 = observation_contract == "split_reference_v2"
    playback_dt = float(retime_dt if duration_s is not None else np.median(dt))
    require(playback_dt > 0.0, "teacher playback dt must be positive")
    lookahead_stride = 1
    if use_reference_v2:
        require(lookahead_dt > 0.0, "reference lookahead dt must be positive")
        require(reference_time_scale_s > 0.0, "reference time scale must be positive")
        stride_float = lookahead_dt / playback_dt
        lookahead_stride = int(round(stride_float))
        require(
            lookahead_stride >= 1 and abs(stride_float - lookahead_stride) <= 1e-3,
            f"lookahead dt {lookahead_dt} is not an integer multiple of playback dt {playback_dt}",
        )
    lookahead_pos = build_strided_lookahead(target_pos, lookahead_steps, lookahead_stride).astype(np.float32)

    q_velocity = backward_difference(q_current, dt)
    base_pos = np.zeros((count, 3), dtype=np.float32)
    base_pos[:, :2] = q_current[:, :2]
    base_quat = yaw_to_quat_wxyz(q_current[:, 2]).astype(np.float32)
    base_lin_vel = np.zeros((count, 3), dtype=np.float32)
    base_lin_vel[:, :2] = q_velocity[:, :2] / MAX_LINEAR_VELOCITY
    base_ang_vel = np.zeros((count, 3), dtype=np.float32)
    base_ang_vel[:, 2] = q_velocity[:, 2] / MAX_ANGULAR_VELOCITY
    ee_lin_vel = backward_difference(ee_pos, dt)
    ee_ang_vel = angular_velocity_from_quats(ee_quat, dt)

    components = {
        "actions": actions,
        "action_valid_mask": action_mask,
        "base_pos": base_pos,
        "base_quat": base_quat,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "joint_pos": q_current,
        "joint_vel": q_velocity,
        "ee_pos": ee_pos.astype(np.float32),
        "ee_quat": ee_quat.astype(np.float32),
        "ee_lin_vel": ee_lin_vel.astype(np.float32),
        "ee_ang_vel": ee_ang_vel.astype(np.float32),
        "target_pos": target_pos.astype(np.float32),
        "target_quat": target_quat,
        "lookahead_pos": lookahead_pos,
        "action_history": build_action_history(actions, action_history_length).astype(np.float32),
        "contact_forces": np.zeros((count, num_contacts), dtype=np.float32),
    }
    if use_reference_v2:
        lookahead_quat = build_strided_lookahead(
            target_quat,
            lookahead_steps,
            lookahead_stride,
        ).astype(np.float32)
        progress = np.arange(count, dtype=np.float32) / float(max(count - 1, 1))
        remaining_s = (count - 1 - np.arange(count, dtype=np.float32)) * playback_dt
        target_lin_vel = np.clip(
            (lookahead_pos[:, 0, :] - target_pos) / float(lookahead_dt) / MAX_LINEAR_VELOCITY,
            -2.0,
            2.0,
        ).astype(np.float32)
        components.update(
            {
                "lookahead_quat": lookahead_quat,
                "trajectory_progress": progress[:, None],
                "trajectory_time_remaining": np.clip(
                    remaining_s / float(reference_time_scale_s),
                    0.0,
                    2.0,
                )[:, None].astype(np.float32),
                "target_lin_vel": target_lin_vel,
            }
        )
    diagnostics = {
        "max_fk_actual_quaternion_error_deg": float(np.max(fk_error_deg)),
        "max_option_b_quaternion_error_deg": float(np.max(option_b_error_deg)),
        "max_teacher_target_orientation_error_deg": float(
            np.max(source_target_error_deg)
        ),
    }
    return components, diagnostics, semantic_quat


def main() -> int:
    args = parse_args()
    root = args.teacher_root.resolve()
    summary_path = root / args.summary
    require(summary_path.exists(), f"missing completed summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("schema") == "gik_physical_split_teacher_all79_v1", "unexpected all79 schema")
    require(summary.get("complete") is True, "teacher batch is incomplete")
    require(int(summary.get("error_count", -1)) == 0, "teacher batch contains export errors")

    accepted = [item for item in summary.get("items", []) if item.get("export_valid_for_training") is True]
    require(len(accepted) == int(summary.get("trainable_count", -1)), "accepted count disagrees with summary")
    require(accepted, "summary contains no accepted teachers")
    duration_map: dict[int, float] = {}
    if args.duration_manifest is not None:
        duration_map = load_duration_map(args.duration_manifest.resolve())
        missing_durations = sorted(
            int(item["episode_index"])
            for item in accepted
            if int(item["episode_index"]) not in duration_map
        )
        require(not missing_durations, f"accepted episodes are missing source durations: {missing_durations}")

    joints = parse_urdf(args.urdf.resolve())
    fk_chain = chain_to_link(joints, "base_root", "cam_link")
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_semantic_targets: list[np.ndarray] = []
    all_physical_targets: list[np.ndarray] = []
    source_index: list[np.ndarray] = []
    source_files: list[str] = []
    episode_indices: list[int] = []
    diagnostics: list[dict[str, float | int]] = []
    stage_records: list[tuple[int, str, dict]] = []

    for source_id, item in enumerate(accepted):
        episode_index = int(item["episode_index"])
        npz_path = resolve_episode_npz(root, episode_index)
        with np.load(npz_path, allow_pickle=False) as data:
            components, episode_diagnostics, semantic_target = build_components(
                data,
                fk_chain=fk_chain,
                lookahead_steps=args.lookahead_steps,
                action_history_length=args.action_history_length,
                num_contacts=args.num_contacts,
                observation_contract=args.observation_contract,
                lookahead_dt=args.lookahead_dt,
                reference_time_scale_s=args.reference_time_scale_s,
                duration_s=duration_map.get(episode_index),
                retime_dt=args.retime_dt,
            )
            obs = compose_batches(components, args.batch_size)
            count = components["actions"].shape[0]
            all_semantic_targets.append(semantic_target)
            all_physical_targets.append(components["target_quat"])
            if args.stage_output is not None:
                q_current = data["q_current_physical_9"].astype(np.float32)
                poses = [
                    {
                        "position": components["target_pos"][row].astype(float).tolist(),
                        # Recorded JSON uses xyzw; TrajectoryManager converts it to wxyz.
                        "orientation": semantic_target[row, [1, 2, 3, 0]].astype(float).tolist(),
                    }
                    for row in range(count)
                ]
                stage_records.append(
                    (
                        episode_index,
                        npz_path.name,
                        {
                            "poses": poses,
                            "metadata": {
                                "source": "corrected_physical_split_teacher",
                                "source_npz": npz_path.name,
                                "scenario": "no_obstacle",
                                "quality_status": "accepted",
                                "episode_index": episode_index,
                                "duration_s": float(duration_map.get(episode_index, np.sum(data["dt_s"]))),
                                "waypoint_dt": float(
                                    duration_map.get(episode_index, np.sum(data["dt_s"])) / count
                                ),
                                "initial_base_pose_xyyaw": q_current[0, :3].astype(float).tolist(),
                                "initial_arm_joint_pos": q_current[0, 3:9].astype(float).tolist(),
                                "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
                                "recorded_quaternion_order": "xyzw",
                                "observation_ee_frame": "physical_cam_link_fk",
                            },
                        },
                    )
                )
        all_obs.append(obs)
        all_actions.append(components["actions"])
        all_masks.append(components["action_valid_mask"])
        source_index.append(np.full(count, source_id, dtype=np.int32))
        source_files.append(npz_path.name)
        episode_indices.append(episode_index)
        diagnostics.append({"episode_index": episode_index, **episode_diagnostics})

    observations = np.concatenate(all_obs)
    actions = np.concatenate(all_actions)
    masks = np.concatenate(all_masks)
    source_index_array = np.concatenate(source_index)
    expected_dim = get_observation_dimensions(
        num_joints=6,
        num_contacts=args.num_contacts,
        use_lookahead=True,
        lookahead_steps=args.lookahead_steps,
        use_action_history=True,
        action_history_length=args.action_history_length,
        action_dim=ACTION_DIM,
        use_obstacles=False,
        use_reference_conditioning=args.observation_contract == "split_reference_v2",
    )
    require(observations.shape[1] == expected_dim, f"obs dim {observations.shape[1]} != {expected_dim}")
    require(np.isfinite(observations).all(), "observations contain non-finite values")
    require(np.array_equal(masks, np.broadcast_to(ACTION_MASK, masks.shape)), "action ownership mask drifted")

    metadata = {
        "schema": "cinebotrl_split_teacher_obs_dataset_v1",
        "source_schema": EXPECTED_SCHEMA,
        "action_contract": "split_base_arm_attitude_v1",
        "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
        "observation_ee_frame": "physical_cam_link_fk",
        "target_position_source": "physical_cam_link_fk(q_next_physical_9)",
        "physical_gimbal_labels": "masked_diagnostic_only",
        "observation_contract": args.observation_contract,
        "lookahead_dt_s": args.lookahead_dt,
        "reference_time_scale_s": args.reference_time_scale_s,
        "synchronized_retime": args.duration_manifest is not None,
        "retime_dt_s": args.retime_dt if args.duration_manifest is not None else None,
        "accepted_episode_count": len(accepted),
        "accepted_episode_indices": episode_indices,
        "rejected_episode_count": int(summary.get("rejected_count", 0)),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations,
        actions=actions,
        action_valid_mask=masks,
        source_index=source_index_array,
        source_files=np.asarray(source_files),
        source_scenarios=np.asarray(["no_obstacle"] * len(source_files)),
        source_episode_index=np.asarray(episode_indices, dtype=np.int32),
        action_names=ACTION_NAMES,
        gimbal_attitude_target_world_dfr_quat_wxyz=np.concatenate(all_semantic_targets),
        target_cam_link_quat_wxyz=np.concatenate(all_physical_targets),
        metadata=json.dumps(metadata, sort_keys=True),
        observation_dim=np.asarray(expected_dim, dtype=np.int32),
        action_contract=np.asarray("split_base_arm_attitude_v1"),
        target_orientation_contract=np.asarray("semantic_dfr_to_physical_cam_v1"),
        observation_contract=np.asarray(args.observation_contract),
    )
    if args.stage_output is not None:
        stage_root = args.stage_output.resolve()
        stage_root.mkdir(parents=True, exist_ok=True)
        manifest_lines = ["# Corrected accepted-only split-teacher stage"]
        for episode_index, source_name, document in stage_records:
            output_json = stage_root / f"episode_{episode_index:04d}_split_teacher_v1.json"
            output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            manifest_lines.append(str(output_json))
        (stage_root / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        reset_config = {
            "reset_base_to_trajectory_metadata": True,
            "reset_arm_to_trajectory_metadata": True,
            "reset_base_x_offset": 0.0,
            "reset_base_y_offset": 0.0,
            "trajectory_dt": float(args.retime_dt),
            "lookahead_dt": float(args.lookahead_dt),
            "observation_contract": args.observation_contract,
            "reference_time_scale_s": float(args.reference_time_scale_s),
            "notes": "Reset base and physical arm/gimbal from each corrected split teacher.",
        }
        (stage_root / "reset_config.json").write_text(
            json.dumps(reset_config, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "output": str(args.output),
        "accepted_episodes": len(accepted),
        "samples": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "action_mask_mean": masks.mean(axis=0).tolist(),
        "max_fk_actual_quaternion_error_deg": max(
            item["max_fk_actual_quaternion_error_deg"] for item in diagnostics
        ),
        "stage_output": str(args.stage_output) if args.stage_output is not None else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
