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
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_gik_obs_dataset import (  # noqa: E402
    build_action_history,
    build_components,
    compose_batches,
    get_observation_dimensions,
    require,
    resolve_npz_path,
)
from rl_platform.tasks.mobile_mm.rs4_adapter import Rs4RateAdapterConfig  # noqa: E402


ACTION_DIM = 9
ARM_LOWER_SAFE = np.array([-1.0, 0.55, -2.0], dtype=np.float32)
ARM_UPPER_SAFE = np.array([1.0, 1.45, -0.4], dtype=np.float32)


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
    parser.add_argument("--enable-roll", action="store_true", help="Include roll-rate labels. Default masks roll.")
    parser.set_defaults(
        arm_envelope_profile="stored",
        ee_state_source="stored",
        velocity_source="action",
        target_shift_steps=0,
        use_obstacles=True,
        resample_dt=0.0,
    )
    return parser.parse_args()


def normalize_arm_targets(q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = 2.0 * (q_arm - ARM_LOWER_SAFE[None, :]) / (ARM_UPPER_SAFE[None, :] - ARM_LOWER_SAFE[None, :]) - 1.0
    clipped = np.clip(raw, -1.0, 1.0).astype(np.float32)
    valid = np.isfinite(raw) & (np.abs(raw) <= 1.0)
    return clipped, raw.astype(np.float32), valid


def quat_wxyz_to_euler_zyx_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).as_euler("zyx", degrees=True).astype(np.float32)


def finite_difference_euler_rates_deg_s(euler_deg: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if euler_deg.shape[0] == 1:
        return np.zeros_like(euler_deg, dtype=np.float32)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(euler_deg), axis=0)).astype(np.float32)
    rates = np.zeros_like(unwrapped, dtype=np.float32)
    rates[:-1] = (unwrapped[1:] - unwrapped[:-1]) / dt[:-1, None]
    rates[-1] = rates[-2]
    return rates


def build_rs4_actions(data: np.lib.npyio.NpzFile, *, enable_roll: bool) -> dict[str, np.ndarray]:
    q_next = data["q_next"].astype(np.float32)
    dt = data["dt"].astype(np.float32)
    old_mask = data["action_valid_mask"].astype(bool)

    arm_actions, arm_unclipped, arm_valid = normalize_arm_targets(q_next[:, 3:6])

    current_attitude_deg = quat_wxyz_to_euler_zyx_deg(data["actual_ee_quat_wxyz"].astype(np.float32))
    target_quat_key = "gimbal_attitude_target_quat_wxyz" if "gimbal_attitude_target_quat_wxyz" in data.files else "target_quat_wxyz"
    target_attitude_deg = quat_wxyz_to_euler_zyx_deg(data[target_quat_key].astype(np.float32))
    attitude_rate_deg_s = finite_difference_euler_rates_deg_s(target_attitude_deg, dt)

    adapter_cfg = Rs4RateAdapterConfig(enable_roll=enable_roll)
    max_rates = adapter_cfg.max_policy_order_rates
    rate_unclipped = attitude_rate_deg_s / max_rates[None, :]
    rate_actions = np.clip(rate_unclipped, -1.0, 1.0).astype(np.float32)
    rate_valid = np.isfinite(rate_unclipped) & (np.abs(rate_unclipped) <= 1.0)
    if not enable_roll:
        rate_actions[:, 2] = 0.0
        rate_valid[:, 2] = False
        rate_unclipped[:, 2] = 0.0

    actions = np.zeros((q_next.shape[0], ACTION_DIM), dtype=np.float32)
    actions[:, 0:3] = arm_actions
    actions[:, 3:6] = rate_actions
    actions[:, 6:9] = data["actions"].astype(np.float32)[:, 6:9]

    mask = np.zeros_like(actions, dtype=bool)
    mask[:, 0:3] = arm_valid
    mask[:, 3:6] = rate_valid
    mask[:, 6:9] = old_mask[:, 6:9]

    return {
        "actions": actions,
        "action_valid_mask": mask,
        "arm_action_unclipped": arm_unclipped,
        "rs4_rate_unclipped": rate_unclipped.astype(np.float32),
        "current_camera_attitude_deg": current_attitude_deg.astype(np.float32),
        "target_camera_attitude_deg": target_attitude_deg.astype(np.float32),
        "camera_attitude_rate_deg_s": attitude_rate_deg_s.astype(np.float32),
        "max_rs4_rate_deg_s": max_rates.astype(np.float32),
        "attitude_label_source": np.asarray(target_quat_key),
    }


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    manifest_path = demo_dir / args.manifest
    require(manifest_path.exists(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    require(items, "manifest contains no items")

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_source_index: list[np.ndarray] = []
    all_current_attitude: list[np.ndarray] = []
    all_target_attitude: list[np.ndarray] = []
    all_attitude_rates: list[np.ndarray] = []
    attitude_label_sources: set[str] = set()
    source_files: list[str] = []

    for idx, item in enumerate(items):
        npz_path = resolve_npz_path(item, demo_dir)
        with np.load(npz_path) as data:
            rs4 = build_rs4_actions(data, enable_roll=args.enable_roll)
            components = build_components(data, args)
            components["actions"] = rs4["actions"]
            components["action_valid_mask"] = rs4["action_valid_mask"]
            components["action_history"] = build_action_history(rs4["actions"], args.action_history_length).astype(np.float32)
            obs = compose_batches(components, args.batch_size)
            actions = rs4["actions"]

        all_obs.append(obs)
        all_actions.append(actions)
        all_masks.append(rs4["action_valid_mask"])
        all_source_index.append(np.full(actions.shape[0], idx, dtype=np.int32))
        all_current_attitude.append(rs4["current_camera_attitude_deg"])
        all_target_attitude.append(rs4["target_camera_attitude_deg"])
        all_attitude_rates.append(rs4["camera_attitude_rate_deg_s"])
        attitude_label_sources.add(str(rs4["attitude_label_source"]))
        source_files.append(npz_path.name)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    action_valid_mask = np.concatenate(all_masks, axis=0)
    source_index = np.concatenate(all_source_index, axis=0)
    current_attitude = np.concatenate(all_current_attitude, axis=0)
    target_attitude = np.concatenate(all_target_attitude, axis=0)
    attitude_rates = np.concatenate(all_attitude_rates, axis=0)

    expected_dim = get_observation_dimensions(
        num_joints=6,
        num_contacts=args.num_contacts,
        use_lookahead=True,
        lookahead_steps=args.lookahead_steps,
        use_action_history=True,
        action_history_length=args.action_history_length,
        action_dim=ACTION_DIM,
        use_obstacles=True,
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
        manifest=str(manifest_path),
        schema=np.asarray("cinebotrl_gik_rs4_attitude_rate_demo_v1"),
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
        current_camera_attitude_deg=current_attitude,
        target_camera_attitude_deg=target_attitude,
        camera_attitude_rate_deg_s=attitude_rates,
        attitude_frame_convention=np.asarray(
            "experimental_scipy_zyx_from_" + ",".join(sorted(attitude_label_sources))
        ),
        attitude_label_source=np.asarray(",".join(sorted(attitude_label_sources))),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
