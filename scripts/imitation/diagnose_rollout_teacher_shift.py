#!/usr/bin/env python3
"""Measure closed-loop rollout shift from an offline BC teacher dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_NAMES = [
    "arm_yaw",
    "arm_pitch",
    "arm_elbow",
    "rs4_yaw_rate",
    "rs4_pitch_rate",
    "rs4_roll_rate",
    "base_vx",
    "base_vy",
    "base_wz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_npz", required=True)
    parser.add_argument("--rollout_npz", required=True)
    parser.add_argument("--teacher_source_max", type=int, default=None)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--max_teacher_rows", type=int, default=10000)
    parser.add_argument("--top_features", type=int, default=12)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nearest_teacher_indices(rollout_z: np.ndarray, teacher_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(rollout_z.shape[0], dtype=np.int64)
    distances = np.empty(rollout_z.shape[0], dtype=np.float32)
    for start in range(0, rollout_z.shape[0], 8):
        stop = min(start + 8, rollout_z.shape[0])
        delta = rollout_z[start:stop, None, :] - teacher_z[None, :, :]
        squared = np.mean(delta * delta, axis=2)
        local_indices = np.argmin(squared, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local_indices])
    return indices, distances


def main() -> int:
    args = parse_args()
    with np.load(args.teacher_npz, allow_pickle=False) as teacher_data:
        teacher_obs = teacher_data["observations"].astype(np.float32)
        teacher_actions = teacher_data["actions"].astype(np.float32)
        source_index = teacher_data["source_index"].astype(np.int64)
    with np.load(args.rollout_npz, allow_pickle=False) as rollout_data:
        rollout_obs = rollout_data["observations"].astype(np.float32)
        rollout_actions = rollout_data[
            "policy_actions" if "policy_actions" in rollout_data else "actions"
        ].astype(np.float32)

    require(teacher_obs.ndim == 2 and rollout_obs.ndim == 2, "observations must be 2D")
    require(teacher_obs.shape[1] == rollout_obs.shape[1], "teacher/rollout observation dimensions differ")
    require(teacher_actions.shape[1] == rollout_actions.shape[1], "teacher/rollout action dimensions differ")
    require(rollout_obs.shape[0] % args.num_envs == 0, "rollout rows must divide evenly by num_envs")

    if args.teacher_source_max is not None:
        selected = source_index < args.teacher_source_max
        teacher_obs = teacher_obs[selected]
        teacher_actions = teacher_actions[selected]
    require(teacher_obs.shape[0] > 0, "teacher selection is empty")

    if teacher_obs.shape[0] > args.max_teacher_rows:
        sample_indices = np.linspace(
            0,
            teacher_obs.shape[0] - 1,
            num=args.max_teacher_rows,
            dtype=np.int64,
        )
        teacher_obs = teacher_obs[sample_indices]
        teacher_actions = teacher_actions[sample_indices]

    teacher_mean = np.mean(teacher_obs, axis=0)
    teacher_std = np.std(teacher_obs, axis=0)
    scale = np.maximum(teacher_std, 1e-3)
    teacher_z = np.clip((teacher_obs - teacher_mean) / scale, -20.0, 20.0)
    rollout_z = np.clip((rollout_obs - teacher_mean) / scale, -20.0, 20.0)
    nearest_indices, nearest_distance = nearest_teacher_indices(rollout_z, teacher_z)
    nearest_actions = teacher_actions[nearest_indices]
    action_error = rollout_actions - nearest_actions

    steps = rollout_obs.shape[0] // args.num_envs
    time_index = np.repeat(np.arange(steps, dtype=np.int64), args.num_envs)
    split_step = max(1, steps // 2)
    early = time_index < split_step
    late = ~early

    abs_z = np.abs(rollout_z)
    feature_p95 = np.percentile(abs_z, 95, axis=0)
    top_indices = np.argsort(feature_p95)[::-1][: args.top_features]

    def phase_metrics(mask: np.ndarray) -> dict[str, object]:
        phase_error = action_error[mask]
        return {
            "rows": int(np.count_nonzero(mask)),
            "nearest_teacher_z_rms_mean": float(np.mean(nearest_distance[mask])),
            "nearest_teacher_z_rms_p95": float(np.percentile(nearest_distance[mask], 95)),
            "policy_action_abs_mean": {
                name: float(value)
                for name, value in zip(ACTION_NAMES, np.mean(np.abs(rollout_actions[mask]), axis=0))
            },
            "nearest_teacher_action_rmse": {
                name: float(value)
                for name, value in zip(ACTION_NAMES, np.sqrt(np.mean(phase_error * phase_error, axis=0)))
            },
        }

    report = {
        "schema": "cinebotrl_rollout_teacher_shift_v1",
        "teacher_npz": str(Path(args.teacher_npz)),
        "rollout_npz": str(Path(args.rollout_npz)),
        "teacher_source_max": args.teacher_source_max,
        "teacher_rows_compared": int(teacher_obs.shape[0]),
        "rollout_rows": int(rollout_obs.shape[0]),
        "num_envs": int(args.num_envs),
        "steps": int(steps),
        "early": phase_metrics(early),
        "late": phase_metrics(late),
        "top_shifted_observation_features": [
            {
                "index": int(index),
                "teacher_std": float(teacher_std[index]),
                "rollout_abs_z_p95": float(feature_p95[index]),
                "rollout_abs_z_max": float(np.max(abs_z[:, index])),
            }
            for index in top_indices
        ],
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
