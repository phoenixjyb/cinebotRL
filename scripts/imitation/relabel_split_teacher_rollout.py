#!/usr/bin/env python3
"""Relabel closed-loop split-policy states with corrected teacher actions.

The rollout capture supplies actual policy-visited observations plus trajectory
episode and waypoint identity. This script resolves each valid row to the exact
accepted Option-B teacher action and can merge repeated correction rows into the
original dataset for one bounded DAgger-style BC update.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


EXPECTED_ACTION_CONTRACT = "split_base_arm_attitude_v1"
EXPECTED_TARGET_CONTRACT = "semantic_dfr_to_physical_cam_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged_output", type=Path, default=None)
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--correction_repeat", type=int, default=1)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    require(key in data, f"dataset missing {key}")
    return str(np.asarray(data[key]).item())


def build_episode_rows(
    source_index: np.ndarray,
    source_episode_index: np.ndarray,
) -> dict[int, tuple[int, np.ndarray]]:
    require(source_index.ndim == 1, "teacher source_index must be 1D")
    require(source_episode_index.ndim == 1, "teacher source_episode_index must be 1D")
    groups = np.unique(source_index)
    require(groups.size == source_episode_index.size, "teacher source episode count mismatch")
    lookup: dict[int, tuple[int, np.ndarray]] = {}
    for ordinal, group in enumerate(groups.tolist()):
        episode = int(source_episode_index[ordinal])
        require(episode not in lookup, f"duplicate teacher episode {episode}")
        rows = np.flatnonzero(source_index == group)
        require(rows.size > 0, f"teacher source {group} has no rows")
        require(np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)), f"teacher source {group} is not contiguous")
        lookup[episode] = (int(group), rows)
    return lookup


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    require(bool(np.any(valid)), "correction mask has no valid labels")
    return float(np.sqrt(np.mean(np.square(prediction[valid] - target[valid]))))


def main() -> int:
    args = parse_args()
    require(args.min_step >= 0, "--min_step must be non-negative")
    require(args.correction_repeat >= 1, "--correction_repeat must be at least 1")

    with np.load(args.teacher, allow_pickle=False) as teacher:
        require(scalar_text(teacher, "action_contract") == EXPECTED_ACTION_CONTRACT, "teacher action contract mismatch")
        require(
            scalar_text(teacher, "target_orientation_contract") == EXPECTED_TARGET_CONTRACT,
            "teacher target orientation contract mismatch",
        )
        teacher_obs = teacher["observations"].astype(np.float32)
        teacher_actions = teacher["actions"].astype(np.float32)
        teacher_mask = teacher["action_valid_mask"].astype(np.float32)
        teacher_source_index = teacher["source_index"].astype(np.int32)
        teacher_source_episode = teacher["source_episode_index"].astype(np.int32)
        source_files = teacher["source_files"].copy()
        source_scenarios = teacher["source_scenarios"].copy()
        action_names = teacher["action_names"].copy()
        teacher_metadata = scalar_text(teacher, "metadata")

    with np.load(args.rollout, allow_pickle=False) as rollout:
        required = [
            "observations",
            "policy_actions",
            "rollout_env_id",
            "rollout_step",
            "rollout_waypoint_idx",
            "source_episode_index",
            "first_episode_valid",
        ]
        missing = [key for key in required if key not in rollout]
        require(not missing, f"rollout missing keys: {missing}")
        observations = rollout["observations"].astype(np.float32)
        policy_actions = rollout["policy_actions"].astype(np.float32)
        env_ids = rollout["rollout_env_id"].astype(np.int32)
        steps = rollout["rollout_step"].astype(np.int32)
        waypoint_indices = rollout["rollout_waypoint_idx"].astype(np.int32)
        episode_indices = rollout["source_episode_index"].astype(np.int32)
        first_episode_valid = rollout["first_episode_valid"].astype(bool)
        rollout_metadata = scalar_text(rollout, "metadata")

    count = observations.shape[0]
    for name, array in {
        "policy_actions": policy_actions,
        "rollout_env_id": env_ids,
        "rollout_step": steps,
        "rollout_waypoint_idx": waypoint_indices,
        "source_episode_index": episode_indices,
        "first_episode_valid": first_episode_valid,
    }.items():
        require(array.shape[0] == count, f"rollout {name} row count mismatch")
    require(observations.shape[1] == teacher_obs.shape[1], "rollout/teacher observation dimension mismatch")
    require(policy_actions.shape[1] == teacher_actions.shape[1], "rollout/teacher action dimension mismatch")

    episode_rows = build_episode_rows(teacher_source_index, teacher_source_episode)
    selected_rollout_rows: list[int] = []
    selected_teacher_rows: list[int] = []
    selected_source_groups: list[int] = []
    for row in range(count):
        if not first_episode_valid[row] or steps[row] < args.min_step:
            continue
        episode = int(episode_indices[row])
        require(episode in episode_rows, f"rollout episode {episode} is not accepted by the teacher")
        source_group, rows = episode_rows[episode]
        waypoint = int(waypoint_indices[row])
        require(0 <= waypoint < rows.size, f"episode {episode} waypoint {waypoint} is outside teacher length {rows.size}")
        selected_rollout_rows.append(row)
        selected_teacher_rows.append(int(rows[waypoint]))
        selected_source_groups.append(source_group)
    require(selected_rollout_rows, "no valid rollout rows remained after filtering")

    rollout_rows = np.asarray(selected_rollout_rows, dtype=np.int64)
    teacher_rows = np.asarray(selected_teacher_rows, dtype=np.int64)
    correction_obs = observations[rollout_rows]
    correction_actions = teacher_actions[teacher_rows]
    correction_mask = teacher_mask[teacher_rows]
    correction_source_index = np.asarray(selected_source_groups, dtype=np.int32)
    correction_policy_actions = policy_actions[rollout_rows]
    disagreement_rmse = masked_rmse(correction_policy_actions, correction_actions, correction_mask)
    metadata = {
        "schema": "cinebotrl_split_teacher_dagger_correction_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rollout": str(args.rollout),
        "teacher": str(args.teacher),
        "source_rollout_rows": int(count),
        "selected_rows": int(correction_obs.shape[0]),
        "min_step": int(args.min_step),
        "policy_teacher_masked_rmse": disagreement_rmse,
        "episodes": sorted(int(value) for value in np.unique(episode_indices[rollout_rows])),
        "first_episode_only": True,
        "action_contract": EXPECTED_ACTION_CONTRACT,
        "target_orientation_contract": EXPECTED_TARGET_CONTRACT,
        "rollout_metadata": json.loads(rollout_metadata),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=correction_obs,
        actions=correction_actions,
        action_valid_mask=correction_mask,
        source_index=correction_source_index,
        source_episode_index=episode_indices[rollout_rows],
        source_rollout_row=rollout_rows.astype(np.int32),
        source_env_id=env_ids[rollout_rows],
        source_step=steps[rollout_rows],
        source_waypoint_idx=waypoint_indices[rollout_rows],
        policy_actions=correction_policy_actions,
        action_names=action_names,
        metadata=json.dumps(metadata, sort_keys=True),
        observation_dim=np.asarray(correction_obs.shape[1], dtype=np.int32),
        action_contract=np.asarray(EXPECTED_ACTION_CONTRACT),
        target_orientation_contract=np.asarray(EXPECTED_TARGET_CONTRACT),
    )
    print(json.dumps(metadata, indent=2))
    print(f"saved correction: {args.output}")

    if args.merged_output is not None:
        repeated_obs = np.tile(correction_obs, (args.correction_repeat, 1))
        repeated_actions = np.tile(correction_actions, (args.correction_repeat, 1))
        repeated_mask = np.tile(correction_mask, (args.correction_repeat, 1))
        repeated_source = np.tile(correction_source_index, args.correction_repeat)
        merged_metadata = {
            "schema": "cinebotrl_split_teacher_dagger_merged_v1",
            "created_at": metadata["created_at"],
            "teacher_metadata": json.loads(teacher_metadata),
            "correction_metadata": metadata,
            "teacher_rows": int(teacher_obs.shape[0]),
            "correction_rows": int(correction_obs.shape[0]),
            "correction_repeat": int(args.correction_repeat),
            "merged_rows": int(teacher_obs.shape[0] + repeated_obs.shape[0]),
        }
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.merged_output,
            observations=np.concatenate([teacher_obs, repeated_obs]),
            actions=np.concatenate([teacher_actions, repeated_actions]),
            action_valid_mask=np.concatenate([teacher_mask, repeated_mask]),
            source_index=np.concatenate([teacher_source_index, repeated_source]),
            source_files=source_files,
            source_scenarios=source_scenarios,
            source_episode_index=teacher_source_episode,
            action_names=action_names,
            metadata=json.dumps(merged_metadata, sort_keys=True),
            observation_dim=np.asarray(teacher_obs.shape[1], dtype=np.int32),
            action_contract=np.asarray(EXPECTED_ACTION_CONTRACT),
            target_orientation_contract=np.asarray(EXPECTED_TARGET_CONTRACT),
        )
        print(f"saved merged dataset: {args.merged_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
