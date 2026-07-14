#!/usr/bin/env python3
"""Join accepted GIK corrective labels to their captured CineBot observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ACTION_DIM = 9
LEARNED_ACTION_INDICES = np.asarray([0, 1, 2, 6, 7, 8], dtype=np.int64)
LEARNED_ACTION_MASK = np.asarray([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float32)
REQUEST_SCHEMA = "corrective_teacher_request_v1"
RESPONSE_SCHEMA = "gik_corrective_teacher_response_smoke_v1"
FRAME_CONTRACT = "semantic_dfr_to_physical_cam_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="Captured corrective request NPZ")
    parser.add_argument("--response-csv", type=Path, required=True, help="GIK response CSV")
    parser.add_argument("--response-summary", type=Path, required=True, help="GIK response summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="Accepted corrective BC dataset NPZ")
    parser.add_argument(
        "--sample-weight",
        type=float,
        default=0.5,
        help="Bounded per-row weight used when merged with nominal teachers",
    )
    return parser.parse_args()


def load_response(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), "corrective response is empty")
    required = {
        "sample_id",
        "source_episode_index",
        "rollout_step",
        "rollout_waypoint_idx",
        "accepted",
        "runtime_transition_valid",
        "action_inside_envelope",
        "srdf_collision_free",
        "terminal_position_residual_m",
        "terminal_orientation_residual_deg",
        *(f"teacher_action_{index}" for index in LEARNED_ACTION_INDICES),
    }
    missing = required.difference(rows[0])
    require(not missing, f"corrective response is missing columns: {sorted(missing)}")
    return rows


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


def build_dataset(
    request_path: Path,
    response_csv_path: Path,
    response_summary_path: Path,
    output_path: Path,
    *,
    sample_weight: float = 0.5,
) -> dict[str, object]:
    require(0.0 < sample_weight <= 1.0, "sample_weight must be in (0, 1]")
    require(request_path.exists(), f"request does not exist: {request_path}")
    require(response_csv_path.exists(), f"response CSV does not exist: {response_csv_path}")
    require(response_summary_path.exists(), f"response summary does not exist: {response_summary_path}")

    summary = json.loads(response_summary_path.read_text(encoding="utf-8"))
    require(summary.get("schema") == RESPONSE_SCHEMA, "unexpected corrective response schema")
    require(summary.get("frame_contract") == FRAME_CONTRACT, "corrective response frame contract drifted")
    require(summary.get("all_accepted") is True, "corrective response is not fully accepted")
    require(summary.get("training_allowed") is False, "GIK response must not authorize training directly")

    response_rows = load_response(response_csv_path)
    with np.load(request_path, allow_pickle=False) as request:
        request_metadata = json.loads(request["metadata"].item())
        require(request_metadata.get("schema") == REQUEST_SCHEMA, "unexpected corrective request schema")
        require(
            request_metadata.get("target_orientation_contract") == FRAME_CONTRACT,
            "corrective request frame contract drifted",
        )
        require(
            request_metadata.get("observation_contract") == "split_reference_v2",
            "corrective observations must use split_reference_v2",
        )
        observations = request["observations"].astype(np.float32)
        request_mask = request["action_label_mask"].astype(np.float32)
        source_episode = request["source_episode_index"].astype(np.int32)
        rollout_step = request["rollout_step"].astype(np.int32)
        waypoint = request["rollout_waypoint_idx"].astype(np.int32)
        progress = request["trajectory_progress"].astype(np.float32)

    count = observations.shape[0]
    require(observations.ndim == 2 and observations.shape[1] == 98, f"unexpected observations: {observations.shape}")
    require(request_mask.shape == (count, ACTION_DIM), f"unexpected action mask: {request_mask.shape}")
    require(
        np.array_equal(request_mask, np.broadcast_to(LEARNED_ACTION_MASK, request_mask.shape)),
        "captured action ownership mask drifted",
    )
    require(len(response_rows) == count, f"response rows {len(response_rows)} != request rows {count}")
    require(int(summary.get("request_sample_count", -1)) == count, "summary request count mismatch")
    require(int(summary.get("sample_count", -1)) == count, "summary response count mismatch")

    actions = np.zeros((count, ACTION_DIM), dtype=np.float32)
    accepted = np.zeros(count, dtype=bool)
    terminal_position = np.full(count, np.nan, dtype=np.float32)
    terminal_orientation = np.full(count, np.nan, dtype=np.float32)
    seen: set[int] = set()
    for row in response_rows:
        sample_id = int(row["sample_id"])
        require(0 <= sample_id < count, f"sample_id out of range: {sample_id}")
        require(sample_id not in seen, f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        require(int(row["source_episode_index"]) == int(source_episode[sample_id]), "source episode mismatch")
        require(int(row["rollout_step"]) == int(rollout_step[sample_id]), "rollout step mismatch")
        require(int(row["rollout_waypoint_idx"]) == int(waypoint[sample_id]), "waypoint mismatch")
        row_accepted = is_true(row["accepted"])
        require(row_accepted, f"response sample {sample_id} is rejected")
        require(is_true(row["runtime_transition_valid"]), f"sample {sample_id} violates runtime transitions")
        require(is_true(row["action_inside_envelope"]), f"sample {sample_id} violates action envelope")
        require(is_true(row["srdf_collision_free"]), f"sample {sample_id} violates SRDF collision gate")
        actions[sample_id, LEARNED_ACTION_INDICES] = np.asarray(
            [float(row[f"teacher_action_{index}"]) for index in LEARNED_ACTION_INDICES],
            dtype=np.float32,
        )
        terminal_position[sample_id] = float(row["terminal_position_residual_m"])
        terminal_orientation[sample_id] = float(row["terminal_orientation_residual_deg"])
        accepted[sample_id] = True

    require(len(seen) == count and accepted.all(), "response does not cover every request sample")
    require(np.isfinite(observations).all(), "observations contain non-finite values")
    require(np.isfinite(actions).all(), "actions contain non-finite values")
    require(float(np.max(np.abs(actions[:, LEARNED_ACTION_INDICES]))) <= 1.0 + 1e-6, "actions exceed [-1, 1]")

    metadata = {
        "schema": "cinebotrl_corrective_teacher_bc_dataset_v1",
        "request_schema": REQUEST_SCHEMA,
        "response_schema": RESPONSE_SCHEMA,
        "observation_contract": "split_reference_v2",
        "action_contract": "split_base_arm_attitude_v1",
        "target_orientation_contract": FRAME_CONTRACT,
        "physical_gimbal_labels": "masked_diagnostic_only",
        "request_path": str(request_path),
        "response_csv_path": str(response_csv_path),
        "response_summary_path": str(response_summary_path),
        "samples": count,
        "source_rollout_count": int(np.unique(source_episode).size),
        "sample_weight": float(sample_weight),
        "effective_weighted_rows": float(count * sample_weight),
        "training_allowed": False,
        "training_blocker": "Merge with accepted nominal teachers and pass a bounded BC live rollout gate.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        action_valid_mask=np.broadcast_to(LEARNED_ACTION_MASK, actions.shape).copy(),
        sample_weight=np.full(count, sample_weight, dtype=np.float32),
        source_index=np.zeros(count, dtype=np.int32),
        source_episode_index=source_episode,
        sample_id=np.arange(count, dtype=np.int32),
        rollout_step=rollout_step,
        rollout_waypoint_idx=waypoint,
        trajectory_progress=progress,
        terminal_position_residual_m=terminal_position,
        terminal_orientation_residual_deg=terminal_orientation,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    metadata = build_dataset(
        args.request.resolve(),
        args.response_csv.resolve(),
        args.response_summary.resolve(),
        args.output.resolve(),
        sample_weight=args.sample_weight,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
