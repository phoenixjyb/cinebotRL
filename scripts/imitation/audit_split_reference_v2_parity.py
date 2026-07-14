#!/usr/bin/env python3
"""Audit source-identical split_reference_v2 fields against an Isaac capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LOOKAHEAD_POSITION_SLICE = slice(56, 65)
REFERENCE_STATIC_SLICE = slice(65, 70)
REFERENCE_ATTITUDE_SLICE = slice(70, 79)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-atol", type=float, default=1e-6)
    parser.add_argument("--initial-attitude-l2-max", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.rollout, allow_pickle=False) as rollout:
        actual = rollout["observations"].astype(np.float32)
        waypoint = rollout["rollout_waypoint_idx"].astype(np.int64)
    with np.load(args.teacher, allow_pickle=False) as teacher:
        observation_contract = str(np.asarray(teacher["observation_contract"]).item())
        if observation_contract != "split_reference_v2":
            raise ValueError(f"teacher observation contract is {observation_contract!r}")
        source_rows = np.flatnonzero(teacher["source_index"].astype(np.int64) == args.source_index)
        if source_rows.size == 0:
            raise ValueError(f"teacher source {args.source_index} has no rows")
        if np.any(waypoint < 0) or np.any(waypoint >= source_rows.size):
            raise ValueError("rollout waypoint is outside selected teacher source")
        expected = teacher["observations"][source_rows[waypoint]].astype(np.float32)
    if actual.shape != expected.shape or actual.shape[1] != 98:
        raise ValueError(f"expected matching 98D observations, got {actual.shape} and {expected.shape}")

    lookahead_abs = np.abs(actual[:, LOOKAHEAD_POSITION_SLICE] - expected[:, LOOKAHEAD_POSITION_SLICE])
    static_abs = np.abs(actual[:, REFERENCE_STATIC_SLICE] - expected[:, REFERENCE_STATIC_SLICE])
    attitude_l2 = np.linalg.norm(
        actual[:, REFERENCE_ATTITUDE_SLICE] - expected[:, REFERENCE_ATTITUDE_SLICE],
        axis=1,
    )
    lookahead_max = float(np.max(lookahead_abs))
    static_max = float(np.max(static_abs))
    initial_attitude_l2 = float(attitude_l2[0])
    passed = (
        lookahead_max <= args.source_atol
        and static_max <= args.source_atol
        and initial_attitude_l2 <= args.initial_attitude_l2_max
    )
    result = {
        "schema": "cinebotrl_split_reference_v2_parity_v1",
        "passed": passed,
        "rollout": str(args.rollout),
        "teacher": str(args.teacher),
        "source_index": args.source_index,
        "waypoint_indices": waypoint.astype(int).tolist(),
        "samples": int(actual.shape[0]),
        "observation_dim": int(actual.shape[1]),
        "lookahead_position_max_abs": lookahead_max,
        "reference_static_max_abs": static_max,
        "initial_future_attitude_l2": initial_attitude_l2,
        "future_attitude_l2_by_step": attitude_l2.astype(float).tolist(),
        "notes": (
            "Later attitude differences are diagnostic, not a source-parity failure: "
            "future attitude is expressed relative to the live physical camera."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
