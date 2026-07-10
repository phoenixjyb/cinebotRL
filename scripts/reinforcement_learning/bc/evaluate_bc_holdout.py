#!/usr/bin/env python3
"""Evaluate an SB3 BC checkpoint on a trajectory-disjoint dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "holdout"], default="holdout")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    selected_sources = np.asarray(split_manifest["source_indices"][args.split], dtype=np.int64)
    with np.load(args.demo_file, allow_pickle=False) as data:
        observations = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        mask = data["action_valid_mask"].astype(bool)
        source_index = data["source_index"].astype(np.int64)
        action_names = data["action_names"].astype(str).tolist()
        source_scenarios = data["source_scenarios"].astype(str) if "source_scenarios" in data else None

    selected_rows = np.isin(source_index, selected_sources)
    if not np.any(selected_rows):
        raise ValueError(f"split {args.split} selects no dataset rows")
    observations = observations[selected_rows]
    actions = actions[selected_rows]
    mask = mask[selected_rows]

    model = PPO.load(str(args.checkpoint), device=args.device)
    predictions = np.empty_like(actions)
    for start in range(0, observations.shape[0], args.batch_size):
        end = min(start + args.batch_size, observations.shape[0])
        predictions[start:end], _ = model.predict(observations[start:end], deterministic=True)

    error = predictions - actions
    squared = np.square(error)
    absolute = np.abs(error)
    valid_count = mask.sum(axis=0)
    per_channel = {}
    for index, name in enumerate(action_names):
        if valid_count[index] == 0:
            per_channel[name] = {"valid_rows": 0, "rmse": None, "mae": None, "max_abs": None}
            continue
        valid = mask[:, index]
        per_channel[name] = {
            "valid_rows": int(valid_count[index]),
            "rmse": float(np.sqrt(np.mean(squared[valid, index]))),
            "mae": float(np.mean(absolute[valid, index])),
            "max_abs": float(np.max(absolute[valid, index])),
        }

    valid_all = mask.astype(np.float32)
    scenario_metrics = {}
    if source_scenarios is not None:
        selected_scenario_rows = source_scenarios[source_index[selected_rows]]
        for scenario in sorted(np.unique(selected_scenario_rows)):
            scenario_rows = selected_scenario_rows == scenario
            scenario_mask = valid_all[scenario_rows]
            scenario_squared = squared[scenario_rows]
            scenario_metrics[str(scenario)] = {
                "source_count": int(np.unique(source_index[selected_rows][scenario_rows]).size),
                "row_count": int(np.sum(scenario_rows)),
                "masked_mse": float(
                    np.sum(scenario_squared * scenario_mask) / np.maximum(scenario_mask.sum(), 1.0)
                ),
            }
    output = {
        "schema": "cinebotrl_bc_holdout_eval_v1",
        "demo_file": str(args.demo_file.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split": args.split,
        "source_count": int(selected_sources.size),
        "row_count": int(observations.shape[0]),
        "masked_mse": float(np.sum(squared * valid_all) / np.maximum(valid_all.sum(), 1.0)),
        "masked_rmse": float(np.sqrt(np.sum(squared * valid_all) / np.maximum(valid_all.sum(), 1.0))),
        "per_channel": per_channel,
        "per_scenario": scenario_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
