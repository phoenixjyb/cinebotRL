#!/usr/bin/env python3
"""Merge passed per-case Isaac residual captures without trajectory leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    ACTION_SCALES,
    DATASET_SCHEMA,
    OBSERVATION_NAMES,
    load_case_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_split(
    cases: list[int], seed: int, validation_fraction: float, holdout_fraction: float
) -> dict[str, list[int]]:
    if len(cases) < 3:
        raise ValueError("at least three cases are required for grouped splits")
    if not (
        validation_fraction > 0.0
        and holdout_fraction > 0.0
        and validation_fraction + holdout_fraction < 1.0
    ):
        raise ValueError("invalid validation/holdout fractions")
    shuffled = np.asarray(sorted(cases), dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    holdout_count = max(1, int(round(len(cases) * holdout_fraction)))
    validation_count = max(1, int(round(len(cases) * validation_fraction)))
    if holdout_count + validation_count >= len(cases):
        raise ValueError("split leaves no training cases")
    return {
        "holdout": sorted(int(x) for x in shuffled[:holdout_count]),
        "validation": sorted(
            int(x)
            for x in shuffled[holdout_count : holdout_count + validation_count]
        ),
        "train": sorted(int(x) for x in shuffled[holdout_count + validation_count :]),
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.case_dir.glob("case_*_executed_residual_v1.npz"))
    if len(paths) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} captures, found {len(paths)}")
    entries = []
    seen = set()
    for path in paths:
        metadata, payload = load_case_dataset(path)
        case = int(metadata["case"])
        if case in seen:
            raise ValueError(f"duplicate case {case}")
        seen.add(case)
        entries.append((case, path, payload))
    entries.sort(key=lambda item: item[0])
    cases = [item[0] for item in entries]
    split_cases = grouped_split(
        cases,
        args.seed,
        args.validation_fraction,
        args.holdout_fraction,
    )
    split_by_case = {
        case: split
        for split, members in split_cases.items()
        for case in members
    }
    source_index = []
    split_labels = []
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "observations",
            "actions",
            "case_ids",
            "elapsed_time_s",
            "phase_time_s",
            "baseline_wheel_actions",
            "teacher_commands",
        )
    }
    split_code = {"train": 0, "validation": 1, "holdout": 2}
    for index, (case, _, payload) in enumerate(entries):
        count = len(payload["observations"])
        for name in arrays:
            arrays[name].append(payload[name])
        source_index.append(np.full(count, index, dtype=np.int16))
        split_labels.append(
            np.full(count, split_code[split_by_case[case]], dtype=np.int8)
        )
    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    merged["source_index"] = np.concatenate(source_index)
    merged["split_labels"] = np.concatenate(split_labels)
    merged["action_valid_mask"] = np.ones_like(merged["actions"], dtype=np.float32)
    reconstructed_commands = np.column_stack(
        (
            merged["observations"][:, 18]
            + ACTION_SCALES[0] * merged["actions"][:, 0],
            merged["observations"][:, 19]
            + ACTION_SCALES[1] * merged["actions"][:, 1],
            merged["observations"][:, 15]
            + ACTION_SCALES[2] * merged["actions"][:, 2],
        )
    )
    reconstruction_error = np.abs(
        reconstructed_commands - merged["teacher_commands"]
    )
    reconstruction_max_error = float(np.max(reconstruction_error))
    if reconstruction_max_error > 2e-6:
        raise ValueError(
            f"residual labels do not reconstruct teacher commands: {reconstruction_max_error}"
        )
    case_sets = [set(split_cases[name]) for name in ("train", "validation", "holdout")]
    if any(case_sets[i] & case_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("trajectory leakage detected")
    row_counts = {
        name: int(np.sum(merged["split_labels"] == code))
        for name, code in split_code.items()
    }
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_residual_merged_v1",
        "source_case_schema": DATASET_SCHEMA,
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "action_scales": ACTION_SCALES.tolist(),
        "action_contract": "trajectory_command_residual_above_frozen_balance_lqr_v1",
        "case_count": len(cases),
        "row_count": len(merged["observations"]),
        "seed": args.seed,
        "split_cases": split_cases,
        "split_row_counts": row_counts,
        "finite_values": True,
        "trajectory_leakage": False,
        "action_abs_p95": np.percentile(
            np.abs(merged["actions"]), 95, axis=0
        ).tolist(),
        "action_abs_max": np.max(np.abs(merged["actions"]), axis=0).tolist(),
        "action_clip_ratio": np.mean(
            np.abs(merged["actions"]) >= 1.0 - 1e-6, axis=0
        ).tolist(),
        "teacher_command_reconstruction_max_error": reconstruction_max_error,
        "source_files": [path.name for _, path, _ in entries],
        "source_sha256": [sha256(path) for _, path, _ in entries],
        "source_action_labels_used": False,
        "physical_gimbal_labels_used_as_actions": False,
        "training_started": False,
        "ppo_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        source_files=np.asarray(metadata["source_files"]),
        **merged,
    )
    summary_payload = metadata | {
        "dataset_file": args.output.name,
        "dataset_size_bytes": args.output.stat().st_size,
        "dataset_sha256": sha256(args.output),
    }
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
