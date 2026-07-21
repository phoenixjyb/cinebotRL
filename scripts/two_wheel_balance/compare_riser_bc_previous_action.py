#!/usr/bin/env python3
"""Compare validation-only BC policies under teacher and recursive action history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PREVIOUS_ACTION_SLICE = slice(23, 26)
VALIDATION_SPLIT = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def infer_teacher(model, observations: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            outputs.append(
                model(torch.from_numpy(observations[start : start + batch_size]))
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def infer_recursive(model, observations: np.ndarray) -> np.ndarray:
    outputs = []
    previous = np.zeros(3, dtype=np.float32)
    with torch.inference_mode():
        for observation in observations:
            policy_input = observation.copy()
            policy_input[PREVIOUS_ACTION_SLICE] = previous
            previous = model(torch.from_numpy(policy_input[None])).cpu().numpy()[0]
            outputs.append(previous.copy())
    return np.asarray(outputs, dtype=np.float32)


def error_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    absolute = np.abs(prediction - target)
    mse = np.mean(np.square(prediction - target), axis=0)
    return {
        "mse_per_action": mse.tolist(),
        "aggregate_mse": float(np.mean(mse)),
        "mae_per_action": np.mean(absolute, axis=0).tolist(),
        "p95_abs_error_per_action": np.quantile(absolute, 0.95, axis=0).tolist(),
        "prediction_abs_max_per_action": np.max(np.abs(prediction), axis=0).tolist(),
    }


def comparison_checks(
    original: dict[str, object],
    masked: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, bool]:
    original_recursive = original["recursive_previous_action"]
    masked_recursive = masked["recursive_previous_action"]
    candidate_recursive = candidate["recursive_previous_action"]
    masked_teacher = masked["teacher_previous_action"]
    candidate_teacher = candidate["teacher_previous_action"]
    recursive_channels = np.asarray(candidate_recursive["mse_per_action"]) <= (
        np.asarray(masked_recursive["mse_per_action"]) * 1.10
    )
    teacher_channels = np.asarray(candidate_teacher["mse_per_action"]) <= (
        np.asarray(masked_teacher["mse_per_action"]) * 1.10
    )
    return {
        "recursive_aggregate_beats_original": candidate_recursive["aggregate_mse"]
        < original_recursive["aggregate_mse"],
        "recursive_aggregate_beats_masked_by_one_percent": candidate_recursive[
            "aggregate_mse"
        ]
        <= masked_recursive["aggregate_mse"] * 0.99,
        "recursive_channels_within_ten_percent_of_masked": bool(
            np.all(recursive_channels)
        ),
        "teacher_aggregate_beats_masked": candidate_teacher["aggregate_mse"]
        < masked_teacher["aggregate_mse"],
        "teacher_channels_within_ten_percent_of_masked": bool(np.all(teacher_channels)),
        "candidate_predictions_within_action_bounds": max(
            candidate_teacher["prediction_abs_max_per_action"]
            + candidate_recursive["prediction_abs_max_per_action"]
        )
        <= 1.0 + 1e-6,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--original-policy", type=Path, required=True)
    parser.add_argument("--masked-policy", type=Path, required=True)
    parser.add_argument("--candidate-policy", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-original-sha256", required=True)
    parser.add_argument("--expected-masked-sha256", required=True)
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = {
        args.dataset: args.expected_dataset_sha256,
        args.original_policy: args.expected_original_sha256,
        args.masked_policy: args.expected_masked_sha256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"input identity mismatch: {path}")
    if args.batch_size <= 0 or args.output.exists():
        raise ValueError("invalid batch size or existing output")
    with np.load(args.dataset, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))
        case_mask = dataset["case_ids"] == args.case
        if not np.any(case_mask):
            raise ValueError("case is absent from dataset")
        if args.case not in metadata["split_cases"]["validation"]:
            raise ValueError("comparison case is not validation-only")
        if not np.all(dataset["split_labels"][case_mask] == VALIDATION_SPLIT):
            raise ValueError("comparison case split labels are invalid")
        observations = dataset["observations"][case_mask].astype(np.float32)
        target = dataset["actions"][case_mask].astype(np.float32)
    policies = {
        "original": args.original_policy,
        "masked": args.masked_policy,
        "candidate": args.candidate_policy,
    }
    results = {}
    for name, path in policies.items():
        model = torch.jit.load(str(path), map_location="cpu").eval()
        results[name] = {
            "teacher_previous_action": error_metrics(
                target, infer_teacher(model, observations, args.batch_size)
            ),
            "recursive_previous_action": error_metrics(
                target, infer_recursive(model, observations)
            ),
        }
    checks = comparison_checks(results["original"], results["masked"], results["candidate"])
    passed = all(checks.values())
    payload = {
        "schema": "cinebotrl_two_wheel_riser_bc_previous_action_comparison_v1",
        "case": args.case,
        "split": "validation",
        "row_count": len(observations),
        "inputs": {
            "dataset": identity(args.dataset),
            **{f"{name}_policy": identity(path) for name, path in policies.items()},
        },
        "results": results,
        "checks": checks,
        "holdout_opened": False,
        "isaac_launched": False,
        "learned_rollout_started": False,
        "ppo_started": False,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
