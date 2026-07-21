#!/usr/bin/env python3
"""Rank training cases similar to a validation reference without opening data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


SPLIT_LABELS = {"train": 0, "validation": 1, "holdout": 2}
SCORE_WEIGHTS = {
    "action_quantile_rms": 0.40,
    "state_mean_rms": 0.25,
    "state_std_rms": 0.25,
    "duration_log_abs": 0.10,
}


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _case_signature(
    payload: dict[str, np.ndarray],
    case: int,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    effective_indices: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    mask = payload["case_ids"] == case
    observations = (
        payload["observations"][mask] - observation_mean
    ) / observation_std
    observations = observations[:, effective_indices]
    actions = payload["actions"][mask]
    elapsed = payload["elapsed_time_s"][mask]
    return {
        "count": int(np.count_nonzero(mask)),
        "duration_s": float(elapsed[-1] - elapsed[0]),
        "action_quantiles": np.quantile(actions, [0.05, 0.50, 0.95], axis=0),
        "state_mean": np.mean(observations, axis=0),
        "state_std": np.std(observations, axis=0),
    }


def rank_cases(
    payload: dict[str, np.ndarray],
    metadata: dict[str, object],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    observation_mask: np.ndarray,
    *,
    reference_case: int,
    maximum_candidates: int,
) -> dict[str, object]:
    split_cases = metadata["split_cases"]
    train_cases = [int(case) for case in split_cases["train"]]
    validation_cases = [int(case) for case in split_cases["validation"]]
    holdout_cases = [int(case) for case in split_cases["holdout"]]
    if reference_case not in validation_cases:
        raise ValueError("reference case must remain in the validation split")
    if maximum_candidates < 1 or maximum_candidates > len(train_cases):
        raise ValueError("maximum candidates is outside the training case count")
    if set(train_cases) & (set(validation_cases) | set(holdout_cases)):
        raise ValueError("dataset split cases overlap")
    effective_indices = np.flatnonzero(observation_mask != 0.0)
    reference = _case_signature(
        payload,
        reference_case,
        observation_mean,
        observation_std,
        effective_indices,
    )
    ranked = []
    for case in train_cases:
        candidate = _case_signature(
            payload,
            case,
            observation_mean,
            observation_std,
            effective_indices,
        )
        components = {
            "action_quantile_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            candidate["action_quantiles"]
                            - reference["action_quantiles"]
                        )
                    )
                )
            ),
            "state_mean_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(candidate["state_mean"] - reference["state_mean"])
                    )
                )
            ),
            "state_std_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(candidate["state_std"] - reference["state_std"])
                    )
                )
            ),
            "duration_log_abs": abs(
                math.log(
                    max(float(candidate["duration_s"]), 1e-9)
                    / max(float(reference["duration_s"]), 1e-9)
                )
            ),
        }
        score = sum(SCORE_WEIGHTS[name] * value for name, value in components.items())
        ranked.append(
            {
                "case": case,
                "split": "train",
                "score": score,
                "components": components,
                "sample_count": candidate["count"],
                "duration_s": candidate["duration_s"],
            }
        )
    ranked.sort(key=lambda item: (item["score"], item["case"]))
    selected = ranked[:maximum_candidates]
    selected_ids = [item["case"] for item in selected]
    if not set(selected_ids).issubset(train_cases):
        raise RuntimeError("non-training case entered the proposal")
    return {
        "schema": "cinebotrl_two_wheel_riser_dagger_training_case_ranking_v1",
        "reference_case": reference_case,
        "reference_split": "validation",
        "reference_sample_count": reference["count"],
        "reference_duration_s": reference["duration_s"],
        "score_weights": SCORE_WEIGHTS,
        "effective_observation_count": len(effective_indices),
        "previous_action_channels_effective": bool(
            np.any(observation_mask[23:26] != 0.0)
        ),
        "train_cases": train_cases,
        "validation_cases": validation_cases,
        "holdout_cases": holdout_cases,
        "ranked_training_cases": ranked,
        "selected_candidate_count": len(selected),
        "selected_training_cases": selected_ids,
        "selected_candidates": selected,
        "validation_cases_excluded_from_training_proposal": True,
        "holdout_cases_excluded_from_training_proposal": True,
        "runtime_authorized": False,
        "shadow_teacher_capture_started": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-case", type=int, default=4)
    parser.add_argument("--maximum-candidates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in (
                "observations",
                "actions",
                "case_ids",
                "elapsed_time_s",
                "split_labels",
            )
        }
    model = torch.jit.load(str(args.policy), map_location="cpu")
    report = rank_cases(
        payload,
        metadata,
        model.observation_mean.detach().cpu().numpy(),
        model.observation_std.detach().cpu().numpy(),
        model.observation_mask.detach().cpu().numpy(),
        reference_case=args.reference_case,
        maximum_candidates=args.maximum_candidates,
    )
    report["inputs"] = {
        "dataset": _identity(args.dataset),
        "policy": _identity(args.policy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
