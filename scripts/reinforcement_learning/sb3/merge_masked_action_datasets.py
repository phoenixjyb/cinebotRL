"""Merge BC-style masked action datasets produced by distillation collectors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge masked action .npz datasets.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .npz dataset paths.")
    parser.add_argument("--output", required=True, help="Output merged .npz path.")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, np.ndarray | str]:
    data = np.load(path, allow_pickle=False)
    required = ["observations", "actions", "action_valid_mask"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    mask = data["action_valid_mask"].astype(np.float32)
    if observations.ndim != 2 or actions.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"{path} must contain 2D observations/actions/mask arrays")
    if observations.shape[0] != actions.shape[0] or actions.shape != mask.shape:
        raise ValueError(
            f"{path} shape mismatch: observations={observations.shape}, "
            f"actions={actions.shape}, mask={mask.shape}"
        )
    if not np.isfinite(observations).all() or not np.isfinite(actions).all() or not np.isfinite(mask).all():
        raise ValueError(f"{path} contains non-finite values")

    metadata = data["metadata"].item() if "metadata" in data else "{}"
    sample_weight = data["sample_weight"].astype(np.float32) if "sample_weight" in data else None
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != observations.shape[0]:
            raise ValueError(
                f"{path} sample_weight must be 1D and match observations: "
                f"sample_weight={sample_weight.shape}, observations={observations.shape}"
            )
        if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0.0):
            raise ValueError(f"{path} sample_weight contains invalid values")
    return {
        "path": str(path),
        "observations": observations,
        "actions": actions,
        "action_valid_mask": mask,
        "sample_weight": sample_weight,
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    datasets = [load_dataset(Path(path)) for path in args.inputs]
    obs_dims = {dataset["observations"].shape[1] for dataset in datasets}  # type: ignore[index,union-attr]
    act_dims = {dataset["actions"].shape[1] for dataset in datasets}  # type: ignore[index,union-attr]
    if len(obs_dims) != 1 or len(act_dims) != 1:
        raise ValueError(f"incompatible dimensions: obs={sorted(obs_dims)}, act={sorted(act_dims)}")

    observations = np.concatenate([dataset["observations"] for dataset in datasets], axis=0)  # type: ignore[list-item]
    actions = np.concatenate([dataset["actions"] for dataset in datasets], axis=0)  # type: ignore[list-item]
    mask = np.concatenate([dataset["action_valid_mask"] for dataset in datasets], axis=0)  # type: ignore[list-item]
    has_any_weight = any(dataset["sample_weight"] is not None for dataset in datasets)
    sample_weight = None
    if has_any_weight:
        sample_weight = np.concatenate(
            [
                dataset["sample_weight"]  # type: ignore[list-item]
                if dataset["sample_weight"] is not None
                else np.ones(dataset["observations"].shape[0], dtype=np.float32)  # type: ignore[index,union-attr]
                for dataset in datasets
            ],
            axis=0,
        )

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(observations.shape[0])
        observations = observations[order]
        actions = actions[order]
        mask = mask[order]
        if sample_weight is not None:
            sample_weight = sample_weight[order]

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [
            {
                "path": dataset["path"],
                "samples": int(dataset["observations"].shape[0]),  # type: ignore[index,union-attr]
                "metadata": dataset["metadata"],
            }
            for dataset in datasets
        ],
        "total_samples": int(observations.shape[0]),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "valid_action_counts": mask.sum(axis=0).astype(float).tolist(),
        "has_sample_weight": sample_weight is not None,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "observations": observations,
        "actions": actions,
        "action_valid_mask": mask,
        "metadata": json.dumps(metadata, indent=2),
    }
    if sample_weight is not None:
        output_payload["sample_weight"] = sample_weight.astype(np.float32)
    np.savez_compressed(output, **output_payload)
    print(f"saved: {output}")
    print(f"observations: {observations.shape}, actions: {actions.shape}")
    print(f"valid action counts: {metadata['valid_action_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
