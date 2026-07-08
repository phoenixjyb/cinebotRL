"""Build a row-blend imitation dataset from PPO/SB3 policy outputs.

This is useful when an evaluator-only row blend works better than either source
policy alone.  The script materializes the blended action rows as supervised
labels so `distill_base_assist_head.py` can train only those rows in a real
checkpoint while keeping the rest of the actor unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_indices(raw: str) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise ValueError("at least one action index is required")
    if any(index < 0 for index in indices):
        raise ValueError(f"action indices must be non-negative: {indices}")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy row-blend distillation dataset")
    parser.add_argument("--dataset", required=True, help="Input .npz with observations")
    parser.add_argument("--primary_checkpoint", required=True, help="Anchor policy checkpoint")
    parser.add_argument(
        "--secondary_checkpoint",
        default=None,
        help="Optional secondary policy checkpoint. Omit when --blend_weight is 0.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--action_indices", default="6,7,8", help="Rows to blend and mark as labels")
    parser.add_argument("--blend_weight", type=float, default=0.25, help="Secondary row weight in [0, 1]")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--sample_weight",
        type=float,
        default=1.0,
        help="Constant sample_weight written for all rows; later scripts can rescale per dataset.",
    )
    parser.add_argument("--label", default="", help="Short metadata label for this generated dataset")
    parser.add_argument(
        "--mask_all_rows",
        action="store_true",
        help="Mark every action row valid. Default marks only --action_indices valid.",
    )
    return parser.parse_args()


def load_policy(path: Path, device: str) -> PPO:
    if not path.exists():
        raise FileNotFoundError(path)
    model = PPO.load(str(path), device=device)
    model.policy.set_training_mode(False)
    return model


def predict_actions(model: PPO, observations: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            batch = torch.as_tensor(observations[start : start + batch_size], dtype=torch.float32, device=device)
            dist = model.policy.get_distribution(batch)
            actions = dist.distribution.mean
            outputs.append(actions.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.blend_weight <= 1.0):
        raise ValueError("--blend_weight must be in [0, 1]")
    if args.blend_weight > 0.0 and not args.secondary_checkpoint:
        raise ValueError("--secondary_checkpoint is required when --blend_weight > 0")
    if args.sample_weight < 0.0:
        raise ValueError("--sample_weight must be non-negative")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    data = np.load(dataset_path, allow_pickle=False)
    observations = data["observations"].astype(np.float32)
    if observations.ndim != 2:
        raise ValueError(f"observations must be 2D, got {observations.shape}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    action_indices = parse_indices(args.action_indices)

    primary_path = Path(args.primary_checkpoint)
    secondary_path = Path(args.secondary_checkpoint) if args.secondary_checkpoint else None
    primary = load_policy(primary_path, device)
    primary_actions = predict_actions(primary, observations, int(args.batch_size), device)
    if max(action_indices) >= primary_actions.shape[1]:
        raise ValueError(f"action index out of range for action dim {primary_actions.shape[1]}: {action_indices}")

    target_actions = primary_actions.copy()
    secondary_actions = None
    if secondary_path is not None:
        secondary = load_policy(secondary_path, device)
        secondary_actions = predict_actions(secondary, observations, int(args.batch_size), device)
        if secondary_actions.shape != primary_actions.shape:
            raise ValueError(
                f"primary/secondary action shapes differ: {primary_actions.shape} vs {secondary_actions.shape}"
            )
        rows = np.asarray(action_indices, dtype=np.int64)
        w = float(args.blend_weight)
        target_actions[:, rows] = (1.0 - w) * primary_actions[:, rows] + w * secondary_actions[:, rows]

    target_actions = np.clip(target_actions, -1.0, 1.0).astype(np.float32)
    mask = np.zeros_like(target_actions, dtype=np.float32)
    if args.mask_all_rows:
        mask[:] = 1.0
    else:
        mask[:, np.asarray(action_indices, dtype=np.int64)] = 1.0
    sample_weight = np.full((observations.shape[0],), float(args.sample_weight), dtype=np.float32)

    metadata = {
        "label": args.label,
        "source_dataset": str(dataset_path),
        "primary_checkpoint": str(primary_path),
        "secondary_checkpoint": str(secondary_path) if secondary_path else None,
        "action_indices": action_indices,
        "blend_weight": float(args.blend_weight),
        "rows": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "action_dim": int(target_actions.shape[1]),
        "sample_weight": float(args.sample_weight),
        "mask_all_rows": bool(args.mask_all_rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=target_actions,
        action_valid_mask=mask,
        sample_weight=sample_weight,
        policy_actions=primary_actions.astype(np.float32),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if secondary_actions is not None:
        rows = np.asarray(action_indices, dtype=np.int64)
        delta = target_actions[:, rows] - primary_actions[:, rows]
        print(
            "blend_delta_selected: "
            f"mean_abs={float(np.mean(np.abs(delta))):.6f}, "
            f"max_abs={float(np.max(np.abs(delta))):.6f}"
        )
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
