"""Evaluate a saved BC/SB3 policy against an offline teacher dataset.

This is a pure Python/SB3 gate: it does not launch Isaac Sim.  It is intended
for fast in-distribution checks before using a BC policy as a PPO warm-start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from stable_baselines3 import PPO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Saved SB3-compatible BC policy .zip.")
    parser.add_argument("--demo_file", required=True, help=".npz with observations/actions/action_valid_mask.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use_action_mask",
        action="store_true",
        help="Use action_valid_mask from the dataset. Without this, all action labels are evaluated.",
    )
    parser.add_argument(
        "--action_indices",
        default=None,
        help="Optional comma-separated action indices to evaluate, e.g. 6,7,8 for base-only.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path for a JSON metrics report.",
    )
    return parser.parse_args()


def parse_indices(raw: str | None, action_dim: int) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        return None
    bad = [idx for idx in indices if idx < 0 or idx >= action_dim]
    if bad:
        raise ValueError(f"action index out of range for dim {action_dim}: {bad}")
    return indices


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def batched_predict(model: PPO, observations: np.ndarray, batch_size: int) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, observations.shape[0], batch_size):
        end = min(start + batch_size, observations.shape[0])
        action, _ = model.predict(observations[start:end], deterministic=True)
        preds.append(np.asarray(action, dtype=np.float32))
    return np.concatenate(preds, axis=0)


def masked_dim_metrics(errors: np.ndarray, mask: np.ndarray) -> list[dict[str, float | int | None]]:
    out: list[dict[str, float | int | None]] = []
    for dim in range(errors.shape[1]):
        selected = mask[:, dim] > 0.0
        count = int(np.count_nonzero(selected))
        if count == 0:
            out.append(
                {
                    "index": dim,
                    "count": 0,
                    "mse": None,
                    "rmse": None,
                    "mae": None,
                    "max_abs": None,
                    "bias": None,
                }
            )
            continue
        values = errors[selected, dim].astype(np.float64)
        out.append(
            {
                "index": dim,
                "count": count,
                "mse": float(np.mean(values * values)),
                "rmse": float(np.sqrt(np.mean(values * values))),
                "mae": float(np.mean(np.abs(values))),
                "max_abs": float(np.max(np.abs(values))),
                "bias": float(np.mean(values)),
            }
        )
    return out


def summarize(errors: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = mask > 0.0
    count = int(np.count_nonzero(selected))
    require(count > 0, "effective action mask selects no labels")
    values = errors[selected].astype(np.float64)
    return {
        "selected_label_count": count,
        "mse": float(np.mean(values * values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "bias": float(np.mean(values)),
        "per_action": masked_dim_metrics(errors, mask),
    }


def source_names(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "source_files" in data:
        return data["source_files"].astype(str)
    if "source_cases" in data:
        return data["source_cases"].astype(str)
    return None


def summarize_by_source(
    data: np.lib.npyio.NpzFile,
    errors: np.ndarray,
    mask: np.ndarray,
    selected_indices: Iterable[int] | None,
) -> list[dict[str, object]]:
    if "source_index" not in data:
        return []
    source_index = data["source_index"].astype(np.int64)
    names = source_names(data)
    action_indices = list(selected_indices) if selected_indices is not None else list(range(errors.shape[1]))
    rows: list[dict[str, object]] = []
    for idx in np.unique(source_index):
        rows_mask = source_index == idx
        action_mask = np.zeros_like(mask, dtype=np.float32)
        action_mask[rows_mask, :] = mask[rows_mask, :]
        action_mask[:, [i for i in range(errors.shape[1]) if i not in action_indices]] = 0.0
        if np.count_nonzero(action_mask) == 0:
            continue
        summary = summarize(errors, action_mask)
        name = str(names[idx]) if names is not None and 0 <= idx < len(names) else str(idx)
        rows.append(
            {
                "source_index": int(idx),
                "source_name": name,
                "samples": int(np.count_nonzero(rows_mask)),
                "selected_label_count": summary["selected_label_count"],
                "rmse": summary["rmse"],
                "mae": summary["mae"],
                "max_abs": summary["max_abs"],
            }
        )
    rows.sort(key=lambda row: float(row["rmse"]), reverse=True)
    return rows


def main() -> int:
    args = parse_args()
    demo_path = Path(args.demo_file)
    policy_path = Path(args.policy)
    require(demo_path.exists(), f"demo file not found: {demo_path}")
    require(policy_path.exists(), f"policy not found: {policy_path}")
    require(args.batch_size > 0, "--batch_size must be positive")

    with np.load(demo_path, allow_pickle=False) as data:
        observations = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        require(observations.ndim == 2, "observations must be 2D")
        require(actions.ndim == 2, "actions must be 2D")
        require(observations.shape[0] == actions.shape[0], "observations/actions row mismatch")
        require(np.isfinite(observations).all(), "observations contain non-finite values")
        require(np.isfinite(actions).all(), "actions contain non-finite values")

        action_dim = actions.shape[1]
        selected_indices = parse_indices(args.action_indices, action_dim)

        if args.use_action_mask:
            require("action_valid_mask" in data, "--use_action_mask requires action_valid_mask in demo file")
            mask = data["action_valid_mask"].astype(np.float32)
            require(mask.shape == actions.shape, "action_valid_mask shape does not match actions")
        else:
            mask = np.ones_like(actions, dtype=np.float32)

        if selected_indices is not None:
            index_mask = np.zeros(action_dim, dtype=np.float32)
            index_mask[selected_indices] = 1.0
            mask = mask * index_mask[None, :]

        model = PPO.load(str(policy_path), device=args.device)
        predictions = batched_predict(model, observations, args.batch_size)
        require(predictions.shape == actions.shape, f"prediction shape {predictions.shape} != actions {actions.shape}")
        require(np.isfinite(predictions).all(), "predictions contain non-finite values")

        errors = predictions - actions
        metrics = summarize(errors, mask)
        metrics.update(
            {
                "policy": str(policy_path),
                "demo_file": str(demo_path),
                "samples": int(actions.shape[0]),
                "obs_dim": int(observations.shape[1]),
                "act_dim": int(action_dim),
                "use_action_mask": bool(args.use_action_mask),
                "action_indices": selected_indices,
                "mask_mean": mask.mean(axis=0).astype(float).tolist(),
                "prediction_mean": predictions.mean(axis=0).astype(float).tolist(),
                "target_mean": actions.mean(axis=0).astype(float).tolist(),
                "source_worst_first": summarize_by_source(data, errors, mask, selected_indices)[:20],
            }
        )

    print("BC offline dataset evaluation")
    print(f"  policy:       {metrics['policy']}")
    print(f"  demo_file:    {metrics['demo_file']}")
    print(f"  samples:      {metrics['samples']}")
    print(f"  obs/action:   {metrics['obs_dim']} / {metrics['act_dim']}")
    print(f"  action mask:  {metrics['mask_mean']}")
    print(f"  labels used:  {metrics['selected_label_count']}")
    print(f"  mse/rmse:     {metrics['mse']:.8f} / {metrics['rmse']:.8f}")
    print(f"  mae/max_abs:  {metrics['mae']:.8f} / {metrics['max_abs']:.8f}")
    print("  per-action:")
    for item in metrics["per_action"]:
        if item["count"] == 0:
            continue
        print(
            "    "
            f"{item['index']}: count={item['count']} "
            f"rmse={item['rmse']:.8f} mae={item['mae']:.8f} "
            f"max_abs={item['max_abs']:.8f} bias={item['bias']:.8f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  wrote:        {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
