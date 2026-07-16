#!/usr/bin/env python3
"""Train and gate a bounded residual student from the accepted all-79 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    OBSERVATION_NAMES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    RiserResidualPolicy,
)


EXPECTED_DATASET_SCHEMA = "cinebotrl_two_wheel_riser_residual_merged_v1"
SPLIT_CODES = {"train": 0, "validation": 1, "holdout": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-sizes", default="256,256,128")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--minimum-improvement-fraction", type=float, default=0.05)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        required = ("observations", "actions", "split_labels", "source_index")
        arrays = {name: np.asarray(data[name]) for name in required}
    if metadata.get("schema") != EXPECTED_DATASET_SCHEMA:
        raise ValueError("wrong merged dataset schema")
    observations = arrays["observations"]
    actions = arrays["actions"]
    labels = arrays["split_labels"]
    sources = arrays["source_index"]
    if observations.ndim != 2 or observations.shape[1] != len(OBSERVATION_NAMES):
        raise ValueError("observation dimension mismatch")
    if actions.shape != (len(observations), len(ACTION_NAMES)):
        raise ValueError("action dimension mismatch")
    if labels.shape != (len(observations),) or sources.shape != labels.shape:
        raise ValueError("split/source dimension mismatch")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("dataset contains non-finite values")
    if np.max(np.abs(actions)) > 1.0 + 1e-6:
        raise ValueError("dataset action exceeds normalized bounds")
    if set(np.unique(labels).tolist()) != set(SPLIT_CODES.values()):
        raise ValueError("dataset must contain train, validation, and holdout rows")
    source_splits = {
        int(source): set(labels[sources == source].tolist())
        for source in np.unique(sources)
    }
    leaking = [source for source, splits in source_splits.items() if len(splits) != 1]
    if leaking:
        raise ValueError(f"source leakage across splits: {leaking}")
    return metadata, arrays


def split_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mse_per_action": np.mean(np.square(error), axis=0).tolist(),
        "mae_per_action": np.mean(absolute, axis=0).tolist(),
        "absolute_error_p95_per_action": np.percentile(absolute, 95, axis=0).tolist(),
        "absolute_error_max_per_action": np.max(absolute, axis=0).tolist(),
        "prediction_abs_max_per_action": np.max(np.abs(prediction), axis=0).tolist(),
    }


def predict(
    model: nn.Module, observations: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("training counts must be positive")
    if not 0.0 < args.minimum_improvement_fraction < 1.0:
        raise ValueError("minimum improvement fraction must be in (0, 1)")
    hidden_sizes = tuple(int(item) for item in args.hidden_sizes.split(",") if item)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    metadata, arrays = load_dataset(args.dataset)
    observations = arrays["observations"].astype(np.float32)
    actions = arrays["actions"].astype(np.float32)
    labels = arrays["split_labels"]
    masks = {name: labels == code for name, code in SPLIT_CODES.items()}
    train_observations = observations[masks["train"]]
    train_actions = actions[masks["train"]]
    observation_mean = train_observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    observation_std = train_observations.std(axis=0, dtype=np.float64).astype(np.float32)
    observation_std = np.maximum(observation_std, 1e-4)
    action_std = train_actions.std(axis=0, dtype=np.float64).astype(np.float32)
    channel_scale = np.maximum(action_std, 0.02)
    model = RiserResidualPolicy(
        torch.from_numpy(observation_mean),
        torch.from_numpy(observation_std),
        hidden_sizes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_scale = torch.as_tensor(channel_scale, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_observations), torch.from_numpy(train_actions)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        rows = 0
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            prediction = model(obs_batch)
            loss = torch.mean(torch.square((prediction - action_batch) / loss_scale))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(obs_batch)
            rows += len(obs_batch)
        validation_prediction = predict(
            model, observations[masks["validation"]], device, args.batch_size
        )
        validation_error = (
            validation_prediction - actions[masks["validation"]]
        ) / channel_scale
        validation_loss = float(np.mean(np.square(validation_error)))
        history.append(
            {
                "epoch": epoch,
                "train_balanced_mse": total / rows,
                "validation_balanced_mse": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no candidate")
    model.load_state_dict(best_state)
    split_results = {}
    improvement_checks = {}
    for name, mask in masks.items():
        target = actions[mask]
        prediction = predict(model, observations[mask], device, args.batch_size)
        candidate = split_metrics(target, prediction)
        baseline = split_metrics(target, np.zeros_like(target))
        split_results[name] = {"candidate": candidate, "zero_residual_baseline": baseline}
        if name != "train":
            candidate_mse = np.asarray(candidate["mse_per_action"])
            baseline_mse = np.asarray(baseline["mse_per_action"])
            improvement_checks[name] = (
                candidate_mse
                <= baseline_mse * (1.0 - args.minimum_improvement_fraction)
            ).tolist()
    offline_gate_passed = all(
        all(checks) for checks in improvement_checks.values()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "residual_policy.pt"
    torch.save(
        {
            "schema": "cinebotrl_two_wheel_riser_residual_policy_v1",
            "model_state_dict": model.state_dict(),
            "hidden_sizes": hidden_sizes,
            "observation_names": OBSERVATION_NAMES,
            "action_names": ACTION_NAMES,
            "dataset_sha256": sha256(args.dataset),
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    model = model.cpu().eval()
    scripted = torch.jit.script(model)
    torchscript = args.output_dir / "residual_policy.torchscript.pt"
    scripted.save(str(torchscript))
    report = {
        "schema": "cinebotrl_two_wheel_riser_residual_bc_gate_v1",
        "training_method": "offline_behavior_cloning",
        "ppo_started": False,
        "learned_rollout_started": False,
        "dataset_schema": metadata["schema"],
        "dataset_sha256": sha256(args.dataset),
        "dataset_case_count": metadata["case_count"],
        "dataset_row_count": metadata["row_count"],
        "hidden_sizes": list(hidden_sizes),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "observation_normalization_from_train_only": True,
        "source_group_leakage": False,
        "action_channel_std_train": action_std.tolist(),
        "split_results": split_results,
        "minimum_improvement_fraction": args.minimum_improvement_fraction,
        "improvement_checks": improvement_checks,
        "offline_gate_passed": offline_gate_passed,
        "learned_rollout_authorized": offline_gate_passed,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "torchscript": torchscript.name,
        "torchscript_sha256": sha256(torchscript),
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if offline_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
