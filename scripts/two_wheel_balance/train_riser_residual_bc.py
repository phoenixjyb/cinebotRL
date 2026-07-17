#!/usr/bin/env python3
"""Train and gate a bounded residual student from the accepted all-79 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

import numpy as np


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    RiserResidualPolicy,
)


EXPECTED_DATASET_SCHEMA = "cinebotrl_two_wheel_riser_residual_merged_v2"
SPLIT_CODES = {"train": 0, "validation": 1, "holdout": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
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
        required = (
            "observations",
            "actions",
            "case_ids",
            "split_labels",
            "source_index",
        )
        arrays = {name: np.asarray(data[name]) for name in required}
    if metadata.get("schema") != EXPECTED_DATASET_SCHEMA:
        raise ValueError("wrong merged dataset schema")
    if metadata.get("observation_names") != list(OBSERVATION_NAMES):
        raise ValueError("merged observation contract mismatch")
    if metadata.get("observation_contract") != (
        "executed_state_with_execution_time_lookahead_v2"
    ):
        raise ValueError("merged observation contract version mismatch")
    if metadata.get("lookahead_horizons_s") != list(LOOKAHEAD_HORIZONS_S):
        raise ValueError("merged lookahead horizon mismatch")
    observations = arrays["observations"]
    actions = arrays["actions"]
    labels = arrays["split_labels"]
    sources = arrays["source_index"]
    case_ids = arrays["case_ids"]
    if observations.ndim != 2 or observations.shape[1] != len(OBSERVATION_NAMES):
        raise ValueError("observation dimension mismatch")
    if actions.shape != (len(observations), len(ACTION_NAMES)):
        raise ValueError("action dimension mismatch")
    if (
        labels.shape != (len(observations),)
        or sources.shape != labels.shape
        or case_ids.shape != labels.shape
    ):
        raise ValueError("case/split/source dimension mismatch")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("dataset contains non-finite values")
    if np.max(np.abs(actions)) > 1.0 + 1e-6:
        raise ValueError("dataset action exceeds normalized bounds")
    if set(np.unique(labels).tolist()) != set(SPLIT_CODES.values()):
        raise ValueError("dataset must contain train, validation, and holdout rows")
    if metadata.get("row_count") != len(observations):
        raise ValueError("metadata row count mismatch")
    unique_cases = sorted(int(case) for case in np.unique(case_ids))
    if metadata.get("case_count") != len(unique_cases):
        raise ValueError("metadata case count mismatch")
    if metadata.get("trajectory_leakage") is not False:
        raise ValueError("dataset does not attest case-disjoint splits")
    source_splits = {
        int(source): set(labels[sources == source].tolist())
        for source in np.unique(sources)
    }
    leaking = [source for source, splits in source_splits.items() if len(splits) != 1]
    if leaking:
        raise ValueError(f"source leakage across splits: {leaking}")
    source_cases = {
        int(source): set(case_ids[sources == source].tolist())
        for source in np.unique(sources)
    }
    invalid_sources = [source for source, cases in source_cases.items() if len(cases) != 1]
    if invalid_sources or len(source_cases) != len(unique_cases):
        raise ValueError("source indices do not map one-to-one to cases")
    split_cases = metadata.get("split_cases")
    if not isinstance(split_cases, dict) or set(split_cases) != set(SPLIT_CODES):
        raise ValueError("metadata split cases missing")
    for name, code in SPLIT_CODES.items():
        declared = sorted(int(case) for case in split_cases[name])
        observed = sorted(int(case) for case in np.unique(case_ids[labels == code]))
        if declared != observed:
            raise ValueError(f"metadata {name} cases mismatch")
    return metadata, arrays


def case_balanced_mse(
    target: np.ndarray, prediction: np.ndarray, case_ids: np.ndarray
) -> np.ndarray:
    cases = np.unique(case_ids)
    if not len(cases):
        raise ValueError("cannot score an empty split")
    return np.mean(
        [
            np.mean(np.square(prediction[case_ids == case] - target[case_ids == case]), axis=0)
            for case in cases
        ],
        axis=0,
    )


def split_metrics(
    target: np.ndarray, prediction: np.ndarray, case_ids: np.ndarray
) -> dict[str, object]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mse_per_action": np.mean(np.square(error), axis=0).tolist(),
        "case_balanced_mse_per_action": case_balanced_mse(
            target, prediction, case_ids
        ).tolist(),
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
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA-1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("training counts must be positive")
    if not 0.0 < args.minimum_improvement_fraction < 1.0:
        raise ValueError("minimum improvement fraction must be in (0, 1)")
    hidden_sizes = tuple(int(item) for item in args.hidden_sizes.split(",") if item)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
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
    case_ids = arrays["case_ids"]
    masks = {name: labels == code for name, code in SPLIT_CODES.items()}
    train_observations = observations[masks["train"]]
    train_actions = actions[masks["train"]]
    train_case_ids = case_ids[masks["train"]]
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
    unique_train_cases, train_case_counts = np.unique(
        train_case_ids, return_counts=True
    )
    case_weight = {
        int(case): len(train_case_ids) / (len(unique_train_cases) * int(count))
        for case, count in zip(unique_train_cases, train_case_counts, strict=True)
    }
    train_weights = np.asarray(
        [case_weight[int(case)] for case in train_case_ids], dtype=np.float32
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_observations),
            torch.from_numpy(train_actions),
            torch.from_numpy(train_weights),
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
        for obs_batch, action_batch, weight_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            weight_batch = weight_batch.to(device)
            prediction = model(obs_batch)
            per_row_loss = torch.mean(
                torch.square((prediction - action_batch) / loss_scale), dim=1
            )
            loss = torch.mean(per_row_loss * weight_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(obs_batch)
            rows += len(obs_batch)
        validation_prediction = predict(
            model, observations[masks["validation"]], device, args.batch_size
        )
        validation_loss = float(
            np.mean(
                case_balanced_mse(
                    actions[masks["validation"]] / channel_scale,
                    validation_prediction / channel_scale,
                    case_ids[masks["validation"]],
                )
            )
        )
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
    baseline_signal_checks = {}
    for name in ("train", "validation"):
        mask = masks[name]
        target = actions[mask]
        prediction = predict(model, observations[mask], device, args.batch_size)
        split_case_ids = case_ids[mask]
        candidate = split_metrics(target, prediction, split_case_ids)
        baseline = split_metrics(target, np.zeros_like(target), split_case_ids)
        split_results[name] = {"candidate": candidate, "zero_action_baseline": baseline}
        if name != "train":
            candidate_mse = np.asarray(candidate["case_balanced_mse_per_action"])
            baseline_mse = np.asarray(baseline["case_balanced_mse_per_action"])
            baseline_signal = baseline_mse > 1e-10
            baseline_signal_checks[name] = baseline_signal.tolist()
            if name == "validation":
                improvement_checks[name] = (
                    baseline_signal
                    & (
                        candidate_mse
                        <= baseline_mse * (1.0 - args.minimum_improvement_fraction)
                    )
                ).tolist()
    offline_gate_passed = all(
        all(checks) for checks in improvement_checks.values()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    torchscript = None
    if offline_gate_passed:
        checkpoint = args.output_dir / "residual_policy.pt"
        torch.save(
            {
                "schema": "cinebotrl_two_wheel_riser_residual_policy_v2",
                "model_state_dict": model.state_dict(),
                "hidden_sizes": hidden_sizes,
                "observation_names": OBSERVATION_NAMES,
                "action_names": ACTION_NAMES,
                "dataset_sha256": sha256(args.dataset),
                "source_commit": args.source_commit,
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
        "source_commit": args.source_commit,
        "ppo_started": False,
        "learned_rollout_started": False,
        "dataset_schema": metadata["schema"],
        "dataset_sha256": sha256(args.dataset),
        "dataset_case_count": metadata["case_count"],
        "dataset_row_count": metadata["row_count"],
        "hidden_sizes": list(hidden_sizes),
        "observation_count": len(OBSERVATION_NAMES),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "observation_normalization_from_train_only": True,
        "case_balanced_training_loss": True,
        "case_balanced_validation_gate": True,
        "deterministic_algorithms_enabled": True,
        "seed": args.seed,
        "source_group_leakage": False,
        "offline_gate_splits": ["validation"],
        "holdout_used_for_model_selection": False,
        "holdout_metrics_computed": False,
        "action_channel_std_train": action_std.tolist(),
        "split_results": split_results,
        "minimum_improvement_fraction": args.minimum_improvement_fraction,
        "baseline_signal_checks": baseline_signal_checks,
        "improvement_checks": improvement_checks,
        "offline_gate_passed": offline_gate_passed,
        "learned_rollout_authorized": offline_gate_passed,
        "checkpoint": None if checkpoint is None else checkpoint.name,
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "torchscript": None if torchscript is None else torchscript.name,
        "torchscript_sha256": None if torchscript is None else sha256(torchscript),
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if offline_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
