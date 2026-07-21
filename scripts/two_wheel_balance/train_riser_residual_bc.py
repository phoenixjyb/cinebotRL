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
    PREVIOUS_ACTION_INDICES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    ATTENUATED_PREVIOUS_ACTION_POLICY_ARCHITECTURE,
    MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE,
    POLICY_ARCHITECTURE,
    RiserResidualPolicy,
)


SUPPORTED_DATASET_SCHEMAS = {
    "cinebotrl_two_wheel_riser_residual_merged_v2",
    "cinebotrl_two_wheel_riser_residual_merged_v3",
}
INITIAL_TEACHER_DATASET_SCHEMA = "cinebotrl_two_wheel_riser_residual_merged_v3"
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
    parser.add_argument("--state-hidden-sizes", default="128,128")
    parser.add_argument("--lookahead-hidden-sizes", default="64,64")
    parser.add_argument("--fusion-hidden-sizes", default="256,128")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--minimum-improvement-fraction", type=float, default=0.05)
    parser.add_argument(
        "--mask-previous-action-observations",
        action="store_true",
        help="Mask teacher-forced previous-action channels inside the policy.",
    )
    parser.add_argument(
        "--scheduled-previous-action-max-probability",
        type=float,
        default=0.0,
        help="Maximum probability of using the prior policy prediction in sequence BC.",
    )
    parser.add_argument("--scheduled-previous-action-warmup-epochs", type=int, default=0)
    parser.add_argument("--scheduled-previous-action-ramp-epochs", type=int, default=1)
    parser.add_argument("--scheduled-sequence-length", type=int, default=32)
    parser.add_argument("--scheduled-sequence-batch-size", type=int, default=256)
    parser.add_argument(
        "--recursive-validation-weight",
        type=float,
        default=0.5,
        help="Weight of recursive-window loss used for scheduled-BC model selection.",
    )
    parser.add_argument(
        "--previous-action-observation-gain",
        type=float,
        default=1.0,
        help="Fixed gain on normalized previous-action channels inside the policy.",
    )
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
        schema = metadata.get("schema")
        required = [
            "observations",
            "actions",
            "case_ids",
            "split_labels",
            "source_index",
        ]
        if schema == INITIAL_TEACHER_DATASET_SCHEMA:
            required.append("action_valid_mask")
        arrays = {name: np.asarray(data[name]) for name in required}
    if schema not in SUPPORTED_DATASET_SCHEMAS:
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
    if schema == INITIAL_TEACHER_DATASET_SCHEMA:
        scales = np.asarray(metadata.get("action_scales"), dtype=np.float64)
        valid_mask = arrays["action_valid_mask"]
        v3_checks = {
            "dataset_admitted": metadata.get("dataset_admission_passed") is True,
            "valid_for_initialization": metadata.get("valid_for_bc_initialization")
            is True,
            "training_closed": metadata.get("bc_authorized") is False
            and metadata.get("ppo_authorized") is False
            and metadata.get("training_started") is False,
            "frozen_scales": scales.shape == (3,)
            and np.isfinite(scales).all()
            and bool(np.all(scales > 0.0)),
            "zero_clip": metadata.get("action_clip_ratio") == [0.0, 0.0, 0.0],
            "previous_contract": metadata.get("previous_action_contract")
            == "previous_normalized_teacher_action_v1"
            and metadata.get("previous_action_rebuilt") is True,
            "action_mask_shape": valid_mask.shape == actions.shape,
            "all_actions_valid": valid_mask.shape == actions.shape
            and bool(np.all(valid_mask == 1.0)),
            "no_source_actions": metadata.get("source_action_labels_used") is False,
            "no_physical_gimbal_actions": metadata.get(
                "physical_gimbal_labels_used_as_actions"
            )
            is False,
        }
        if not all(v3_checks.values()):
            raise ValueError(f"v3 dataset admission failed: {v3_checks}")
        for case in unique_cases:
            mask = case_ids == case
            case_actions = actions[mask]
            previous = observations[mask][:, PREVIOUS_ACTION_INDICES]
            if not np.allclose(previous[0], 0.0, atol=1e-12) or not np.allclose(
                previous[1:], case_actions[:-1], atol=1e-7
            ):
                raise ValueError(f"v3 previous-action recurrence failed for case {case}")
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


def scheduled_sampling_probability(
    epoch: int, *, maximum: float, warmup_epochs: int, ramp_epochs: int
) -> float:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("scheduled-sampling maximum must be in [0, 1]")
    if warmup_epochs < 0 or ramp_epochs <= 0:
        raise ValueError("scheduled-sampling warmup/ramp is invalid")
    progress = max(0.0, min(1.0, (epoch - warmup_epochs) / ramp_epochs))
    return maximum * progress


def build_sequence_windows(case_ids: np.ndarray, sequence_length: int) -> np.ndarray:
    if case_ids.ndim != 1 or not len(case_ids):
        raise ValueError("case IDs must be a non-empty vector")
    if sequence_length < 2:
        raise ValueError("sequence length must be at least two")
    windows: list[np.ndarray] = []
    for case in np.unique(case_ids):
        indices = np.flatnonzero(case_ids == case)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"case {int(case)} rows are not contiguous")
        for start in range(0, len(indices), sequence_length):
            window = np.full(sequence_length, -1, dtype=np.int64)
            chunk = indices[start : start + sequence_length]
            window[: len(chunk)] = chunk
            windows.append(window)
    return np.stack(windows)


def predict_recursive_previous_action_windows(
    model: nn.Module,
    observations: np.ndarray,
    case_ids: np.ndarray,
    device: torch.device,
    sequence_length: int,
    window_batch_size: int,
) -> np.ndarray:
    """Predict with policy-generated previous actions inside bounded case windows."""
    windows = build_sequence_windows(case_ids, sequence_length)
    prediction = np.empty((len(observations), len(ACTION_NAMES)), dtype=np.float32)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), window_batch_size):
            batch_indices = windows[start : start + window_batch_size]
            valid = batch_indices >= 0
            safe_indices = np.maximum(batch_indices, 0)
            observation_batch = torch.as_tensor(
                observations[safe_indices], dtype=torch.float32, device=device
            )
            previous_prediction = torch.zeros(
                (len(batch_indices), len(ACTION_NAMES)),
                dtype=torch.float32,
                device=device,
            )
            for step in range(sequence_length):
                step_observation = observation_batch[:, step].clone()
                if step > 0:
                    step_observation[:, PREVIOUS_ACTION_INDICES] = previous_prediction
                step_prediction = model(step_observation)
                active = valid[:, step]
                if np.any(active):
                    active_tensor = torch.as_tensor(active, device=device)
                    prediction[batch_indices[active, step]] = (
                        step_prediction[active_tensor].cpu().numpy()
                    )
                previous_prediction = step_prediction
    model.train(was_training)
    return prediction


def train_scheduled_sampling_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    windows: np.ndarray,
    device: torch.device,
    loss_scale: torch.Tensor,
    probability: float,
    batch_size: int,
    shuffle_generator: torch.Generator,
    sampling_generator: torch.Generator,
) -> tuple[float, int, int]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows)),
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        drop_last=False,
    )
    model.train()
    total_loss = 0.0
    total_rows = 0
    sampled_rows = 0
    for (index_batch,) in loader:
        valid = index_batch >= 0
        safe_indices = torch.clamp(index_batch, min=0)
        observation_batch = torch.from_numpy(observations)[safe_indices].to(device)
        action_batch = torch.from_numpy(actions)[safe_indices].to(device)
        weight_batch = torch.from_numpy(weights)[safe_indices].to(device)
        decisions = (
            torch.rand(index_batch.shape, generator=sampling_generator) < probability
        ).to(device)
        valid = valid.to(device)
        previous_prediction = torch.zeros(
            (len(index_batch), len(ACTION_NAMES)), dtype=torch.float32, device=device
        )
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        valid_count = 0
        for step in range(index_batch.shape[1]):
            step_observation = observation_batch[:, step].clone()
            replace = valid[:, step] & decisions[:, step] & (step > 0)
            if step > 0:
                step_observation[:, PREVIOUS_ACTION_INDICES] = torch.where(
                    replace[:, None],
                    previous_prediction.detach(),
                    step_observation[:, PREVIOUS_ACTION_INDICES],
                )
            step_prediction = model(step_observation)
            per_row_loss = torch.mean(
                torch.square((step_prediction - action_batch[:, step]) / loss_scale),
                dim=1,
            )
            active = valid[:, step]
            loss_sum = loss_sum + torch.sum(
                per_row_loss[active] * weight_batch[:, step][active]
            )
            valid_count += int(active.sum().item())
            sampled_rows += int(replace.sum().item())
            previous_prediction = step_prediction
        loss = loss_sum / valid_count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * valid_count
        total_rows += valid_count
    return total_loss / total_rows, total_rows, sampled_rows


def main() -> int:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA-1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("training counts must be positive")
    if not 0.0 < args.minimum_improvement_fraction < 1.0:
        raise ValueError("minimum improvement fraction must be in (0, 1)")
    scheduled_sampling_enabled = args.scheduled_previous_action_max_probability > 0.0
    if not 0.0 <= args.previous_action_observation_gain <= 1.0:
        raise ValueError("previous-action observation gain must be in [0, 1]")
    if args.mask_previous_action_observations and scheduled_sampling_enabled:
        raise ValueError("masking and scheduled previous-action sampling are exclusive")
    if args.mask_previous_action_observations and args.previous_action_observation_gain != 1.0:
        raise ValueError("masking and previous-action attenuation are exclusive")
    if scheduled_sampling_enabled and args.previous_action_observation_gain != 1.0:
        raise ValueError("scheduled sampling and previous-action attenuation are exclusive")
    if not 0.0 <= args.scheduled_previous_action_max_probability <= 1.0:
        raise ValueError("scheduled previous-action maximum must be in [0, 1]")
    if (
        args.scheduled_previous_action_warmup_epochs < 0
        or args.scheduled_previous_action_ramp_epochs <= 0
        or args.scheduled_sequence_length < 2
        or args.scheduled_sequence_batch_size <= 0
    ):
        raise ValueError("scheduled previous-action configuration is invalid")
    if not 0.0 <= args.recursive_validation_weight <= 1.0:
        raise ValueError("recursive validation weight must be in [0, 1]")
    state_hidden_sizes = tuple(
        int(item) for item in args.state_hidden_sizes.split(",") if item
    )
    lookahead_hidden_sizes = tuple(
        int(item) for item in args.lookahead_hidden_sizes.split(",") if item
    )
    fusion_hidden_sizes = tuple(
        int(item) for item in args.fusion_hidden_sizes.split(",") if item
    )
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
    masked_observation_indices = (
        PREVIOUS_ACTION_INDICES if args.mask_previous_action_observations else ()
    )
    if masked_observation_indices:
        policy_architecture = MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE
    elif args.previous_action_observation_gain < 1.0:
        policy_architecture = ATTENUATED_PREVIOUS_ACTION_POLICY_ARCHITECTURE
    else:
        policy_architecture = POLICY_ARCHITECTURE
    model = RiserResidualPolicy(
        torch.from_numpy(observation_mean),
        torch.from_numpy(observation_std),
        state_hidden_sizes,
        lookahead_hidden_sizes,
        fusion_hidden_sizes,
        masked_observation_indices,
        args.previous_action_observation_gain,
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
    loader = None
    train_windows = None
    sampling_generator = torch.Generator().manual_seed(args.seed + 1)
    if scheduled_sampling_enabled:
        train_windows = build_sequence_windows(
            train_case_ids, args.scheduled_sequence_length
        )
    else:
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
        probability = scheduled_sampling_probability(
            epoch,
            maximum=args.scheduled_previous_action_max_probability,
            warmup_epochs=args.scheduled_previous_action_warmup_epochs,
            ramp_epochs=args.scheduled_previous_action_ramp_epochs,
        )
        sampled_rows = 0
        if scheduled_sampling_enabled:
            assert train_windows is not None
            train_loss, rows, sampled_rows = train_scheduled_sampling_epoch(
                model,
                optimizer,
                train_observations,
                train_actions,
                train_weights,
                train_windows,
                device,
                loss_scale,
                probability,
                args.scheduled_sequence_batch_size,
                generator,
                sampling_generator,
            )
        else:
            assert loader is not None
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
            train_loss = total / rows
        validation_prediction = predict(
            model, observations[masks["validation"]], device, args.batch_size
        )
        teacher_validation_loss = float(
            np.mean(
                case_balanced_mse(
                    actions[masks["validation"]] / channel_scale,
                    validation_prediction / channel_scale,
                    case_ids[masks["validation"]],
                )
            )
        )
        recursive_validation_loss = teacher_validation_loss
        if scheduled_sampling_enabled:
            recursive_validation_prediction = predict_recursive_previous_action_windows(
                model,
                observations[masks["validation"]],
                case_ids[masks["validation"]],
                device,
                args.scheduled_sequence_length,
                args.scheduled_sequence_batch_size,
            )
            recursive_validation_loss = float(
                np.mean(
                    case_balanced_mse(
                        actions[masks["validation"]] / channel_scale,
                        recursive_validation_prediction / channel_scale,
                        case_ids[masks["validation"]],
                    )
                )
            )
        validation_loss = (
            (1.0 - args.recursive_validation_weight) * teacher_validation_loss
            + args.recursive_validation_weight * recursive_validation_loss
            if scheduled_sampling_enabled
            else teacher_validation_loss
        )
        history.append(
            {
                "epoch": epoch,
                "train_balanced_mse": train_loss,
                "validation_balanced_mse": validation_loss,
                "teacher_validation_balanced_mse": teacher_validation_loss,
                "recursive_validation_balanced_mse": recursive_validation_loss,
                "scheduled_previous_action_probability": probability,
                "scheduled_previous_action_rows": sampled_rows,
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
    recursive_split_results = {}
    improvement_checks = {}
    recursive_improvement_checks = {}
    baseline_signal_checks = {}
    for name in ("train", "validation"):
        mask = masks[name]
        target = actions[mask]
        prediction = predict(model, observations[mask], device, args.batch_size)
        split_case_ids = case_ids[mask]
        candidate = split_metrics(target, prediction, split_case_ids)
        baseline = split_metrics(target, np.zeros_like(target), split_case_ids)
        split_results[name] = {"candidate": candidate, "zero_action_baseline": baseline}
        if scheduled_sampling_enabled:
            recursive_prediction = predict_recursive_previous_action_windows(
                model,
                observations[mask],
                split_case_ids,
                device,
                args.scheduled_sequence_length,
                args.scheduled_sequence_batch_size,
            )
            recursive_candidate = split_metrics(
                target, recursive_prediction, split_case_ids
            )
            recursive_split_results[name] = {
                "candidate": recursive_candidate,
                "zero_action_baseline": baseline,
            }
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
                if scheduled_sampling_enabled:
                    recursive_mse = np.asarray(
                        recursive_split_results[name]["candidate"][
                            "case_balanced_mse_per_action"
                        ]
                    )
                    recursive_improvement_checks[name] = (
                        baseline_signal
                        & (
                            recursive_mse
                            <= baseline_mse * (1.0 - args.minimum_improvement_fraction)
                        )
                    ).tolist()
    offline_gate_passed = all(
        all(checks) for checks in improvement_checks.values()
    ) and (
        not scheduled_sampling_enabled
        or all(all(checks) for checks in recursive_improvement_checks.values())
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    torchscript = None
    if offline_gate_passed:
        checkpoint = args.output_dir / "residual_policy.pt"
        torch.save(
            {
                "schema": "cinebotrl_two_wheel_riser_residual_policy_v2",
                "policy_architecture": policy_architecture,
                "model_state_dict": model.state_dict(),
                "state_hidden_sizes": state_hidden_sizes,
                "lookahead_hidden_sizes": lookahead_hidden_sizes,
                "fusion_hidden_sizes": fusion_hidden_sizes,
                "observation_names": OBSERVATION_NAMES,
                "action_names": ACTION_NAMES,
                "masked_observation_indices": list(masked_observation_indices),
                "scheduled_previous_action_enabled": scheduled_sampling_enabled,
                "scheduled_sequence_length": args.scheduled_sequence_length,
                "previous_action_observation_gain": (
                    args.previous_action_observation_gain
                ),
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
        "schema": "cinebotrl_two_wheel_riser_residual_bc_gate_v2",
        "training_method": (
            "offline_behavior_cloning_deterministic_scheduled_sampling"
            if scheduled_sampling_enabled
            else "offline_behavior_cloning"
        ),
        "policy_architecture": policy_architecture,
        "source_commit": args.source_commit,
        "ppo_started": False,
        "learned_rollout_started": False,
        "dataset_schema": metadata["schema"],
        "dataset_sha256": sha256(args.dataset),
        "dataset_case_count": metadata["case_count"],
        "dataset_row_count": metadata["row_count"],
        "state_hidden_sizes": list(state_hidden_sizes),
        "lookahead_hidden_sizes": list(lookahead_hidden_sizes),
        "fusion_hidden_sizes": list(fusion_hidden_sizes),
        "observation_count": len(OBSERVATION_NAMES),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "observation_normalization_from_train_only": True,
        "masked_observation_indices": list(masked_observation_indices),
        "previous_action_observation_gain": args.previous_action_observation_gain,
        "previous_action_observation_contract": (
            "masked_after_normalization_v1"
            if masked_observation_indices
            else (
                "deterministic_scheduled_policy_previous_action_v1"
                if scheduled_sampling_enabled
                else (
                    "attenuated_after_normalization_v1"
                    if args.previous_action_observation_gain < 1.0
                    else "teacher_previous_action_v1"
                )
            )
        ),
        "scheduled_previous_action_enabled": scheduled_sampling_enabled,
        "scheduled_previous_action_max_probability": (
            args.scheduled_previous_action_max_probability
        ),
        "scheduled_previous_action_warmup_epochs": (
            args.scheduled_previous_action_warmup_epochs
        ),
        "scheduled_previous_action_ramp_epochs": (
            args.scheduled_previous_action_ramp_epochs
        ),
        "scheduled_sequence_length": args.scheduled_sequence_length,
        "scheduled_sequence_batch_size": args.scheduled_sequence_batch_size,
        "recursive_validation_weight": args.recursive_validation_weight,
        "scheduled_sampling_detaches_previous_prediction": True,
        "sequence_windows_cross_case_boundaries": False,
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
        "recursive_previous_action_split_results": recursive_split_results,
        "minimum_improvement_fraction": args.minimum_improvement_fraction,
        "baseline_signal_checks": baseline_signal_checks,
        "improvement_checks": improvement_checks,
        "recursive_improvement_checks": recursive_improvement_checks,
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
