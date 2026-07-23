"""CPU-only preflight adapter for projection-aware corrective BC."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from .riser_model_based_bc_loss import (
    MODEL_BASED_PROJECTED_BC_LOSS,
    REQUESTED_OUTPUT_SLEW_REGULARIZATION,
    ModelBasedProjectedBCLoss,
)
from .riser_model_based_corrective_training_dataset import (
    MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
    validate_training_dataset,
)
from .riser_residual_dataset import PREVIOUS_ACTION_INDICES

MODEL_BASED_CORRECTIVE_BC_PREFLIGHT_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_bc_preflight_v1"
)
MODEL_BASED_CORRECTIVE_BC_ADAPTER_CONTRACT = (
    "projection_aware_effective_label_bc_adapter_v1"
)
MODEL_BASED_CORRECTIVE_BC_OPTIMIZER_CONTRACT = (
    "exact_case_balanced_projection_aware_gradient_accumulation_v1"
)
MODEL_BASED_CORRECTIVE_BC_VALIDATION_CONTRACT = (
    "projected_effective_action_case_balanced_recursive_validation_v2"
)
MODEL_BASED_CORRECTIVE_BC_RECURSIVE_VALIDATION_CONTRACT = (
    "case_reset_recursive_effective_action_validation_v1"
)


def _previous_requested_actions(
    requested_actions: np.ndarray,
    previous_row_index: np.ndarray,
    transition_valid: np.ndarray,
) -> np.ndarray:
    requested = np.asarray(requested_actions, dtype=np.float32)
    previous = np.asarray(previous_row_index, dtype=np.int64)
    valid = np.asarray(transition_valid)
    if (
        requested.ndim != 2
        or requested.shape[1] != 3
        or previous.shape != (len(requested),)
        or valid.shape != (len(requested),)
        or valid.dtype != np.bool_
    ):
        raise ValueError("projection-aware previous-request input mismatch")
    result = np.zeros_like(requested)
    result[valid] = requested[previous[valid]]
    return result


def build_projection_aware_split(
    payload: Mapping[str, np.ndarray],
    *,
    split_code: int,
) -> dict[str, np.ndarray]:
    """Build a case-safe training split with explicit predecessor observations."""
    observations = np.asarray(payload["observations"], dtype=np.float32)
    case_ids = np.asarray(payload["case_ids"], dtype=np.int64)
    labels = np.asarray(payload["split_labels"], dtype=np.int64)
    previous = np.asarray(payload["previous_row_index"], dtype=np.int64)
    transition_valid = np.asarray(payload["transition_valid"])
    selected = np.flatnonzero(labels == split_code)
    if not len(selected):
        raise ValueError("projection-aware split is empty")
    selected_previous = previous[selected]
    selected_valid = transition_valid[selected]
    if (
        selected_valid.dtype != np.bool_
        or np.any(selected_previous[selected_valid] < 0)
        or np.any(selected_previous[selected_valid] >= len(observations))
        or not np.array_equal(
            labels[selected_previous[selected_valid]],
            np.full(np.sum(selected_valid), split_code, dtype=labels.dtype),
        )
        or not np.array_equal(
            case_ids[selected_previous[selected_valid]],
            case_ids[selected][selected_valid],
        )
    ):
        raise ValueError("projection-aware predecessor crosses a case or split")
    previous_observations = np.zeros_like(observations[selected])
    previous_observations[selected_valid] = observations[
        selected_previous[selected_valid]
    ]
    result = {
        "observations": observations[selected],
        "previous_observations": previous_observations,
        "effective_target_actions": np.asarray(
            payload["actions"], dtype=np.float32
        )[selected],
        "model_based_commands": np.asarray(
            payload["model_based_commands"], dtype=np.float32
        )[selected],
        "delta_time_s": np.asarray(payload["delta_time_s"], dtype=np.float32)[
            selected
        ],
        "transition_valid": selected_valid,
        "sample_weights": np.asarray(
            payload["case_balanced_sample_weights"], dtype=np.float32
        )[selected],
        "case_ids": case_ids[selected],
        "source_row_index": selected.astype(np.int64),
    }
    row_count = len(selected)
    if any(len(value) != row_count for value in result.values()):
        raise ValueError("projection-aware split array length mismatch")
    return result


def _as_tensor(
    values: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.from_numpy(values)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device)


def _predict_current_and_previous(
    model: torch.nn.Module,
    split: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = _as_tensor(
        np.asarray(split["observations"])[indices],
        device=device,
        dtype=torch.float32,
    )
    previous_observations = _as_tensor(
        np.asarray(split["previous_observations"])[indices],
        device=device,
        dtype=torch.float32,
    )
    transition_valid = _as_tensor(
        np.asarray(split["transition_valid"])[indices],
        device=device,
    )
    requested = model(observations)
    previous_requested = model(previous_observations)
    previous_requested = torch.where(
        transition_valid[:, None],
        previous_requested,
        torch.zeros_like(previous_requested),
    )
    return requested, previous_requested


def train_projection_aware_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    split: Mapping[str, np.ndarray],
    loss: ModelBasedProjectedBCLoss,
    *,
    device: torch.device,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, float | int]:
    """Apply one exact case-balanced gradient-accumulation optimizer step."""
    row_count = len(split["observations"])
    if row_count <= 0 or batch_size <= 0:
        raise ValueError("projection-aware training counts must be positive")
    sample_weights = np.asarray(split["sample_weights"], dtype=np.float64)
    transition_valid = np.asarray(split["transition_valid"])
    total_weight = float(np.sum(sample_weights))
    total_transition_weight = float(
        np.sum(sample_weights * transition_valid.astype(np.float64))
    )
    if total_weight <= 0.0 or total_transition_weight <= 0.0:
        raise ValueError("projection-aware split has invalid aggregate weights")
    order = torch.randperm(row_count, generator=generator).numpy()
    optimizer.zero_grad(set_to_none=True)
    pointwise_total = 0.0
    slew_total = 0.0
    clipped_rows = 0
    model.train()
    for start in range(0, row_count, batch_size):
        indices = order[start : start + batch_size]
        requested, previous_requested = _predict_current_and_previous(
            model,
            split,
            indices,
            device=device,
        )
        batch_weights = _as_tensor(
            np.asarray(split["sample_weights"])[indices],
            device=device,
            dtype=torch.float32,
        )
        batch_valid = _as_tensor(
            np.asarray(split["transition_valid"])[indices],
            device=device,
        )
        (
            _,
            pointwise_loss,
            slew_loss,
            _,
            command_clipped,
        ) = loss(
            _as_tensor(
                np.asarray(split["model_based_commands"])[indices],
                device=device,
                dtype=torch.float32,
            ),
            requested,
            _as_tensor(
                np.asarray(split["effective_target_actions"])[indices],
                device=device,
                dtype=torch.float32,
            ),
            previous_requested,
            _as_tensor(
                np.asarray(split["delta_time_s"])[indices],
                device=device,
                dtype=torch.float32,
            ),
            batch_valid,
            batch_weights,
        )
        batch_weight = float(np.sum(sample_weights[indices]))
        batch_transition_weight = float(
            np.sum(
                sample_weights[indices]
                * transition_valid[indices].astype(np.float64)
            )
        )
        scaled_loss = pointwise_loss * (batch_weight / total_weight)
        if batch_transition_weight > 0.0:
            scaled_loss = scaled_loss + (
                loss.slew_regularization_weight
                * slew_loss
                * (batch_transition_weight / total_transition_weight)
            )
        scaled_loss.backward()
        pointwise_total += float(pointwise_loss.item()) * batch_weight
        slew_total += float(slew_loss.item()) * batch_transition_weight
        clipped_rows += int(torch.any(command_clipped, dim=1).sum().item())
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
    if not np.isfinite(gradient_norm):
        raise ValueError("projection-aware optimizer produced a non-finite gradient")
    optimizer.step()
    pointwise = pointwise_total / total_weight
    slew = slew_total / total_transition_weight
    return {
        "row_count": row_count,
        "batch_count": int((row_count + batch_size - 1) // batch_size),
        "loss_total": pointwise + loss.slew_regularization_weight * slew,
        "loss_pointwise": pointwise,
        "loss_requested_slew": slew,
        "projection_clipped_rows": clipped_rows,
        "gradient_norm_before_clip": gradient_norm,
    }


def _case_balanced_mse(
    target: np.ndarray,
    prediction: np.ndarray,
    case_ids: np.ndarray,
) -> np.ndarray:
    return np.mean(
        [
            np.mean(
                np.square(prediction[case_ids == case] - target[case_ids == case]),
                axis=0,
            )
            for case in np.unique(case_ids)
        ],
        axis=0,
    )


def evaluate_projection_aware_model(
    model: torch.nn.Module,
    split: Mapping[str, np.ndarray],
    loss: ModelBasedProjectedBCLoss,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate requested outputs through the frozen safety projection."""
    row_count = len(split["observations"])
    if row_count <= 0 or batch_size <= 0:
        raise ValueError("projection-aware validation counts must be positive")
    requested_chunks: list[np.ndarray] = []
    previous_chunks: list[np.ndarray] = []
    projected_chunks: list[np.ndarray] = []
    clipped_chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, row_count, batch_size):
            indices = np.arange(start, min(start + batch_size, row_count))
            requested, previous_requested = _predict_current_and_previous(
                model,
                split,
                indices,
                device=device,
            )
            _, projected, clipped = loss.projection(
                _as_tensor(
                    np.asarray(split["model_based_commands"])[indices],
                    device=device,
                    dtype=torch.float32,
                ),
                requested,
            )
            requested_chunks.append(requested.cpu().numpy())
            previous_chunks.append(previous_requested.cpu().numpy())
            projected_chunks.append(projected.cpu().numpy())
            clipped_chunks.append(clipped.cpu().numpy())
    requested = np.concatenate(requested_chunks)
    previous_requested = np.concatenate(previous_chunks)
    projected = np.concatenate(projected_chunks)
    clipped = np.concatenate(clipped_chunks)
    targets = np.asarray(split["effective_target_actions"], dtype=np.float32)
    case_ids = np.asarray(split["case_ids"], dtype=np.int64)
    model_commands = np.asarray(split["model_based_commands"], dtype=np.float32)
    zero_requested = np.zeros_like(requested)
    with torch.inference_mode():
        total_loss, pointwise_loss, slew_loss, _, _ = loss(
            torch.from_numpy(model_commands).to(device=device),
            torch.from_numpy(requested).to(device=device),
            torch.from_numpy(targets).to(device=device),
            torch.from_numpy(previous_requested).to(device=device),
            torch.from_numpy(
                np.asarray(split["delta_time_s"], dtype=np.float32)
            ).to(device=device),
            torch.from_numpy(
                np.asarray(split["transition_valid"])
            ).to(device=device),
            torch.from_numpy(
                np.asarray(split["sample_weights"], dtype=np.float32)
            ).to(device=device),
        )
        _, zero_projected, _ = loss.projection(
            torch.from_numpy(model_commands).to(device=device),
            torch.from_numpy(zero_requested).to(device=device),
        )
    candidate_mse = _case_balanced_mse(targets, projected, case_ids)
    zero_mse = _case_balanced_mse(
        targets,
        zero_projected.cpu().numpy(),
        case_ids,
    )
    transition_valid = np.asarray(split["transition_valid"])
    requested_rates = (
        np.abs(requested - previous_requested)
        * loss.action_scales.cpu().numpy()
        / np.asarray(split["delta_time_s"], dtype=np.float32)[:, None]
    )
    requested_rates[~transition_valid] = 0.0
    maximum_rates = loss.maximum_slew_rates.cpu().numpy()
    return {
        "row_count": row_count,
        "case_count": int(len(np.unique(case_ids))),
        "loss_total": float(total_loss.item()),
        "loss_pointwise": float(pointwise_loss.item()),
        "loss_requested_slew": float(slew_loss.item()),
        "case_balanced_mse_per_action": candidate_mse.tolist(),
        "zero_requested_case_balanced_mse_per_action": zero_mse.tolist(),
        "improves_over_zero_requested": (candidate_mse < zero_mse).tolist(),
        "requested_action_abs_max": np.max(np.abs(requested), axis=0).tolist(),
        "projected_action_abs_max": np.max(np.abs(projected), axis=0).tolist(),
        "projection_clipped_rows": int(np.any(clipped, axis=1).sum()),
        "requested_rate_abs_max": np.max(requested_rates, axis=0).tolist(),
        "requested_slew_violation_count": np.sum(
            requested_rates > maximum_rates[None, :] + 1e-7,
            axis=0,
        ).tolist(),
    }


def evaluate_projection_aware_recursive_model(
    model: torch.nn.Module,
    split: Mapping[str, np.ndarray],
    loss: ModelBasedProjectedBCLoss,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate the runtime recurrence using prior projected policy actions."""

    observations = np.asarray(split["observations"], dtype=np.float32)
    targets = np.asarray(split["effective_target_actions"], dtype=np.float32)
    model_commands = np.asarray(split["model_based_commands"], dtype=np.float32)
    case_ids = np.asarray(split["case_ids"], dtype=np.int64)
    transition_valid = np.asarray(split["transition_valid"])
    delta_time_s = np.asarray(split["delta_time_s"], dtype=np.float32)
    source_rows = np.asarray(split["source_row_index"], dtype=np.int64)
    row_count = len(observations)
    if (
        row_count <= 0
        or observations.ndim != 2
        or observations.shape[1] <= max(PREVIOUS_ACTION_INDICES)
        or targets.shape != (row_count, 3)
        or model_commands.shape != (row_count, 3)
        or case_ids.shape != (row_count,)
        or transition_valid.shape != (row_count,)
        or transition_valid.dtype != np.bool_
        or delta_time_s.shape != (row_count,)
        or source_rows.shape != (row_count,)
        or not np.isfinite(observations).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(model_commands).all()
        or not np.isfinite(delta_time_s).all()
    ):
        raise ValueError("recursive projection validation input mismatch")

    requested = np.zeros((row_count, 3), dtype=np.float32)
    projected = np.zeros((row_count, 3), dtype=np.float32)
    clipped = np.zeros((row_count, 3), dtype=bool)
    requested_rates = np.zeros((row_count, 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for case in np.unique(case_ids):
            indices = np.flatnonzero(case_ids == case)
            if (
                not len(indices)
                or transition_valid[indices[0]]
                or np.any(~transition_valid[indices[1:]])
                or np.any(np.diff(source_rows[indices]) <= 0)
                or np.any(delta_time_s[indices[1:]] <= 0.0)
            ):
                raise ValueError(
                    f"recursive projection case sequence is invalid: {int(case)}"
                )
            previous_effective = np.zeros(3, dtype=np.float32)
            previous_requested = np.zeros(3, dtype=np.float32)
            for offset, row in enumerate(indices):
                recurrent_observation = observations[row].copy()
                recurrent_observation[list(PREVIOUS_ACTION_INDICES)] = (
                    previous_effective
                )
                request_tensor = model(
                    torch.from_numpy(recurrent_observation[None, :]).to(
                        device=device
                    )
                )
                _, effective_tensor, clipped_tensor = loss.projection(
                    torch.from_numpy(model_commands[row : row + 1]).to(
                        device=device
                    ),
                    request_tensor,
                )
                row_requested = request_tensor.cpu().numpy()[0]
                row_effective = effective_tensor.cpu().numpy()[0]
                requested[row] = row_requested
                projected[row] = row_effective
                clipped[row] = clipped_tensor.cpu().numpy()[0]
                if offset:
                    requested_rates[row] = (
                        np.abs(row_requested - previous_requested)
                        * loss.action_scales.cpu().numpy()
                        / delta_time_s[row]
                    )
                previous_requested = row_requested
                previous_effective = row_effective

        zero_requested = torch.zeros(
            (row_count, 3), dtype=torch.float32, device=device
        )
        _, zero_projected, _ = loss.projection(
            torch.from_numpy(model_commands).to(device=device),
            zero_requested,
        )

    candidate_mse = _case_balanced_mse(targets, projected, case_ids)
    zero_mse = _case_balanced_mse(
        targets,
        zero_projected.cpu().numpy(),
        case_ids,
    )
    maximum_rates = loss.maximum_slew_rates.cpu().numpy()
    return {
        "recurrence_contract": (
            MODEL_BASED_CORRECTIVE_BC_RECURSIVE_VALIDATION_CONTRACT
        ),
        "row_count": row_count,
        "case_count": int(len(np.unique(case_ids))),
        "case_reset_count": int(len(np.unique(case_ids))),
        "loss_total": float(np.mean(candidate_mse)),
        "case_balanced_mse_per_action": candidate_mse.tolist(),
        "zero_requested_case_balanced_mse_per_action": zero_mse.tolist(),
        "improves_over_zero_requested": (candidate_mse < zero_mse).tolist(),
        "requested_action_abs_max": np.max(np.abs(requested), axis=0).tolist(),
        "projected_action_abs_max": np.max(np.abs(projected), axis=0).tolist(),
        "projection_clipped_rows": int(np.any(clipped, axis=1).sum()),
        "requested_rate_abs_max": np.max(requested_rates, axis=0).tolist(),
        "requested_slew_violation_count": np.sum(
            requested_rates > maximum_rates[None, :] + 1e-7,
            axis=0,
        ).tolist(),
    }


def build_projection_aware_bc_preflight(
    metadata: Mapping[str, object],
    payload: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Validate the BC mechanics using audit requests without training a model."""
    validate_training_dataset(metadata, payload)
    if metadata.get("schema") != MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA:
        raise ValueError("projection-aware BC preflight requires the training schema")

    requested = np.asarray(payload["requested_actions_audit"], dtype=np.float32)
    previous_requested = _previous_requested_actions(
        requested,
        np.asarray(payload["previous_row_index"]),
        np.asarray(payload["transition_valid"]),
    )
    model_commands = np.asarray(payload["model_based_commands"], dtype=np.float32)
    effective_targets = np.asarray(payload["actions"], dtype=np.float32)
    delta_time_s = np.asarray(payload["delta_time_s"], dtype=np.float32)
    transition_valid = np.asarray(payload["transition_valid"])
    sample_weights = np.asarray(
        payload["case_balanced_sample_weights"], dtype=np.float32
    )
    loss = ModelBasedProjectedBCLoss(
        action_scales=metadata["action_scales"],
    )
    with torch.inference_mode():
        (
            total_loss,
            pointwise_loss,
            slew_loss,
            projected_actions,
            command_clipped,
        ) = loss(
            torch.from_numpy(model_commands),
            torch.from_numpy(requested),
            torch.from_numpy(effective_targets),
            torch.from_numpy(previous_requested),
            torch.from_numpy(delta_time_s),
            torch.from_numpy(transition_valid),
            torch.from_numpy(sample_weights),
        )
    projected = projected_actions.numpy()
    clipped = command_clipped.numpy()
    action_scales = np.asarray(metadata["action_scales"], dtype=np.float32)
    maximum_slew_rates = loss.maximum_slew_rates.numpy()
    requested_rates = (
        np.abs(requested - previous_requested)
        * action_scales
        / delta_time_s[:, None]
    )
    requested_rates[~transition_valid] = 0.0
    projection_error = np.abs(projected - effective_targets)
    reconstruction_abs_max = float(np.max(projection_error))
    reconstruction_passed = reconstruction_abs_max <= 1e-6
    finite_loss = all(
        np.isfinite(value)
        for value in (
            float(total_loss.item()),
            float(pointwise_loss.item()),
            float(slew_loss.item()),
        )
    )
    return {
        "schema": MODEL_BASED_CORRECTIVE_BC_PREFLIGHT_SCHEMA,
        "adapter_contract": MODEL_BASED_CORRECTIVE_BC_ADAPTER_CONTRACT,
        "dataset_schema": metadata["schema"],
        "row_count": int(metadata["row_count"]),
        "case_count": int(metadata["case_count"]),
        "split_cases": metadata["split_cases"],
        "reserved_holdout_cases": metadata["reserved_holdout_cases"],
        "holdout_rows_present": False,
        "loss_contract": MODEL_BASED_PROJECTED_BC_LOSS,
        "requested_slew_regularization_contract": (
            REQUESTED_OUTPUT_SLEW_REGULARIZATION
        ),
        "loss_total": float(total_loss.item()),
        "loss_pointwise": float(pointwise_loss.item()),
        "loss_requested_slew": float(slew_loss.item()),
        "teacher_projection_reconstruction_abs_max": reconstruction_abs_max,
        "teacher_projection_reconstruction_passed": reconstruction_passed,
        "teacher_projection_reconstruction_required_for_preflight": False,
        "teacher_projection_clipped_rows": int(np.any(clipped, axis=1).sum()),
        "teacher_requested_rate_abs_max": np.max(
            requested_rates, axis=0
        ).tolist(),
        "teacher_requested_slew_violation_count": np.sum(
            requested_rates > maximum_slew_rates[None, :] + 1e-7,
            axis=0,
        ).tolist(),
        "requested_actions_used_only_for_contract_preflight": True,
        "requested_actions_used_as_training_targets": False,
        "effective_actions_remain_training_targets": True,
        "dataset_valid_for_projection_aware_bc_input": True,
        "preflight_passed": finite_loss,
        "bc_authorized": False,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
        "training_started": False,
        "checkpoint_created": False,
        "valid_for_training": False,
    }
