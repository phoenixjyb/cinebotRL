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

MODEL_BASED_CORRECTIVE_BC_PREFLIGHT_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_bc_preflight_v1"
)
MODEL_BASED_CORRECTIVE_BC_ADAPTER_CONTRACT = (
    "projection_aware_effective_label_bc_adapter_v1"
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
