"""Projection-aware BC loss for bounded model-based riser residuals."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .riser_residual_policy import ModelBasedResidualSafetyProjection

MODEL_BASED_PROJECTED_BC_LOSS = "model_based_projected_effective_action_bc_loss_v1"
REQUESTED_OUTPUT_SLEW_REGULARIZATION = "requested_physical_residual_slew_hinge_v1"


class ModelBasedProjectedBCLoss(nn.Module):
    """Compare safety-projected policy requests with effective teacher labels."""

    def __init__(
        self,
        action_scales: Sequence[float] = (0.05, 0.05, 0.02),
        maximum_slew_rates: Sequence[float] = (0.1, 0.1, 0.04),
        pointwise_loss_scale: Sequence[float] = (1.0, 1.0, 1.0),
        slew_regularization_weight: float = 0.1,
    ) -> None:
        super().__init__()
        scales = torch.as_tensor(action_scales, dtype=torch.float32).reshape(-1)
        slew = torch.as_tensor(maximum_slew_rates, dtype=torch.float32).reshape(-1)
        pointwise_scale = torch.as_tensor(
            pointwise_loss_scale, dtype=torch.float32
        ).reshape(-1)
        if scales.shape != (3,) or slew.shape != (3,) or pointwise_scale.shape != (3,):
            raise ValueError("model-based BC loss dimension mismatch")
        if (
            not torch.isfinite(scales).all()
            or not torch.isfinite(slew).all()
            or not torch.isfinite(pointwise_scale).all()
            or torch.any(scales <= 0.0)
            or torch.any(slew <= 0.0)
            or torch.any(pointwise_scale <= 0.0)
            or not 0.0 <= float(slew_regularization_weight)
        ):
            raise ValueError("model-based BC loss limits are invalid")
        self.projection = ModelBasedResidualSafetyProjection(
            action_scales=scales.tolist()
        )
        self.register_buffer("action_scales", scales)
        self.register_buffer("maximum_slew_rates", slew)
        self.register_buffer("pointwise_loss_scale", pointwise_scale)
        self.slew_regularization_weight = float(slew_regularization_weight)

    def forward(
        self,
        model_based_commands: torch.Tensor,
        requested_normalized_actions: torch.Tensor,
        effective_target_actions: torch.Tensor,
        previous_requested_normalized_actions: torch.Tensor,
        delta_time_s: torch.Tensor,
        transition_valid: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if model_based_commands.ndim != 2 or model_based_commands.shape[1] != 3:
            raise ValueError("model-based BC loss input shape mismatch")
        row_count = model_based_commands.shape[0]
        if (
            requested_normalized_actions.ndim != 2
            or requested_normalized_actions.shape[0] != row_count
            or requested_normalized_actions.shape[1] != 3
            or effective_target_actions.ndim != 2
            or effective_target_actions.shape[0] != row_count
            or effective_target_actions.shape[1] != 3
            or previous_requested_normalized_actions.ndim != 2
            or previous_requested_normalized_actions.shape[0] != row_count
            or previous_requested_normalized_actions.shape[1] != 3
            or delta_time_s.ndim != 1
            or delta_time_s.shape[0] != row_count
            or transition_valid.ndim != 1
            or transition_valid.shape[0] != row_count
            or sample_weights.ndim != 1
            or sample_weights.shape[0] != row_count
        ):
            raise ValueError("model-based BC loss input shape mismatch")
        if transition_valid.dtype != torch.bool:
            raise ValueError("model-based BC transition mask must be boolean")
        if (
            not torch.isfinite(model_based_commands).all()
            or not torch.isfinite(requested_normalized_actions).all()
            or not torch.isfinite(effective_target_actions).all()
            or not torch.isfinite(previous_requested_normalized_actions).all()
            or not torch.isfinite(delta_time_s).all()
            or not torch.isfinite(sample_weights).all()
        ):
            raise ValueError("model-based BC loss input is non-finite")
        if (
            torch.any(torch.abs(requested_normalized_actions) > 1.0 + 1e-6)
            or torch.any(torch.abs(previous_requested_normalized_actions) > 1.0 + 1e-6)
            or torch.any(torch.abs(effective_target_actions) > 1.0 + 1e-6)
        ):
            raise ValueError("model-based BC action exceeds normalized bounds")
        if (
            torch.any(delta_time_s <= 0.0)
            or torch.any(sample_weights < 0.0)
            or torch.sum(sample_weights) <= 0.0
        ):
            raise ValueError("model-based BC timing or sample weights are invalid")
        transition_weights = sample_weights * transition_valid.to(
            dtype=sample_weights.dtype
        )
        if torch.any(transition_valid) and torch.sum(transition_weights) <= 0.0:
            raise ValueError("model-based BC valid transitions have zero weight")

        _, projected_actions, command_clipped = self.projection(
            model_based_commands, requested_normalized_actions
        )
        pointwise_per_row = torch.mean(
            torch.square(
                (projected_actions - effective_target_actions)
                / self.pointwise_loss_scale
            ),
            dim=1,
        )
        pointwise_loss = torch.sum(pointwise_per_row * sample_weights) / torch.sum(
            sample_weights
        )

        requested_physical_rates = torch.abs(
            (requested_normalized_actions - previous_requested_normalized_actions)
            * self.action_scales
            / delta_time_s[:, None]
        )
        normalized_slew_excess = torch.relu(
            requested_physical_rates / self.maximum_slew_rates - 1.0
        )
        if torch.any(transition_valid):
            slew_loss = torch.sum(
                torch.mean(torch.square(normalized_slew_excess), dim=1)
                * transition_weights
            ) / torch.sum(transition_weights)
        else:
            slew_loss = torch.sum(requested_normalized_actions * 0.0)
        total_loss = pointwise_loss + self.slew_regularization_weight * slew_loss
        return (
            total_loss,
            pointwise_loss,
            slew_loss,
            projected_actions,
            command_clipped,
        )
