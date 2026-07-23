"""Bounded deployable policy network for high-level riser trajectory residuals."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .riser_residual_dataset import (
    ACTION_NAMES,
    BASE_OBSERVATION_NAMES,
    LOOKAHEAD_CHANNEL_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


POLICY_ARCHITECTURE = "state_shared_lookahead_fusion_v1"
MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE = (
    "state_shared_lookahead_fusion_previous_action_masked_v1"
)
ATTENUATED_PREVIOUS_ACTION_POLICY_ARCHITECTURE = (
    "state_shared_lookahead_fusion_previous_action_attenuated_v1"
)
MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE = (
    "model_based_shared_encoder_zero_initialized_residual_v1"
)
MODEL_BASED_RESIDUAL_SAFETY_PROJECTION = (
    "model_based_residual_safety_projection_v1"
)


def _encoder(
    input_size: int, hidden_sizes: Sequence[int]
) -> tuple[nn.Sequential, int]:
    sizes = tuple(int(size) for size in hidden_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("encoder hidden sizes must be positive")
    layers: list[nn.Module] = []
    for output_size in sizes:
        linear = nn.Linear(input_size, output_size)
        nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.LayerNorm(output_size), nn.SiLU()))
        input_size = output_size
    return nn.Sequential(*layers), input_size


class RiserResidualPolicy(nn.Module):
    """Fuse physical state with shared-encoder trajectory lookahead features."""

    def __init__(
        self,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        state_hidden_sizes: Sequence[int] = (128, 128),
        lookahead_hidden_sizes: Sequence[int] = (64, 64),
        fusion_hidden_sizes: Sequence[int] = (256, 128),
        masked_observation_indices: Sequence[int] = (),
        previous_action_observation_gain: float | Sequence[float] = 1.0,
        zero_initialize_action_head: bool = False,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(observation_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(observation_std, dtype=torch.float32).reshape(-1)
        if mean.shape != (len(OBSERVATION_NAMES),) or std.shape != mean.shape:
            raise ValueError("observation normalization dimension mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("observation normalization must be finite")
        if torch.any(std <= 0.0):
            raise ValueError("observation standard deviations must be positive")
        self.register_buffer("observation_mean", mean)
        self.register_buffer("observation_std", std)
        indices = tuple(sorted(set(int(index) for index in masked_observation_indices)))
        if any(index < 0 or index >= len(OBSERVATION_NAMES) for index in indices):
            raise ValueError("masked observation index is out of range")
        if isinstance(previous_action_observation_gain, (int, float)):
            gains = (float(previous_action_observation_gain),) * len(
                PREVIOUS_ACTION_INDICES
            )
        else:
            gains = tuple(float(value) for value in previous_action_observation_gain)
        if len(gains) != len(PREVIOUS_ACTION_INDICES) or any(
            not 0.0 <= gain <= 1.0 for gain in gains
        ):
            raise ValueError("previous-action observation gains must be three values in [0, 1]")
        observation_mask = torch.ones(len(OBSERVATION_NAMES), dtype=torch.float32)
        observation_mask[list(PREVIOUS_ACTION_INDICES)] = torch.tensor(gains)
        if indices:
            observation_mask[list(indices)] = 0.0
        self.register_buffer("observation_mask", observation_mask)
        self.state_observation_count = len(BASE_OBSERVATION_NAMES)
        self.lookahead_count = len(LOOKAHEAD_HORIZONS_S)
        self.lookahead_channel_count = len(LOOKAHEAD_CHANNEL_NAMES)
        self.state_encoder, state_size = _encoder(
            self.state_observation_count, state_hidden_sizes
        )
        self.lookahead_encoder, lookahead_size = _encoder(
            self.lookahead_channel_count, lookahead_hidden_sizes
        )
        self.fusion_encoder, fusion_size = _encoder(
            state_size + self.lookahead_count * lookahead_size,
            fusion_hidden_sizes,
        )
        self.action_head = nn.Linear(fusion_size, len(ACTION_NAMES))
        if zero_initialize_action_head:
            nn.init.zeros_(self.action_head.weight)
        else:
            nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (
            (observations - self.observation_mean)
            / self.observation_std
            * self.observation_mask
        )
        state_embedding = self.state_encoder(
            normalized[:, : self.state_observation_count]
        )
        lookahead = normalized[:, self.state_observation_count :].reshape(
            -1, self.lookahead_channel_count
        )
        lookahead_embedding = self.lookahead_encoder(lookahead).reshape(
            normalized.shape[0], -1
        )
        fused = torch.cat((state_embedding, lookahead_embedding), dim=1)
        return torch.tanh(self.action_head(self.fusion_encoder(fused)))


class ModelBasedResidualSafetyProjection(nn.Module):
    """Project requested residuals through the deterministic command supervisor."""

    def __init__(
        self,
        action_scales: Sequence[float] = (0.05, 0.05, 0.02),
        command_lower_bounds: Sequence[float] = (-0.4, -0.4, 0.0),
        command_upper_bounds: Sequence[float] = (0.4, 0.4, 1.2),
    ) -> None:
        super().__init__()
        scales = torch.as_tensor(action_scales, dtype=torch.float32).reshape(-1)
        lower = torch.as_tensor(command_lower_bounds, dtype=torch.float32).reshape(-1)
        upper = torch.as_tensor(command_upper_bounds, dtype=torch.float32).reshape(-1)
        if scales.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("model-based residual projection dimension mismatch")
        if (
            not torch.isfinite(scales).all()
            or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
            or torch.any(scales <= 0.0)
            or torch.any(lower >= upper)
        ):
            raise ValueError("model-based residual projection limits are invalid")
        self.register_buffer("action_scales", scales)
        self.register_buffer("command_lower_bounds", lower)
        self.register_buffer("command_upper_bounds", upper)

    def forward(
        self,
        model_based_commands: torch.Tensor,
        requested_normalized_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            model_based_commands.ndim != 2
            or requested_normalized_actions.ndim != 2
            or model_based_commands.shape != requested_normalized_actions.shape
            or model_based_commands.shape[1] != 3
        ):
            raise ValueError("model-based residual projection input shape mismatch")
        if not torch.isfinite(model_based_commands).all() or not torch.isfinite(
            requested_normalized_actions
        ).all():
            raise ValueError("model-based residual projection input is non-finite")
        bounded_actions = torch.clamp(requested_normalized_actions, -1.0, 1.0)
        requested_commands = (
            model_based_commands + self.action_scales * bounded_actions
        )
        effective_commands = torch.maximum(
            torch.minimum(requested_commands, self.command_upper_bounds),
            self.command_lower_bounds,
        )
        effective_actions = (
            effective_commands - model_based_commands
        ) / self.action_scales
        command_clipped = torch.abs(effective_commands - requested_commands) > 1e-7
        return effective_commands, effective_actions, command_clipped


def initialize_model_based_residual_from_planner_imitation(
    target: RiserResidualPolicy, source: RiserResidualPolicy
) -> None:
    """Reuse planner-imitation encoders while resetting the residual head to zero."""

    target_state = target.state_dict()
    source_state = source.state_dict()
    encoder_names = [
        name for name in target_state if not name.startswith("action_head.")
    ]
    if set(encoder_names) != {
        name for name in source_state if not name.startswith("action_head.")
    }:
        raise ValueError("planner-imitation and residual encoder contracts differ")
    for name in encoder_names:
        if target_state[name].shape != source_state[name].shape:
            raise ValueError(f"planner-imitation encoder shape mismatch: {name}")
        target_state[name] = source_state[name].detach().clone()
    target.load_state_dict(target_state)
    nn.init.zeros_(target.action_head.weight)
    nn.init.zeros_(target.action_head.bias)
