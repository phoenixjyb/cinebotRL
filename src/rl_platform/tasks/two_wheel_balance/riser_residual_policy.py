"""Bounded deployable policy network for high-level riser trajectory residuals."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .riser_residual_dataset import ACTION_NAMES, OBSERVATION_NAMES


class RiserResidualPolicy(nn.Module):
    """Normalized MLP whose tanh head cannot exceed the residual contract."""

    def __init__(
        self,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        hidden_sizes: Sequence[int] = (256, 256, 128),
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
        sizes = tuple(int(size) for size in hidden_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("hidden sizes must be positive")
        self.register_buffer("observation_mean", mean)
        self.register_buffer("observation_std", std)
        layers: list[nn.Module] = []
        input_size = len(OBSERVATION_NAMES)
        for output_size in sizes:
            linear = nn.Linear(input_size, output_size)
            nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.LayerNorm(output_size), nn.SiLU()))
            input_size = output_size
        self.encoder = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_size, len(ACTION_NAMES))
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.observation_mean) / self.observation_std
        return torch.tanh(self.action_head(self.encoder(normalized)))
