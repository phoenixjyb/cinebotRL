"""Grouped-head SB3 policies for Proto2 whole-body control.

The default SB3 ``MlpPolicy`` uses one flat actor trunk for all action
dimensions.  For CineBotRL that mixes three different control surfaces:
arm joints, gimbal/attitude channels, and holonomic base velocity.  This
module keeps SB3's PPO machinery unchanged while making those action groups
explicit in the actor.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "arm": (0, 1, 2),
    "gimbal": (3, 4, 5),
    "base": (6, 7, 8),
}


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation_fn())
        last_dim = int(hidden_dim)
    return nn.Sequential(*layers), last_dim


class GroupedActionMlpExtractor(nn.Module):
    """Shared encoder with separate arm, gimbal, and base action heads."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        shared_hidden_dims: Iterable[int] = (256, 256),
        head_hidden_dim: int = 128,
        value_hidden_dims: Iterable[int] = (256, 128),
        activation_fn: type[nn.Module] = nn.ELU,
        action_groups: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_groups = action_groups or ACTION_GROUPS
        self._validate_action_groups()

        self.shared_encoder, shared_dim = _build_mlp(feature_dim, shared_hidden_dims, activation_fn)
        self.action_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(shared_dim, int(head_hidden_dim)),
                    activation_fn(),
                    nn.Linear(int(head_hidden_dim), len(indices)),
                )
                for name, indices in self.action_groups.items()
            }
        )
        self.value_net, value_dim = _build_mlp(feature_dim, value_hidden_dims, activation_fn)

        # SB3 reads these attributes when constructing action_net and value_net.
        # The actor latent is already the action mean, so the policy uses Identity
        # as action_net.
        self.latent_dim_pi = self.action_dim
        self.latent_dim_vf = value_dim

    def _validate_action_groups(self) -> None:
        indices = sorted(index for group in self.action_groups.values() for index in group)
        expected = list(range(self.action_dim))
        if indices != expected:
            raise ValueError(f"action groups must cover {expected}, got {indices}")

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        shared = self.shared_encoder(features)
        action_mean = features.new_zeros((features.shape[0], self.action_dim))
        for name, indices in self.action_groups.items():
            action_mean[:, list(indices)] = self.action_heads[name](shared)
        return action_mean

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def copy_actor_from(
        self,
        source: "GroupedActionMlpExtractor",
        action_indices: Iterable[int] | None = None,
        zero_unselected: bool = False,
    ) -> None:
        """Copy grouped actor weights from another extractor.

        ``action_indices`` allows base-only or masked student policies to warm
        start selected output rows while keeping the rest freshly initialized.
        The shared encoder is copied because selected heads depend on that
        representation.
        """

        if not isinstance(source, GroupedActionMlpExtractor):
            raise TypeError(f"source must be GroupedActionMlpExtractor, got {type(source)!r}")

        self.shared_encoder.load_state_dict(source.shared_encoder.state_dict())
        selected = None if action_indices is None else {int(index) for index in action_indices}

        with th.no_grad():
            for name, indices in self.action_groups.items():
                target_head = self.action_heads[name]
                source_head = source.action_heads[name]
                target_final = target_head[-1]
                source_final = source_head[-1]

                if selected is None or set(indices).issubset(selected):
                    target_head.load_state_dict(source_head.state_dict())
                    continue

                if zero_unselected:
                    target_final.weight.zero_()
                    target_final.bias.zero_()

                for row, action_index in enumerate(indices):
                    if action_index in selected:
                        target_final.weight[row].copy_(source_final.weight[row])
                        target_final.bias[row].copy_(source_final.bias[row])


class GroupedActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with separate heads for action groups."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        shared_hidden_dims: Iterable[int] = (256, 256),
        head_hidden_dim: int = 128,
        value_hidden_dims: Iterable[int] = (256, 128),
        action_groups: dict[str, tuple[int, ...]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.grouped_shared_hidden_dims = tuple(int(x) for x in shared_hidden_dims)
        self.grouped_head_hidden_dim = int(head_hidden_dim)
        self.grouped_value_hidden_dims = tuple(int(x) for x in value_hidden_dims)
        self.grouped_action_groups = action_groups or ACTION_GROUPS
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        action_dim = int(self.action_space.shape[0])  # type: ignore[index]
        self.mlp_extractor = GroupedActionMlpExtractor(
            feature_dim=self.features_dim,
            action_dim=action_dim,
            shared_hidden_dims=self.grouped_shared_hidden_dims,
            head_hidden_dim=self.grouped_head_hidden_dim,
            value_hidden_dims=self.grouped_value_hidden_dims,
            activation_fn=self.activation_fn,
            action_groups=self.grouped_action_groups,
        )

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        action_dim = int(self.action_space.shape[0])  # type: ignore[index]
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise NotImplementedError(
                f"GroupedActorCriticPolicy currently supports continuous Box/DiagGaussian actions only, "
                f"got {self.action_dist!r}"
            )

        # ``GroupedActionMlpExtractor.forward_actor`` already emits the action
        # mean in canonical 9D order.  Keep SB3's Gaussian distribution/log_std.
        self.action_net = nn.Identity()
        self.log_std = nn.Parameter(th.ones(action_dim) * self.log_std_init, requires_grad=True)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
            for head in self.mlp_extractor.action_heads.values():
                head[-1].apply(partial(self.init_weights, gain=0.01))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                shared_hidden_dims=self.grouped_shared_hidden_dims,
                head_hidden_dim=self.grouped_head_hidden_dim,
                value_hidden_dims=self.grouped_value_hidden_dims,
                action_groups=self.grouped_action_groups,
            )
        )
        return data


def is_grouped_policy(policy: nn.Module) -> bool:
    return isinstance(getattr(policy, "mlp_extractor", None), GroupedActionMlpExtractor)
