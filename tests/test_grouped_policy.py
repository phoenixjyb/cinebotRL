"""Unit tests for grouped-head SB3 policy wiring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from scripts.reinforcement_learning.sb3.grouped_policy import GroupedActorCriticPolicy


def _constant_schedule(_progress_remaining: float) -> float:
    return 1e-4


def test_grouped_policy_preserves_9d_action_order():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(85,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
    policy = GroupedActorCriticPolicy(
        obs_space,
        act_space,
        _constant_schedule,
        shared_hidden_dims=(16,),
        head_hidden_dim=8,
        value_hidden_dims=(16,),
        activation_fn=nn.ELU,
    )

    with torch.no_grad():
        for param in policy.mlp_extractor.parameters():
            param.zero_()
        for _name, indices in policy.mlp_extractor.action_groups.items():
            head = policy.mlp_extractor.action_heads[_name]
            final = head[-1]
            for row, action_index in enumerate(indices):
                final.bias[row] = float(action_index)

    latent = policy.mlp_extractor.forward_actor(torch.zeros(2, 85))
    expected = torch.arange(9, dtype=latent.dtype).repeat(2, 1)
    assert latent.shape == (2, 9)
    torch.testing.assert_close(latent, expected)
    assert isinstance(policy.action_net, nn.Identity)


def test_grouped_policy_can_be_used_by_ppo():
    class DummyEnv(gym.Env):
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(85,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        metadata = {"render_modes": []}

        def reset(self, *, seed=None, options=None):
            return np.zeros(85, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(85, dtype=np.float32), 0.0, False, False, {}

    model = PPO(
        GroupedActorCriticPolicy,
        DummyEnv(),
        policy_kwargs=dict(
            shared_hidden_dims=(16,),
            head_hidden_dim=8,
            value_hidden_dims=(16,),
            activation_fn=nn.ELU,
        ),
        n_steps=8,
        batch_size=4,
        device="cpu",
        verbose=0,
    )
    obs = np.zeros((1, 85), dtype=np.float32)
    action, _state = model.predict(obs, deterministic=True)
    assert action.shape == (1, 9)
