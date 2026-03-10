"""Tests for observation composition — no Isaac Sim required."""

from __future__ import annotations

import pytest
import torch

from rl_platform.tasks.mobile_mm.observations import (
    compose_observation,
    get_observation_dimensions,
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

N = 4  # num_envs for all tests


def _unit_quat(n: int) -> torch.Tensor:
    """Identity quaternion [w, x, y, z] = [1, 0, 0, 0]."""
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _base_obs_kwargs(n: int = N) -> dict:
    """Minimal valid kwargs for compose_observation."""
    return dict(
        base_pos=torch.zeros(n, 3),
        base_quat=_unit_quat(n),
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        joint_pos=torch.zeros(n, 9),
        joint_vel=torch.zeros(n, 9),
        ee_pos=torch.ones(n, 3) * 0.5,
        ee_quat=_unit_quat(n),
        ee_lin_vel=torch.zeros(n, 3),
        ee_ang_vel=torch.zeros(n, 3),
        target_pos=torch.ones(n, 3) * 0.6,
        target_quat=_unit_quat(n),
    )


# -------------------------------------------------------------------
# get_observation_dimensions
# -------------------------------------------------------------------

class TestGetObservationDimensions:
    def test_baseline(self):
        """Default config: 6 joints, no optional features."""
        dim = get_observation_dimensions(num_joints=6)
        # base(13) + joints(12) + ee(13) + error(10) + base-target(8) = 56
        assert dim == 56

    def test_with_lookahead(self):
        dim_no = get_observation_dimensions(num_joints=6, use_lookahead=False)
        dim_la = get_observation_dimensions(num_joints=6, use_lookahead=True, lookahead_steps=3)
        assert dim_la == dim_no + 3 * 3  # 3 steps × 3 (xyz)

    def test_with_action_history(self):
        dim_no = get_observation_dimensions(num_joints=6, use_action_history=False)
        dim_ah = get_observation_dimensions(
            num_joints=6, use_action_history=True, action_history_length=2, action_dim=8
        )
        assert dim_ah == dim_no + 2 * 8

    def test_with_contacts(self):
        dim_no = get_observation_dimensions(num_joints=6, num_contacts=0)
        dim_c = get_observation_dimensions(num_joints=6, num_contacts=1)
        assert dim_c == dim_no + 1

    def test_with_obstacles(self):
        dim_no = get_observation_dimensions(num_joints=6, use_obstacles=False)
        dim_ob = get_observation_dimensions(num_joints=6, use_obstacles=True)
        assert dim_ob == dim_no + 1

    def test_all_features(self):
        dim = get_observation_dimensions(
            num_joints=6,
            num_contacts=1,
            use_lookahead=True,
            lookahead_steps=3,
            use_action_history=True,
            action_history_length=2,
            action_dim=8,
            use_obstacles=True,
        )
        expected = 56 + 9 + 16 + 1 + 1  # baseline + lookahead + history + contacts + obstacle
        assert dim == expected


# -------------------------------------------------------------------
# compose_observation — output shape
# -------------------------------------------------------------------

class TestComposeObservationShape:
    def test_baseline_shape(self):
        obs = compose_observation(**_base_obs_kwargs())
        expected_dim = get_observation_dimensions(num_joints=6)
        assert obs.shape == (N, expected_dim)

    def test_with_contact_forces(self):
        kwargs = _base_obs_kwargs()
        kwargs["contact_forces"] = torch.zeros(N, 1)
        obs = compose_observation(**kwargs)
        expected_dim = get_observation_dimensions(num_joints=6, num_contacts=1)
        assert obs.shape == (N, expected_dim)

    def test_with_min_obstacle_dist(self):
        kwargs = _base_obs_kwargs()
        kwargs["min_obstacle_dist"] = torch.ones(N, 1) * 2.0
        obs = compose_observation(**kwargs)
        expected_dim = get_observation_dimensions(num_joints=6, use_obstacles=True)
        assert obs.shape == (N, expected_dim)

    def test_with_lookahead(self):
        kwargs = _base_obs_kwargs()
        kwargs["lookahead_pos"] = torch.zeros(N, 3, 3)  # 3 steps, 3 xyz
        obs = compose_observation(**kwargs)
        expected_dim = get_observation_dimensions(num_joints=6, use_lookahead=True, lookahead_steps=3)
        assert obs.shape == (N, expected_dim)

    def test_with_action_history(self):
        kwargs = _base_obs_kwargs()
        kwargs["action_history"] = torch.zeros(N, 2, 8)
        obs = compose_observation(**kwargs)
        expected_dim = get_observation_dimensions(
            num_joints=6, use_action_history=True, action_history_length=2, action_dim=8
        )
        assert obs.shape == (N, expected_dim)

    def test_batch_size_one(self):
        kwargs = _base_obs_kwargs(n=1)
        obs = compose_observation(**kwargs)
        assert obs.shape[0] == 1

    def test_large_batch(self):
        kwargs = _base_obs_kwargs(n=8192)
        obs = compose_observation(**kwargs)
        assert obs.shape[0] == 8192


# -------------------------------------------------------------------
# compose_observation — content correctness
# -------------------------------------------------------------------

class TestComposeObservationContent:
    def test_position_error_reflected(self):
        """pos_error = target_pos - ee_pos should appear in the observation."""
        kwargs = _base_obs_kwargs()
        kwargs["ee_pos"] = torch.zeros(N, 3)
        kwargs["target_pos"] = torch.ones(N, 3)
        obs = compose_observation(**kwargs)
        # pos_error = [1, 1, 1] should appear somewhere in obs
        assert (obs == 1.0).any()

    def test_all_finite(self):
        obs = compose_observation(**_base_obs_kwargs())
        assert torch.isfinite(obs).all()

    def test_obstacle_dist_1d_auto_unsqueezed(self):
        """min_obstacle_dist passed as 1-D tensor should be accepted and unsqueezed."""
        kwargs = _base_obs_kwargs()
        kwargs["min_obstacle_dist"] = torch.ones(N)  # 1-D
        obs = compose_observation(**kwargs)
        expected_dim = get_observation_dimensions(num_joints=6, use_obstacles=True)
        assert obs.shape == (N, expected_dim)
