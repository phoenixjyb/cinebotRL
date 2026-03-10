"""Tests for reward functions — no Isaac Sim required."""

from __future__ import annotations

import pytest
import torch

from rl_platform.tasks.mobile_mm.rewards import (
    position_tracking_reward,
    orientation_tracking_reward,
    action_magnitude_penalty,
    action_rate_penalty,
    self_collision_penalty,
    obstacle_distance_reward,
    velocity_limit_penalty,
    joint_limit_penalty,
    lateral_motion_penalty,
    stability_penalty,
    jerk_penalty,
)

N = 4  # num_envs


def _unit_quat(n: int = N) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


# -------------------------------------------------------------------
# position_tracking_reward
# -------------------------------------------------------------------

class TestPositionTracking:
    def test_zero_error_max_reward(self):
        pos = torch.zeros(N, 3)
        r = position_tracking_reward(pos, pos, scale=1.0)
        assert r.shape == (N,)
        assert (r > 0.99).all(), "Zero error should yield near-maximum reward"

    def test_large_error_low_reward(self):
        current = torch.zeros(N, 3)
        target = torch.ones(N, 3) * 10.0  # 10 m away
        r = position_tracking_reward(current, target, scale=1.0)
        assert (r < 0.1).all(), "Large error should yield near-zero reward"

    def test_reward_decreases_with_error(self):
        pos = torch.zeros(N, 3)
        r_close = position_tracking_reward(pos, torch.ones(N, 3) * 0.01, scale=1.0)
        r_far = position_tracking_reward(pos, torch.ones(N, 3) * 1.0, scale=1.0)
        assert (r_close > r_far).all()

    def test_scale_doubles_exponent(self):
        """scale is the decay rate in exp(-scale * error), so doubling it squares the reward."""
        pos = torch.zeros(N, 3)
        target = torch.ones(N, 3) * 0.5
        r1 = position_tracking_reward(pos, target, scale=1.0)
        r2 = position_tracking_reward(pos, target, scale=2.0)
        # r2 = exp(-2*alpha) = (exp(-alpha))^2 = r1^2
        assert torch.allclose(r2, r1 ** 2, atol=1e-5)

    def test_output_range(self):
        pos = torch.zeros(N, 3)
        target = torch.rand(N, 3)
        r = position_tracking_reward(pos, target, scale=1.0)
        assert (r >= 0.0).all() and (r <= 1.0).all()


# -------------------------------------------------------------------
# orientation_tracking_reward
# -------------------------------------------------------------------

class TestOrientationTracking:
    def test_identical_orientation_max_reward(self):
        q = _unit_quat()
        r = orientation_tracking_reward(q, q, scale=1.0)
        assert (r > 0.99).all()

    def test_output_range(self):
        q1 = _unit_quat()
        q2 = torch.randn(N, 4)
        q2 = q2 / q2.norm(dim=-1, keepdim=True)
        r = orientation_tracking_reward(q1, q2, scale=1.0)
        assert (r >= 0.0).all() and (r <= 1.0).all()


# -------------------------------------------------------------------
# action_magnitude_penalty / action_rate_penalty
# -------------------------------------------------------------------

class TestActionPenalties:
    def test_zero_actions_zero_penalty(self):
        actions = torch.zeros(N, 8)
        p = action_magnitude_penalty(actions, scale=1.0)
        assert torch.allclose(p, torch.zeros(N))

    def test_penalty_nonnegative(self):
        actions = torch.randn(N, 8)
        p = action_magnitude_penalty(actions, scale=1.0)
        assert (p >= 0.0).all()

    def test_rate_zero_when_unchanged(self):
        actions = torch.ones(N, 8)
        p = action_rate_penalty(actions, actions, scale=1.0)
        assert torch.allclose(p, torch.zeros(N))

    def test_rate_positive_when_changed(self):
        a1 = torch.ones(N, 8)
        a2 = torch.zeros(N, 8)
        p = action_rate_penalty(a1, a2, scale=1.0)
        assert (p > 0.0).all()


# -------------------------------------------------------------------
# self_collision_penalty
# -------------------------------------------------------------------

class TestSelfCollisionPenalty:
    def _forces(self, n_bodies: int, magnitude: float) -> torch.Tensor:
        """Create uniform contact force tensors [N, n_bodies, 3]."""
        f = torch.zeros(N, n_bodies, 3)
        f[:, :, 0] = magnitude  # Force along x-axis
        return f

    def test_no_contact_no_penalty(self):
        forces = self._forces(3, 0.0)
        p = self_collision_penalty(forces, threshold=1.0, scale=1.0)
        assert torch.allclose(p, torch.zeros(N))

    def test_contact_above_threshold_penalized(self):
        forces = self._forces(3, 10.0)  # 10 N >> 1 N threshold
        p = self_collision_penalty(forces, threshold=1.0, scale=1.0, exclude_base=False)
        assert (p > 0.0).all()

    def test_base_excluded(self):
        # Only base link has large force — should yield zero penalty
        forces = self._forces(3, 0.0)
        forces[:, 0, 0] = 100.0  # Base link only
        p = self_collision_penalty(forces, threshold=1.0, scale=1.0, exclude_base=True)
        assert torch.allclose(p, torch.zeros(N))

    def test_binary_mode(self):
        forces = self._forces(2, 5.0)
        p = self_collision_penalty(forces, threshold=1.0, scale=1.0, continuous=False, exclude_base=False)
        assert set(p.tolist()).issubset({0.0, 1.0})


# -------------------------------------------------------------------
# obstacle_distance_reward
# -------------------------------------------------------------------

class TestObstacleDistanceReward:
    def test_beyond_safety_radius_positive(self):
        dist = torch.ones(N) * 2.0  # 2 m > 1 m safety radius
        r = obstacle_distance_reward(dist, safety_radius=1.0, scale=1.0)
        assert (r > 0).all()

    def test_at_contact_negative(self):
        dist = torch.zeros(N)  # At contact
        r = obstacle_distance_reward(dist, safety_radius=1.0, scale=1.0)
        assert (r < 0).all()

    def test_monotonically_increasing(self):
        dists = torch.linspace(0.0, 3.0, 20)
        r = obstacle_distance_reward(dists, safety_radius=1.0, scale=1.0)
        assert (r[1:] >= r[:-1]).all()


# -------------------------------------------------------------------
# velocity_limit_penalty
# -------------------------------------------------------------------

class TestVelocityLimitPenalty:
    def test_within_limits_no_penalty(self):
        lin_vel = torch.zeros(N, 3)
        joint_vel = torch.zeros(N, 6)
        p = velocity_limit_penalty(
            lin_vel, joint_vel, max_linear_vel=1.5, max_joint_vel=3.0, scale=1.0
        )
        assert torch.allclose(p, torch.zeros(N))

    def test_exceed_limit_penalized(self):
        lin_vel = torch.ones(N, 3) * 10.0  # Way over limit
        joint_vel = torch.zeros(N, 6)
        p = velocity_limit_penalty(
            lin_vel, joint_vel, max_linear_vel=1.5, max_joint_vel=3.0, scale=1.0
        )
        assert (p > 0.0).all()


# -------------------------------------------------------------------
# joint_limit_penalty
# -------------------------------------------------------------------

class TestJointLimitPenalty:
    def test_within_limits_no_penalty(self):
        n_joints = 6
        joint_pos = torch.zeros(N, n_joints)
        lower = torch.ones(n_joints) * -1.0
        upper = torch.ones(n_joints) * 1.0
        p = joint_limit_penalty(joint_pos, lower, upper, margin=0.05, scale=1.0)
        assert torch.allclose(p, torch.zeros(N))

    def test_outside_limits_penalized(self):
        n_joints = 6
        joint_pos = torch.ones(N, n_joints) * 5.0  # Way outside [-1, 1]
        lower = torch.ones(n_joints) * -1.0
        upper = torch.ones(n_joints) * 1.0
        p = joint_limit_penalty(joint_pos, lower, upper, margin=0.05, scale=1.0)
        assert (p > 0.0).all()


# -------------------------------------------------------------------
# lateral_motion_penalty
# -------------------------------------------------------------------

class TestLateralMotionPenalty:
    def test_pure_forward_no_penalty(self):
        """Robot facing +x, moving purely forward → no lateral penalty."""
        lin_vel = torch.zeros(N, 3)
        lin_vel[:, 0] = 1.0  # Forward only
        quat = _unit_quat()  # Identity = facing +x
        p = lateral_motion_penalty(lin_vel, quat, scale=1.0)
        assert torch.allclose(p, torch.zeros(N), atol=1e-5)

    def test_pure_lateral_penalized(self):
        """Robot facing +x, moving purely sideways → penalty."""
        lin_vel = torch.zeros(N, 3)
        lin_vel[:, 1] = 1.0  # Sideways only
        quat = _unit_quat()
        p = lateral_motion_penalty(lin_vel, quat, scale=1.0)
        assert (p > 0.0).all()


# -------------------------------------------------------------------
# stability_penalty
# -------------------------------------------------------------------

class TestStabilityPenalty:
    def test_zero_velocity_no_penalty(self):
        p = stability_penalty(torch.zeros(N, 3), torch.zeros(N, 3), scale=1.0)
        assert torch.allclose(p, torch.zeros(N))

    def test_high_velocity_penalized(self):
        p = stability_penalty(torch.ones(N, 3) * 10.0, torch.ones(N, 3) * 5.0, scale=1.0)
        assert (p > 0.0).all()


# -------------------------------------------------------------------
# jerk_penalty
# -------------------------------------------------------------------

class TestJerkPenalty:
    def test_constant_accel_no_jerk(self):
        accel = torch.ones(N, 3)
        p = jerk_penalty(accel, accel, dt=0.05, max_jerk=100.0, scale=1.0)
        assert torch.allclose(p, torch.zeros(N))

    def test_high_jerk_penalized(self):
        a_prev = torch.zeros(N, 3)
        a_curr = torch.ones(N, 3) * 1000.0
        p = jerk_penalty(a_curr, a_prev, dt=0.05, max_jerk=1.0, scale=1.0)
        assert (p > 0.0).all()
