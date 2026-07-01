"""Tests for recorded trajectory playback helpers."""

from __future__ import annotations

import torch

from rl_platform.tasks.mobile_mm.trajectories import TrajectoryManager


def _unit_orientations(num_envs: int, num_steps: int) -> torch.Tensor:
    quats = torch.zeros(num_envs, num_steps, 4)
    quats[:, :, 0] = 1.0
    return quats


def test_recorded_lookahead_and_step_wrap_use_real_lengths():
    """Variable-length recorded demos should wrap before padded tail waypoints."""
    manager = TrajectoryManager(
        "recorded",
        num_envs=2,
        device="cpu",
        dt=0.05,
        waypoint_dt=0.1,
    )
    manager.recorded_positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [99.0, 0.0, 0.0],
                [99.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [4.0, 1.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    manager.recorded_orientations = _unit_orientations(num_envs=2, num_steps=5)
    manager.recorded_lengths = torch.tensor([3, 5], dtype=torch.long)
    manager.current_waypoint_idx = torch.tensor([2, 4], dtype=torch.long)

    lookahead, _ = manager.get_lookahead(steps=2, lookahead_dt=0.1)
    assert torch.allclose(lookahead[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(lookahead[0, 1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(lookahead[1, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(lookahead[1, 1], torch.tensor([1.0, 1.0, 0.0]))

    manager.step()
    assert manager.current_waypoint_idx.tolist() == [2, 4]
    manager.step()
    assert manager.current_waypoint_idx.tolist() == [0, 0]

