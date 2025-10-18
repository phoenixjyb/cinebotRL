"""Reference trajectory generators for end-effector tracking."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from typing import Literal


class TrajectoryManager:
    """Manages reference trajectories for end-effector tracking."""
    
    def __init__(
        self,
        traj_type: Literal["line", "circle", "figure_eight", "recorded", "multi_recorded"],
        num_envs: int,
        device: str,
        amplitude: float = 0.5,
        speed: float = 0.2,
        height: float = 1.0,
        dt: float = 0.02,
        waypoint_dt: float | None = None,
        waypoint_file: str | None = None,
        trajectory_dir: str | None = None,
        trajectory_pattern: str = "**/*.json",
        trajectory_filter_indices: list[int] | None = None,
        max_trajectories: int | None = None,
    ):
        """Initialize trajectory manager.
        
        Args:
            traj_type: Type of trajectory to generate
            num_envs: Number of parallel environments
            device: Torch device (cpu/cuda)
            amplitude: Trajectory amplitude in meters
            speed: Trajectory speed in m/s
            height: Height (z-coordinate) of trajectory plane
            dt: Time step in seconds
            waypoint_file: Path to recorded waypoint JSON file (for 'recorded' type)
            trajectory_dir: Directory with multiple trajectories (for 'multi_recorded' type)
            trajectory_pattern: Glob pattern for finding trajectories (default: "**/*.json")
            trajectory_filter_indices: Filter to specific trajectory indices from analysis
            max_trajectories: Maximum number of trajectories to load
        """
        self.traj_type = traj_type
        self.num_envs = num_envs
        self.device = device
        self.amplitude = amplitude
        self.speed = speed
        self.height = height
        self.dt = dt
        self.waypoint_dt = waypoint_dt if waypoint_dt is not None else dt
        
        # Phase tracking (one per environment)
        self.phase = torch.zeros(num_envs, device=device)
        
        # Center offset for each environment (randomization)
        self.center_x = torch.zeros(num_envs, device=device)
        self.center_y = torch.zeros(num_envs, device=device)
        
        # Recorded trajectory data (single trajectory mode)
        self.recorded_positions = None
        self.recorded_orientations = None
        self.current_waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._recorded_time_accum = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
        # Multi-trajectory loader
        self.multi_loader = None
        
        # Load trajectory based on type
        print(f"[TrajectoryManager] Initializing with type='{traj_type}'")
        if traj_type == "recorded" and waypoint_file is not None:
            print(f"[TrajectoryManager] Loading single recorded trajectory from {waypoint_file}")
            self._load_recorded_trajectory(waypoint_file)
        elif traj_type == "multi_recorded" and trajectory_dir is not None:
            print(f"[TrajectoryManager] Loading multi-recorded trajectories from {trajectory_dir}")
            self._init_multi_trajectory(
                trajectory_dir, 
                trajectory_pattern, 
                trajectory_filter_indices,
                max_trajectories
            )
        elif traj_type == "multi_recorded":
            print(f"[TrajectoryManager] ⚠️  WARNING: traj_type='multi_recorded' but trajectory_dir is None!")
        else:
            print(f"[TrajectoryManager] Using parametric trajectory type: {traj_type}")
        
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset trajectory phase for specified environments.
        
        For multi_recorded mode, this also resamples new trajectories.
        
        Args:
            env_ids: Indices of environments to reset
        """
        self.phase[env_ids] = 0.0
        self.current_waypoint_idx[env_ids] = 0
        self._recorded_time_accum[env_ids] = 0.0
        
        # Resample trajectories in multi-trajectory mode
        if self.traj_type == "multi_recorded":
            self._resample_multi_trajectories(env_ids)
        
    def set_center(self, center_x: torch.Tensor, center_y: torch.Tensor) -> None:
        """Set trajectory center offsets.
        
        Args:
            center_x: X-coordinates of trajectory centers [num_envs]
            center_y: Y-coordinates of trajectory centers [num_envs]
        """
        self.center_x = center_x
        self.center_y = center_y
        
    def get_target_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current target position and orientation.
        
        Returns:
            position: Target positions [num_envs, 3]
            orientation: Target orientations as quaternions [num_envs, 4] (wxyz)
        """
        if self.traj_type == "circle":
            return self._circle_trajectory()
        elif self.traj_type == "line":
            return self._line_trajectory()
        elif self.traj_type == "figure_eight":
            return self._figure_eight_trajectory()
        elif self.traj_type in ["recorded", "multi_recorded"]:
            return self._recorded_trajectory()
        else:
            raise NotImplementedError(f"Trajectory type {self.traj_type} not implemented")
    
    def get_lookahead(self, steps: int, lookahead_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Get lookahead target positions.
        
        Args:
            steps: Number of lookahead steps
            lookahead_dt: Time delta between lookahead steps
            
        Returns:
            positions: Lookahead positions [num_envs, steps, 3]
            orientations: Lookahead orientations [num_envs, steps, 4]
        """
        # Save current phase
        original_phase = self.phase.clone()
        
        positions = []
        orientations = []
        
        for i in range(1, steps + 1):
            # Advance phase temporarily
            self.phase = original_phase + (i * lookahead_dt * self.speed / self.amplitude)
            pos, ori = self.get_target_pose()
            positions.append(pos)
            orientations.append(ori)
        
        # Restore original phase
        self.phase = original_phase
        
        # Stack along new dimension
        positions = torch.stack(positions, dim=1)  # [num_envs, steps, 3]
        orientations = torch.stack(orientations, dim=1)  # [num_envs, steps, 4]
        
        return positions, orientations
    
    def step(self) -> None:
        """Advance trajectory by one timestep."""
        # Phase advance rate based on speed and amplitude
        phase_rate = self.speed / self.amplitude if self.amplitude > 0 else 0.0
        self.phase += phase_rate * self.dt
        
        # Wrap phase to [0, 2π]
        self.phase = torch.remainder(self.phase, 2 * np.pi)
        
        # For recorded/multi_recorded trajectories, advance waypoint index with correct cadence
        if self.traj_type in ["recorded", "multi_recorded"] and self.recorded_positions is not None:
            self._recorded_time_accum += self.dt

            steps_to_advance = torch.floor(self._recorded_time_accum / self.waypoint_dt).to(torch.long)

            if torch.any(steps_to_advance > 0):
                self.current_waypoint_idx += steps_to_advance
                self._recorded_time_accum -= steps_to_advance.float() * self.waypoint_dt

                max_length = self.recorded_positions.shape[1]
                if max_length > 0:
                    self.current_waypoint_idx = torch.remainder(self.current_waypoint_idx, max_length)
    
    def _circle_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate circular trajectory.
        
        Returns:
            position: Target positions [num_envs, 3]
            orientation: Target orientations (identity quaternion) [num_envs, 4]
        """
        x = self.center_x + self.amplitude * torch.cos(self.phase)
        y = self.center_y + self.amplitude * torch.sin(self.phase)
        z = torch.full_like(x, self.height)
        
        position = torch.stack([x, y, z], dim=-1)
        
        # Identity quaternion (no rotation constraint for now)
        orientation = torch.zeros(self.num_envs, 4, device=self.device)
        orientation[:, 0] = 1.0  # w=1, x=y=z=0
        
        return position, orientation
    
    def _line_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate linear trajectory (back and forth).
        
        Returns:
            position: Target positions [num_envs, 3]
            orientation: Target orientations [num_envs, 4]
        """
        # Oscillate along x-axis
        x = self.center_x + self.amplitude * torch.sin(self.phase)
        y = self.center_y.clone()
        z = torch.full_like(x, self.height)
        
        position = torch.stack([x, y, z], dim=-1)
        
        # Identity quaternion
        orientation = torch.zeros(self.num_envs, 4, device=self.device)
        orientation[:, 0] = 1.0
        
        return position, orientation
    
    def _figure_eight_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate figure-eight trajectory (Lissajous curve).
        
        Returns:
            position: Target positions [num_envs, 3]
            orientation: Target orientations [num_envs, 4]
        """
        x = self.center_x + self.amplitude * torch.sin(self.phase)
        y = self.center_y + self.amplitude * torch.sin(2 * self.phase) / 2
        z = torch.full_like(x, self.height)
        
        position = torch.stack([x, y, z], dim=-1)
        
        # Identity quaternion
        orientation = torch.zeros(self.num_envs, 4, device=self.device)
        orientation[:, 0] = 1.0
        
        return position, orientation
    
    def _recorded_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Playback recorded trajectory from waypoints.
        
        Returns:
            position: Target positions [num_envs, 3]
            orientation: Target orientations [num_envs, 4]
        """
        if self.recorded_positions is None:
            raise RuntimeError("No recorded trajectory loaded. Set waypoint_file in config.")
        
        # Get current waypoint for each environment
        batch_indices = torch.arange(self.num_envs, device=self.device)
        position = self.recorded_positions[batch_indices, self.current_waypoint_idx]
        orientation = self.recorded_orientations[batch_indices, self.current_waypoint_idx]
        
        return position, orientation
    
    def _load_recorded_trajectory(self, waypoint_file: str) -> None:
        """Load recorded trajectory from JSON file.
        
        Expected format:
        {
            "poses": [
                {
                    "position": [x, y, z],
                    "orientation": [x, y, z, w]  # xyzw order, will convert to wxyz
                },
                ...
            ]
        }
        
        Args:
            waypoint_file: Path to JSON file with recorded poses
        """
        file_path = Path(waypoint_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Waypoint file not found: {waypoint_file}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        poses = data.get("poses", [])
        if not poses:
            raise ValueError(f"No poses found in {waypoint_file}")
        
        # Extract positions and orientations
        positions = []
        orientations = []
        
        for pose in poses:
            pos = pose["position"]
            ori = pose["orientation"]
            
            positions.append(pos)
            
            # Convert from xyzw to wxyz
            orientations.append([ori[3], ori[0], ori[1], ori[2]])
        
        # Convert to tensors [num_waypoints, 3/4]
        positions_array = torch.tensor(positions, dtype=torch.float32, device=self.device)
        orientations_array = torch.tensor(orientations, dtype=torch.float32, device=self.device)
        
        # Replicate for all environments [num_envs, num_waypoints, 3/4]
        self.recorded_positions = positions_array.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        self.recorded_orientations = orientations_array.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        
        print(f"[TrajectoryManager] Loaded {len(poses)} waypoints from {waypoint_file}")
        print(f"[TrajectoryManager] Position range: {positions_array.min(dim=0)[0]} to {positions_array.max(dim=0)[0]}")
    
    def _init_multi_trajectory(
        self, 
        trajectory_dir: str,
        pattern: str = "**/*.json",
        filter_indices: list[int] | None = None,
        max_trajectories: int | None = None
    ) -> None:
        """Initialize multi-trajectory loader.
        
        Args:
            trajectory_dir: Directory containing multiple trajectory JSON files
            pattern: Glob pattern for finding trajectories
            filter_indices: Filter to specific trajectory indices
            max_trajectories: Maximum number of trajectories to load
        """
        from .multi_trajectory import MultiTrajectoryLoader
        
        self.multi_loader = MultiTrajectoryLoader(
            trajectory_dir=trajectory_dir,
            pattern=pattern,
            device=self.device,
            max_trajectories=max_trajectories,
            filter_by_indices=filter_indices,
        )
        
        # Sample initial trajectories for all environments
        self._resample_multi_trajectories()
    
    def _resample_multi_trajectories(self, env_ids: torch.Tensor | None = None) -> None:
        """Resample trajectories for specified environments.
        
        Args:
            env_ids: Environment IDs to resample (None = all)
        """
        if self.multi_loader is None:
            return
        
        if env_ids is None:
            # Resample all environments
            positions, orientations = self.multi_loader.sample_trajectories(self.num_envs)
            self.recorded_positions = positions
            self.recorded_orientations = orientations
            self.current_waypoint_idx.zero_()
            self._recorded_time_accum.zero_()
        else:
            # For partial resets, we need to handle variable trajectory lengths.
            # Since recorded_positions has a fixed shape, we need to resample ALL envs
            # to get a consistent max_length, then only reset the indices for env_ids.
            
            # Resample ALL environments to get new max_length
            positions, orientations = self.multi_loader.sample_trajectories(self.num_envs)
            
            # Keep existing trajectories for non-reset envs by copying their data
            if self.recorded_positions is not None:
                # Find which envs to keep (not in env_ids)
                all_env_ids = torch.arange(self.num_envs, device=self.device)
                keep_mask = ~torch.isin(all_env_ids, env_ids)
                keep_ids = all_env_ids[keep_mask]
                
                # If new max_length matches old, preserve non-reset trajectories
                if positions.shape[1] == self.recorded_positions.shape[1]:
                    positions[keep_ids] = self.recorded_positions[keep_ids]
                    orientations[keep_ids] = self.recorded_orientations[keep_ids]
                # Otherwise, just use all new trajectories (unavoidable due to shape change)
            
            self.recorded_positions = positions
            self.recorded_orientations = orientations
            self.current_waypoint_idx[env_ids] = 0
            self._recorded_time_accum[env_ids] = 0.0


