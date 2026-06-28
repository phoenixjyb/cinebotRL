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
        dt: float = 0.05,  # 20Hz control frequency (changed from 0.02 @ 50Hz)
        waypoint_dt: float | None = None,
        waypoint_file: str | None = None,
        trajectory_dir: str | None = None,
        trajectory_pattern: str = "**/*.json",
        trajectory_manifest_file: str | None = None,
        trajectory_filter_indices: list[int] | None = None,
        max_trajectories: int | None = None,
        min_duration_seconds: float = 0.0,
        randomize_start_waypoint: bool = False,
        start_waypoint_min_fraction: float = 0.0,
        start_waypoint_max_fraction: float = 0.0,
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
            trajectory_manifest_file: Optional newline-delimited trajectory manifest
            trajectory_filter_indices: Filter to specific trajectory indices from analysis
            max_trajectories: Maximum number of trajectories to load
            min_duration_seconds: Reject recorded trajectories shorter than this duration
            randomize_start_waypoint: Reset recorded trajectories at a later waypoint
            start_waypoint_min_fraction: Minimum start waypoint as trajectory fraction
            start_waypoint_max_fraction: Maximum start waypoint as trajectory fraction
        """
        self.traj_type = traj_type
        self.num_envs = num_envs
        self.device = device
        self.amplitude = amplitude
        self.speed = speed
        self.height = height
        self.dt = dt
        self.waypoint_dt = waypoint_dt if waypoint_dt is not None else dt
        self.min_duration_seconds = min_duration_seconds
        self.randomize_start_waypoint = randomize_start_waypoint
        self.start_waypoint_min_fraction = start_waypoint_min_fraction
        self.start_waypoint_max_fraction = start_waypoint_max_fraction
        
        # Phase tracking (one per environment)
        self.phase = torch.zeros(num_envs, device=device)
        
        # Center offset for each environment (randomization)
        self.center_x = torch.zeros(num_envs, device=device)
        self.center_y = torch.zeros(num_envs, device=device)
        
        # Recorded trajectory data (single trajectory mode)
        self.recorded_positions = None
        self.recorded_orientations = None
        self.recorded_lengths = None
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
                trajectory_manifest_file,
                trajectory_filter_indices,
                max_trajectories,
                min_duration_seconds,
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
            self._randomize_reset_waypoint(env_ids)
        elif self.traj_type == "recorded":
            self._randomize_reset_waypoint(env_ids)

    def _randomize_reset_waypoint(self, env_ids: torch.Tensor) -> None:
        """Optionally start recorded playback from a later waypoint."""
        if (
            not self.randomize_start_waypoint
            or self.recorded_positions is None
            or env_ids.numel() == 0
        ):
            return

        max_length = self.recorded_positions.shape[1]
        if max_length <= 1:
            return

        if self.recorded_lengths is None:
            real_lengths = torch.full((len(env_ids),), max_length, dtype=torch.long, device=self.device)
        else:
            real_lengths = self.recorded_lengths[env_ids].to(torch.long)
            real_lengths = torch.clamp(real_lengths, min=1, max=max_length)

        min_frac = max(0.0, min(1.0, float(self.start_waypoint_min_fraction)))
        max_frac = max(0.0, min(1.0, float(self.start_waypoint_max_fraction)))
        if max_frac < min_frac:
            min_frac, max_frac = max_frac, min_frac

        last_real_idx = real_lengths - 1
        min_idx = torch.round(min_frac * last_real_idx.float()).to(torch.long)
        max_idx = torch.round(max_frac * last_real_idx.float()).to(torch.long)
        max_idx = torch.maximum(min_idx, torch.minimum(max_idx, last_real_idx))

        span = torch.clamp(max_idx - min_idx + 1, min=1)
        start_idx = min_idx + torch.floor(torch.rand(len(env_ids), device=self.device) * span.float()).to(torch.long)

        self.current_waypoint_idx[env_ids] = start_idx
        self._recorded_time_accum[env_ids] = 0.0
        
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
        """Playback recorded trajectory from waypoints with smooth interpolation.
        
        Interpolates between waypoints to provide smooth target motion at control frequency,
        eliminating step-wise jumps that confuse the policy.
        
        Returns:
            position: Target positions [num_envs, 3]
            orientation: Target orientations [num_envs, 4]
        """
        if self.recorded_positions is None:
            raise RuntimeError("No recorded trajectory loaded. Set waypoint_file in config.")
        
        batch_indices = torch.arange(self.num_envs, device=self.device)
        max_length = self.recorded_positions.shape[1]
        
        # Get current and next waypoint indices
        current_idx = self.current_waypoint_idx
        next_idx = (current_idx + 1) % max_length
        
        # Safety: Clamp indices to valid range
        current_idx = torch.clamp(current_idx, 0, max_length - 1)
        next_idx = torch.clamp(next_idx, 0, max_length - 1)
        
        # Interpolation factor (0.0 to 1.0 between waypoints)
        # _recorded_time_accum accumulates control_dt (0.05s @ 20Hz) until it reaches waypoint_dt (0.1s)
        alpha = torch.clamp(self._recorded_time_accum / self.waypoint_dt, 0.0, 1.0)
        
        # Linear interpolation for positions
        pos_current = self.recorded_positions[batch_indices, current_idx]
        pos_next = self.recorded_positions[batch_indices, next_idx]
        position = (1.0 - alpha.unsqueeze(-1)) * pos_current + alpha.unsqueeze(-1) * pos_next
        
        # Spherical linear interpolation (slerp) for orientations
        quat_current = self.recorded_orientations[batch_indices, current_idx]
        quat_next = self.recorded_orientations[batch_indices, next_idx]
        orientation = self._slerp_quaternions(quat_current, quat_next, alpha)
        
        return position, orientation
    
    def _slerp_quaternions(
        self, q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Spherical linear interpolation between quaternions.
        
        Args:
            q1: Start quaternions [num_envs, 4] (wxyz)
            q2: End quaternions [num_envs, 4] (wxyz)
            t: Interpolation factors [num_envs] (0.0 to 1.0)
            
        Returns:
            Interpolated quaternions [num_envs, 4]
        """
        # Safety: Ensure inputs are valid
        if q1.numel() == 0 or q2.numel() == 0 or t.numel() == 0:
            print(f"[WARNING _slerp_quaternions] Empty tensor input: q1={q1.shape}, q2={q2.shape}, t={t.shape}")
            return q1 if q1.numel() > 0 else torch.zeros((0, 4), device=self.device)
        
        # Safety: Clamp t to valid range
        t = torch.clamp(t, 0.0, 1.0)
        
        # Compute dot product
        dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
        
        # If dot < 0, negate q2 to take shorter path
        q2 = torch.where(dot < 0, -q2, q2)
        dot = torch.abs(dot)
        
        # Threshold for linear interpolation (quaternions very close)
        DOT_THRESHOLD = 0.9995
        
        # For quaternions very close together, use linear interpolation
        use_linear = dot > DOT_THRESHOLD
        
        # Slerp calculation
        theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta = torch.sin(theta)
        
        # Avoid division by zero
        sin_theta = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)
        
        t_expanded = t.unsqueeze(-1)
        w1 = torch.sin((1.0 - t_expanded) * theta) / sin_theta
        w2 = torch.sin(t_expanded * theta) / sin_theta
        
        result_slerp = w1 * q1 + w2 * q2
        
        # Linear interpolation fallback
        result_linear = (1.0 - t_expanded) * q1 + t_expanded * q2
        result_linear = result_linear / torch.norm(result_linear, dim=-1, keepdim=True)
        
        # Choose based on threshold
        result = torch.where(use_linear, result_linear, result_slerp)
        
        # Normalize to ensure unit quaternion
        result = result / torch.norm(result, dim=-1, keepdim=True)
        
        return result
    
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
        
        duration_seconds = len(poses) * self.waypoint_dt
        if duration_seconds < self.min_duration_seconds:
            raise ValueError(
                f"Recorded trajectory {waypoint_file} is too short: "
                f"{duration_seconds:.2f}s < {self.min_duration_seconds:.2f}s"
            )

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
        self.recorded_lengths = torch.full(
            (self.num_envs,),
            len(poses),
            dtype=torch.long,
            device=self.device,
        )
        
        print(f"[TrajectoryManager] Loaded {len(poses)} waypoints from {waypoint_file}")
        print(f"[TrajectoryManager] Recorded duration: {duration_seconds:.2f}s")
        print(f"[TrajectoryManager] Position range: {positions_array.min(dim=0)[0]} to {positions_array.max(dim=0)[0]}")
    
    def _init_multi_trajectory(
        self, 
        trajectory_dir: str,
        pattern: str = "**/*.json",
        manifest_file: str | None = None,
        filter_indices: list[int] | None = None,
        max_trajectories: int | None = None,
        min_duration_seconds: float = 0.0
    ) -> None:
        """Initialize multi-trajectory loader.
        
        Args:
            trajectory_dir: Directory containing multiple trajectory JSON files
            pattern: Glob pattern for finding trajectories
            manifest_file: Optional newline-delimited trajectory manifest
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
            manifest_file=manifest_file,
            waypoint_dt=self.waypoint_dt,
            min_duration_seconds=min_duration_seconds,
        )
        
        # Sample initial trajectories for all environments
        self._resample_multi_trajectories()
        
        # Debug: Print first waypoint of first few environments
        if self.recorded_positions is not None:
            print(f"[TrajectoryManager] First waypoints after initialization:")
            for i in range(min(3, self.num_envs)):
                first_wp = self.recorded_positions[i, 0].cpu().numpy()
                print(f"  Env {i}: [{first_wp[0]:.3f}, {first_wp[1]:.3f}, {first_wp[2]:.3f}]")
    
    def _resample_multi_trajectories(self, env_ids: torch.Tensor | None = None) -> None:
        """Resample trajectories for specified environments.
        
        Args:
            env_ids: Environment IDs to resample (None = all)
        """
        print(f"[DEBUG _resample_multi_trajectories] Called with env_ids={'ALL' if env_ids is None else f'{len(env_ids)} envs (first: {env_ids[0].item() if len(env_ids) > 0 else None})'}")
        
        if self.multi_loader is None:
            print(f"[DEBUG _resample_multi_trajectories] multi_loader is None! Exiting.")
            return
        
        if env_ids is None:
            # Resample all environments
            positions, orientations, lengths = self.multi_loader.sample_trajectories_with_lengths(self.num_envs)
            self.recorded_positions = positions
            self.recorded_orientations = orientations
            self.recorded_lengths = lengths
            self.current_waypoint_idx.zero_()
            self._recorded_time_accum.zero_()
        else:
            # FIX (8c-v2): For partial resets, only resample the envs that need new trajectories
            # This fixes the performance issue and prevents non-reset envs from getting new trajectories
            
            num_to_resample = len(env_ids)
            print(f"[TrajectoryManager] Partial reset: resampling {num_to_resample} envs (was resampling all {self.num_envs})")
            
            # Sample only the trajectories needed for reset envs
            positions, orientations, lengths = self.multi_loader.sample_trajectories_with_lengths(num_to_resample)
            
            # Get current max_length from existing trajectories
            current_max_length = self.recorded_positions.shape[1] if self.recorded_positions is not None else positions.shape[1]
            new_max_length = positions.shape[1]
            
            # Handle shape mismatch if new trajectories have different length
            if new_max_length != current_max_length:
                # Need to resize buffers - resample ALL envs (rare case)
                print(f"[TrajectoryManager] Max length changed {current_max_length} -> {new_max_length}, resampling all envs")
                positions, orientations, lengths = self.multi_loader.sample_trajectories_with_lengths(self.num_envs)
                self.recorded_positions = positions
                self.recorded_orientations = orientations
                self.recorded_lengths = lengths
                self.current_waypoint_idx[env_ids] = 0
                self._recorded_time_accum[env_ids] = 0.0
            else:
                # Shape matches - insert new trajectories only at reset env indices
                if self.recorded_positions is None:
                    # First call, initialize buffers
                    self.recorded_positions = positions
                    self.recorded_orientations = orientations
                    self.recorded_lengths = lengths
                else:
                    # Insert sampled trajectories into the correct positions
                    self.recorded_positions[env_ids] = positions
                    self.recorded_orientations[env_ids] = orientations
                    if self.recorded_lengths is None:
                        self.recorded_lengths = torch.full(
                            (self.num_envs,),
                            self.recorded_positions.shape[1],
                            dtype=torch.long,
                            device=self.device,
                        )
                    self.recorded_lengths[env_ids] = lengths
                
                self.current_waypoint_idx[env_ids] = 0
                self._recorded_time_accum[env_ids] = 0.0
            
            # Debug: Print first waypoint for reset environments
            if len(env_ids) > 0:
                first_env = env_ids[0].item()
                first_wp = self.recorded_positions[first_env, 0].cpu().numpy()
                print(f"[TrajectoryManager] Resampled Env {first_env}: First waypoint [{first_wp[0]:.3f}, {first_wp[1]:.3f}, {first_wp[2]:.3f}]")

