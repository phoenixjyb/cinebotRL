"""Multi-trajectory loader for training with diverse demonstrations.

This module extends TrajectoryManager to randomly sample from a dataset
of recorded trajectories, enabling diverse training across different
cinematic camera movements.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import torch


class MultiTrajectoryLoader:
    """Loads and manages multiple recorded trajectories for training."""
    
    def __init__(
        self,
        trajectory_dir: str | Path,
        pattern: str = "**/*.json",
        device: str = "cuda",
        max_trajectories: int | None = None,
        filter_by_indices: List[int] | None = None,
        exclude_macosx: bool = True,
    ):
        """Initialize multi-trajectory loader.
        
        Args:
            trajectory_dir: Root directory containing trajectory JSON files
            pattern: Glob pattern to match trajectory files
            device: Torch device for tensors
            max_trajectories: Maximum number of trajectories to load (None = all)
            filter_by_indices: Only load trajectories at these indices (from analysis results)
            exclude_macosx: Filter out __MACOSX files (default True)
        """
        self.device = device
        self.trajectory_dir = Path(trajectory_dir)
        
        # Find all trajectory files
        self.trajectory_files = sorted(self.trajectory_dir.glob(pattern))
        
        # Filter out __MACOSX files
        if exclude_macosx:
            self.trajectory_files = [f for f in self.trajectory_files if '__MACOSX' not in str(f)]
            print(f"[MultiTrajectoryLoader] Found {len(self.trajectory_files)} trajectory files (excluding __MACOSX)")
        else:
            print(f"[MultiTrajectoryLoader] Found {len(self.trajectory_files)} trajectory files")
        
        # Apply index filtering if specified
        if filter_by_indices is not None:
            if not filter_by_indices:
                raise ValueError("filter_by_indices is empty")
            if max(filter_by_indices) >= len(self.trajectory_files):
                raise ValueError(f"Index {max(filter_by_indices)} exceeds available trajectories ({len(self.trajectory_files)})")
            
            self.trajectory_files = [self.trajectory_files[i] for i in filter_by_indices]
            print(f"[MultiTrajectoryLoader] Filtered to {len(self.trajectory_files)} trajectories by indices")
        
        # Limit number of trajectories
        if max_trajectories is not None:
            self.trajectory_files = self.trajectory_files[:max_trajectories]
            print(f"[MultiTrajectoryLoader] Limited to {len(self.trajectory_files)} trajectories")
        
        if not self.trajectory_files:
            raise ValueError(f"No trajectory files found in {trajectory_dir} with pattern {pattern}")
        
        # Pre-load all trajectories
        self.trajectories: List[dict] = []
        self._load_all_trajectories()
    
    def _load_all_trajectories(self):
        """Load all trajectory files into memory."""
        print(f"[MultiTrajectoryLoader] Loading {len(self.trajectory_files)} trajectories...")
        
        for traj_file in self.trajectory_files:
            try:
                with open(traj_file, 'r') as f:
                    data = json.load(f)
                
                poses = data.get("poses", [])
                if not poses:
                    print(f"  ⚠ Skipping {traj_file.name}: no poses found")
                    continue
                
                # Extract positions and orientations
                positions = []
                orientations = []
                
                for pose in poses:
                    pos = pose["position"]
                    ori = pose["orientation"]
                    
                    positions.append(pos)
                    # Convert xyzw → wxyz
                    orientations.append([ori[3], ori[0], ori[1], ori[2]])
                
                # Convert to tensors
                self.trajectories.append({
                    "file": traj_file.name,
                    "category": traj_file.parent.name,
                    "positions": torch.tensor(positions, dtype=torch.float32, device=self.device),
                    "orientations": torch.tensor(orientations, dtype=torch.float32, device=self.device),
                    "length": len(poses),
                })
                
            except Exception as e:
                print(f"  ✗ Error loading {traj_file.name}: {e}")
        
        print(f"[MultiTrajectoryLoader] Successfully loaded {len(self.trajectories)} trajectories")
        
        # Print statistics
        if self.trajectories:
            lengths = [t["length"] for t in self.trajectories]
            categories = set(t["category"] for t in self.trajectories)
            print(f"  - Trajectory lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
            print(f"  - Categories: {len(categories)} - {', '.join(sorted(categories))}")
    
    def sample_trajectory(self) -> dict:
        """Randomly sample one trajectory.
        
        Returns:
            Dictionary with 'positions', 'orientations', 'file', 'category', 'length'
        """
        return random.choice(self.trajectories)
    
    def sample_trajectories(self, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectories for multiple environments.
        
        Each environment gets a randomly selected trajectory. Trajectories are
        replicated/padded to match the longest one in the batch.
        
        Args:
            num_envs: Number of environments
        
        Returns:
            positions: [num_envs, max_length, 3]
            orientations: [num_envs, max_length, 4]
        """
        # Sample one trajectory per environment
        sampled = [self.sample_trajectory() for _ in range(num_envs)]
        
        # Find max length
        max_length = max(t["length"] for t in sampled)
        
        # Pad trajectories to max length
        positions_list = []
        orientations_list = []
        
        for traj in sampled:
            pos = traj["positions"]  # [length, 3]
            ori = traj["orientations"]  # [length, 4]
            
            # Pad by repeating last waypoint
            if traj["length"] < max_length:
                pad_length = max_length - traj["length"]
                pos = torch.cat([pos, pos[-1:].repeat(pad_length, 1)], dim=0)
                ori = torch.cat([ori, ori[-1:].repeat(pad_length, 1)], dim=0)
            
            positions_list.append(pos)
            orientations_list.append(ori)
        
        # Stack into batch
        positions = torch.stack(positions_list, dim=0)  # [num_envs, max_length, 3]
        orientations = torch.stack(orientations_list, dim=0)  # [num_envs, max_length, 4]
        
        return positions, orientations
    
    def get_trajectory_by_category(self, category: str) -> List[dict]:
        """Get all trajectories from a specific category.
        
        Args:
            category: Category name (e.g., 'crane_up', 'dolly_push_in')
        
        Returns:
            List of trajectory dictionaries matching the category
        """
        return [t for t in self.trajectories if t["category"] == category]
    
    def get_categories(self) -> List[str]:
        """Get list of all trajectory categories."""
        return sorted(set(t["category"] for t in self.trajectories))
    
    def __len__(self) -> int:
        """Number of loaded trajectories."""
        return len(self.trajectories)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MultiTrajectoryLoader({len(self)} trajectories, {len(self.get_categories())} categories)"
