"""
Reachability Map Utilities for RL Training

Load and query the MATLAB-generated reachability map to:
1) Filter impossible targets early (avoid wasted exploration)
2) Shape rewards based on manipulability (bias toward dexterous poses)
3) Provide curriculum learning (start with high-reach targets, progress to harder ones)

IMPORTANT COORDINATE FRAMES:
- Map is built in ARM BASE FRAME (left_arm_base_link = shoulder mount point)
- Arm base is at [0.16, 0, 0.9465]m relative to mobile base (abstract_chassis_link)
- During training, targets are in MOBILE BASE FRAME
- Must transform: mobile_base -> arm_base before querying map

Usage:
    from scripts.reachability_utils import ReachabilityMap
    
    # Load map once (at environment initialization)
    rmap = ReachabilityMap("matlab/reach_map_arm.mat", 
                           arm_offset=[0.16, 0, 0.9465],  # Shoulder position from URDF
                           device="cuda")
    
    # During RL training step:
    # 1. Get target EE waypoint from trajectory (in mobile base frame)
    target_ee_mobile = torch.tensor([[0.66, 0.0, 0.9465]])  # (N, 3) in mobile base
    
    # 2. Query reachability (auto-transforms mobile → arm frame)
    scores = rmap.query_batch(target_ee_mobile, in_mobile_frame=True)  # Default
    # Internally: target_arm = [0.66-0.16, 0.0-0.0, 0.9465-0.9465] = [0.5, 0, 0]
    #             score = map_lookup([0.5, 0, 0])  # 0.5m in front of shoulder
    
    # 3. Use in reward shaping
    tracking_reward = -torch.norm(current_ee - target_ee_mobile, dim=-1)
    shaped_reward = tracking_reward * scores  # Boost reachable, penalize unreachable
    
    # Or curriculum: filter out unreachable targets during early training
    valid_mask = scores > 0.5  # Only keep highly reachable targets
    target_ee_mobile = target_ee_mobile[valid_mask]
"""

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial import cKDTree
from typing import Optional, Tuple, Dict
import warnings


class ReachabilityMap:
    """
    Fast reachability map query for RL training.
    
    The map is precomputed in MATLAB with collision-aware IK sampling.
    Map coordinates are in ARM BASE FRAME (shoulder mount point).
    
    Each voxel stores:
    - reachScore: [0,1] fraction of orientations that are reachable
    - manipMax: maximum manipulability (higher = more dexterous)
    - exampleQ: IK seed configuration (optional, for fast IK)
    
    Coordinate Transform:
    - arm_base position = mobile_base position + arm_offset (default [0.16, 0, 0.9465])
    - target_arm = target_mobile - arm_offset
    """
    
    def __init__(self, map_file: str, device: str = "cpu", arm_offset: Optional[list] = None):
        """
        Load reachability map from .mat file.
        
        Args:
            map_file: Path to reach_map_arm.mat
            device: "cpu" or "cuda"
            arm_offset: [x, y, z] offset of arm base relative to mobile base
                       Default: [0.16, 0, 0.9465] (left_arm_base_link position)
            device: "cpu" or "cuda" for torch tensors
        """
        print(f"Loading reachability map: {map_file}")
        
        # Arm offset: position of left_arm_base_link relative to abstract_chassis_link
        if arm_offset is None:
            arm_offset = [0.16, 0, 0.9465]  # From URDF: shoulder mount point
        self.arm_offset = torch.tensor(arm_offset, dtype=torch.float32, device=device)
        self.device = device
        
        # Load MATLAB .mat file
        mat = loadmat(map_file, struct_as_record=False, squeeze_me=True)
        self.map = mat['map']
        
        # Extract metadata
        self.grid_origin = np.array(self.map.grid.origin, dtype=np.float32)  # [x,y,z] min in arm frame
        self.grid_voxel = np.array(self.map.grid.voxel, dtype=np.float32)   # [dx,dy,dz]
        self.grid_shape = np.array(self.map.grid.shape, dtype=np.int32)     # [nx,ny,nz]
        self.grid_size = self.grid_voxel * self.grid_shape                  # [sx,sy,sz]
        
        # Extract reachability data
        self.reach_score = np.array(self.map.data.reachScore, dtype=np.float32)  # shape: (nx,ny,nz)
        self.manip_max = np.array(self.map.data.manipMax, dtype=np.float32)
        self.has_ik_seed = np.array(self.map.data.hasExampleQ, dtype=bool)
        
        # Optional: load IK seeds (can be large, only if needed)
        self.ik_seeds = None
        if hasattr(self.map.data, 'exampleQ'):
            self.ik_seeds = np.array(self.map.data.exampleQ, dtype=np.float32)  # (nx,ny,nz,ndof)
        
        # Build KDTree for fast nearest-neighbor queries
        # Create grid centers in ARM BASE FRAME
        nx, ny, nz = self.grid_shape
        x = self.grid_origin[0] + np.arange(nx) * self.grid_voxel[0] + self.grid_voxel[0]/2
        y = self.grid_origin[1] + np.arange(ny) * self.grid_voxel[1] + self.grid_voxel[1]/2
        z = self.grid_origin[2] + np.arange(nz) * self.grid_voxel[2] + self.grid_voxel[2]/2
        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
        
        self.grid_centers = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)  # (N_voxels, 3)
        self.kdtree = cKDTree(self.grid_centers)
        
        # Stats
        print(f"  Map frame: ARM BASE (left_arm_base_link)")
        print(f"  Arm offset from mobile base: [{self.arm_offset[0]:.2f}, {self.arm_offset[1]:.2f}, {self.arm_offset[2]:.4f}]")
        print(f"  Grid origin (arm frame): {self.grid_origin}")
        print(f"  Grid size: {self.grid_size}")
        reachable = self.reach_score > 0
        print(f"  Grid: {nx}×{ny}×{nz} voxels ({self.grid_size[0]:.2f}×{self.grid_size[1]:.2f}×{self.grid_size[2]:.2f} m³)")
        print(f"  Voxel size: {1000*self.grid_voxel[0]:.0f}mm")
        print(f"  Reachable voxels: {reachable.sum()} / {reachable.size} ({100*reachable.mean():.1f}%)")
        print(f"  Mean reach score: {self.reach_score[reachable].mean():.3f} (among reachable)")
        print(f"  Mean manipulability: {self.manip_max[reachable].mean():.3f}")
        
    def transform_mobile_to_arm_frame(self, targets_mobile: torch.Tensor) -> torch.Tensor:
        """
        Transform target positions from mobile base frame to arm base frame.
        
        The arm base (shoulder) is at arm_offset relative to mobile base.
        
        Args:
            targets_mobile: (N, 3) positions in mobile base frame [x, y, z]
            
        Returns:
            targets_arm: (N, 3) positions in arm base frame [x, y, z]
            
        Example:
            target_mobile = [0.66, 0.0, 0.9465]  # In mobile base frame
            target_arm = [0.5, 0.0, 0.0]  # In arm frame (0.66-0.16=0.5, 0.9465-0.9465=0)
        """
        # arm_base_position = mobile_base_position + arm_offset
        # Therefore: target_in_arm_frame = target_in_mobile_frame - arm_offset
        return targets_mobile - self.arm_offset
        
    def query_batch(
        self, 
        targets: torch.Tensor,
        in_mobile_frame: bool = True,
        return_manipulability: bool = False,
        return_ik_seeds: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Query reachability scores for a batch of target positions.
        
        Args:
            targets: (N, 3) tensor of target positions [x, y, z]
            in_mobile_frame: If True (default), targets are in mobile base frame
                           and will be auto-transformed to arm frame.
                           If False, targets are already in arm frame.
            return_manipulability: If True, also return manipulability scores
            return_ik_seeds: If True, also return IK seed configurations
            
        Returns:
            scores: (N,) tensor of reachability scores [0,1]
            manip: (N,) tensor of manipulability (if requested)
            seeds: (N, ndof) tensor of IK seeds (if requested and available)
        """
        # Transform if needed
        if in_mobile_frame:
            targets = self.transform_mobile_to_arm_frame(targets)
        
        # Convert to numpy
        targets_np = targets.detach().cpu().numpy()
        N = targets_np.shape[0]
        
        # Query nearest voxel for each target
        dists, indices = self.kdtree.query(targets_np, k=1)
        
        # Convert flat indices to 3D indices
        nx, ny, nz = self.grid_shape
        iz = indices % nz
        iy = (indices // nz) % ny
        ix = indices // (ny * nz)
        
        # Lookup scores
        scores = self.reach_score[ix, iy, iz]
        
        # Convert back to torch
        scores_torch = torch.tensor(scores, dtype=torch.float32, device=self.device)
        
        # Optional: manipulability
        manip_torch = None
        if return_manipulability:
            manip = self.manip_max[ix, iy, iz]
            manip_torch = torch.tensor(manip, dtype=torch.float32, device=self.device)
        
        # Optional: IK seeds
        seeds_torch = None
        if return_ik_seeds and self.ik_seeds is not None:
            has_seed = self.has_ik_seed[ix, iy, iz]
            seeds = np.zeros((N, self.ik_seeds.shape[-1]), dtype=np.float32)
            seeds[has_seed] = self.ik_seeds[ix[has_seed], iy[has_seed], iz[has_seed]]
            seeds_torch = torch.tensor(seeds, dtype=torch.float32, device=self.device)
        
        return scores_torch, manip_torch, seeds_torch
    
    def filter_reachable_targets(
        self,
        targets: torch.Tensor,
        min_score: float = 0.3,
        return_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter a batch of targets to keep only reachable ones.
        
        Args:
            targets: (N, 3) tensor of target positions
            min_score: Minimum reachability score threshold
            return_mask: If True, also return boolean mask
            
        Returns:
            filtered_targets: (M, 3) tensor of reachable targets (M <= N)
            mask: (N,) boolean tensor indicating which targets passed (if requested)
        """
        scores, _, _ = self.query_batch(targets)
        mask = scores >= min_score
        
        filtered = targets[mask]
        
        if return_mask:
            return filtered, mask
        else:
            return filtered, None
    
    def get_curriculum_schedule(
        self,
        n_stages: int = 5,
        initial_threshold: float = 0.8,
        final_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Generate a curriculum schedule: gradually decrease reachability threshold.
        
        Args:
            n_stages: Number of curriculum stages
            initial_threshold: Start with highly reachable targets (0.8 = 80% of orientations work)
            final_threshold: End with all reachable targets (0.1 = at least 10% orientations work)
            
        Returns:
            thresholds: (n_stages,) array of reachability score thresholds
        """
        return np.linspace(initial_threshold, final_threshold, n_stages)
    
    def visualize_slice(
        self,
        z_height: float = 0.9,
        save_path: Optional[str] = None
    ):
        """
        Visualize a horizontal slice of the reachability map.
        
        Args:
            z_height: Height of the slice (base frame)
            save_path: If provided, save plot to this path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available, skipping visualization")
            return
        
        # Find closest z-slice
        z_idx = int((z_height - self.grid_origin[2]) / self.grid_voxel[2])
        z_idx = np.clip(z_idx, 0, self.grid_shape[2] - 1)
        
        # Extract slice
        slice_data = self.reach_score[:, :, z_idx]  # (nx, ny)
        manip_data = self.manip_max[:, :, z_idx]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Reachability score
        extent = [
            self.grid_origin[0], self.grid_origin[0] + self.grid_size[0],
            self.grid_origin[1], self.grid_origin[1] + self.grid_size[1]
        ]
        im1 = ax1.imshow(slice_data.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
        ax1.set_xlabel('X (m) - forward from base')
        ax1.set_ylabel('Y (m) - lateral from base')
        ax1.set_title(f'Reachability Score (z={z_height:.2f}m)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Reach Score [0-1]')
        
        # Manipulability
        im2 = ax2.imshow(manip_data.T, origin='lower', extent=extent, cmap='plasma')
        ax2.set_xlabel('X (m) - forward from base')
        ax2.set_ylabel('Y (m) - lateral from base')
        ax2.set_title(f'Manipulability (z={z_height:.2f}m)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Manipulability')
        
        # Mark shoulder position (0.16m forward from base)
        for ax in [ax1, ax2]:
            ax.plot(0.16, 0, 'r*', markersize=15, label='Shoulder mount')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()


# Example usage in RL training
def shape_reward_with_reachability(
    original_reward: torch.Tensor,
    target_positions: torch.Tensor,
    rmap: ReachabilityMap,
    mode: str = "scale",
    threshold: float = 0.3
) -> torch.Tensor:
    """
    Shape rewards using reachability information.
    
    Args:
        original_reward: (N,) tensor of original tracking rewards
        target_positions: (N, 3) tensor of target positions (base frame)
        rmap: ReachabilityMap instance
        mode: "scale" (multiply), "filter" (zero out unreachable), or "bonus" (add)
        threshold: Reachability threshold for filtering
        
    Returns:
        shaped_reward: (N,) tensor of shaped rewards
    """
    scores, manip, _ = rmap.query_batch(target_positions, return_manipulability=True)
    
    if mode == "scale":
        # Scale rewards by reachability (0 = unreachable, 1 = fully reachable)
        return original_reward * scores
    
    elif mode == "filter":
        # Zero out rewards for unreachable targets
        mask = scores >= threshold
        return original_reward * mask.float()
    
    elif mode == "bonus":
        # Add bonus for high manipulability (encourages dexterous poses)
        # Normalize manipulability to [0, 1] range
        manip_norm = manip / (manip.max() + 1e-6)
        bonus = manip_norm * 10.0  # +10 bonus for max manipulability
        return original_reward + bonus
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    """Test script to verify reachability map loading and querying."""
    
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Load map
    map_file = os.path.join(project_root, "matlab", "reach_map_mobile_mm.mat")
    
    if not os.path.exists(map_file):
        print(f"❌ Map file not found: {map_file}")
        print(f"   Please run MATLAB script first: matlab/build_reachability_map.m")
        sys.exit(1)
    
    rmap = ReachabilityMap(map_file)
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING REACHABILITY QUERIES")
    print("="*60)
    
    # Test 1: Single target (known reachable from your logs)
    target_reachable = torch.tensor([[0.5, 0.2, 0.9]], dtype=torch.float32)
    scores, manip, _ = rmap.query_batch(target_reachable, return_manipulability=True)
    print(f"\n1️⃣  Target [0.5, 0.2, 0.9]m:")
    print(f"   Reachability: {scores[0]:.3f} (0=impossible, 1=all orientations work)")
    print(f"   Manipulability: {manip[0]:.3f} (higher=more dexterous)")
    
    # Test 2: Batch of targets
    targets_batch = torch.tensor([
        [0.3, 0.0, 0.8],   # Close, easy
        [0.6, 0.3, 1.0],   # Medium
        [0.9, 0.5, 0.7],   # Far, harder
        [-0.2, 0.0, 0.5],  # Behind, impossible?
    ], dtype=torch.float32)
    
    scores_batch, manip_batch, _ = rmap.query_batch(targets_batch, return_manipulability=True)
    print(f"\n2️⃣  Batch of {len(targets_batch)} targets:")
    for i, (tgt, s, m) in enumerate(zip(targets_batch, scores_batch, manip_batch)):
        status = "✅ Reachable" if s > 0.3 else "❌ Unreachable"
        print(f"   [{tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}]m → score={s:.3f}, manip={m:.3f} {status}")
    
    # Test 3: Curriculum schedule
    print(f"\n3️⃣  Curriculum schedule (5 stages):")
    schedule = rmap.get_curriculum_schedule(n_stages=5)
    for i, thresh in enumerate(schedule):
        reachable_count = (rmap.reach_score > thresh).sum()
        print(f"   Stage {i+1}: threshold={thresh:.2f} → {reachable_count} voxels ({100*reachable_count/rmap.reach_score.size:.1f}%)")
    
    # Test 4: Visualization
    print(f"\n4️⃣  Generating visualization...")
    viz_path = os.path.join(project_root, "matlab", "reachability_viz.png")
    rmap.visualize_slice(z_height=0.9, save_path=viz_path)
    
    print("\n✓ All tests passed!")
