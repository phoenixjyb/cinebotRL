"""
Reachability Map Loader and Query Module

Loads MATLAB-generated reachability maps and provides fast queries for:
- Is a target EE position reachable?
- What's the best joint configuration for reaching it?
- How far is it from the reachable workspace?

The reachability map is built in ARM BASE FRAME (shoulder frame), not world frame.
"""

import os
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial import cKDTree
from typing import Tuple, Optional
import h5py


class ReachabilityMap:
    """
    Reachability map for mobile manipulator arm workspace.
    
    The map is defined in the ARM BASE FRAME (left_arm_base_link), which is located at:
    - Translation: [0.16, 0, 0.9465] meters from mobile base (from URDF)
    - Rotation: -90° around Z-axis (RPY [0, 0, -1.5708])
    
    Usage:
        reach_map = ReachabilityMap('reach_map_mobile_mm_arm_only.mat')
        is_reachable = reach_map.query(target_positions_in_arm_frame)
        best_configs = reach_map.get_best_configs(target_positions_in_arm_frame)
    """
    
    def __init__(self, mat_file_path: str, device: str = "cuda:0"):
        """
        Load reachability map from MATLAB .mat file.
        
        Args:
            mat_file_path: Path to .mat file (relative to project root or absolute)
            device: PyTorch device for tensors
        """
        self.device = device
        
        # Handle relative paths
        if not os.path.isabs(mat_file_path):
            # Assume relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            mat_file_path = os.path.join(project_root, mat_file_path)
        
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"Reachability map not found: {mat_file_path}")
        
        print(f"[ReachabilityMap] Loading from: {mat_file_path}")
        
        # Load MATLAB data (handle both v7 and v7.3 formats)
        try:
            # Try v7.3 format (HDF5) first
            with h5py.File(mat_file_path, 'r') as f:
                # Extract grid configuration
                config = f['config']
                self.grid_origin = np.array(config['grid_origin']).flatten()  # [x, y, z] min corner
                self.grid_size = np.array(config['grid_size']).flatten()      # [dx, dy, dz] extent
                self.voxel_size = float(np.array(config['voxel_size']).flatten()[0])
                self.grid_dims = np.array(config['grid_dims']).flatten().astype(int)  # [nx, ny, nz]
                
                # Extract reachability data
                # MATLAB stores as (nz, ny, nx), need (nx, ny, nz)
                self.reach_score = np.transpose(np.array(f['reachScore']), (2, 1, 0))  # (nx, ny, nz)
                self.manip_max = np.transpose(np.array(f['manipMax']), (2, 1, 0))      # (nx, ny, nz)
                
                # qExample has shape (ndof, nz, ny, nx) in HDF5, need (nx, ny, nz, ndof)
                q_raw = np.array(f['qExample'])  # (6, 28, 40, 32) - only ARM joints stored!
                self.q_example = np.transpose(q_raw, (3, 2, 1, 0))  # (32, 40, 28, 6)
        except (OSError, KeyError):
            # Fall back to v7 format
            mat_data = loadmat(mat_file_path)
            
            # Extract grid configuration
            config = mat_data['config'][0, 0]
            self.grid_origin = config['origin'][0]  # [x, y, z] min corner
            self.grid_size = config['size'][0]      # [dx, dy, dz] extent
            self.voxel_size = float(config['voxel_size'][0, 0])
            self.grid_dims = config['dims'][0]      # [nx, ny, nz] number of voxels
            
            # Extract reachability data
            self.reach_score = mat_data['reachScore']  # (nx, ny, nz) - 1.0 if reachable
            self.manip_max = mat_data['manipMax']      # (nx, ny, nz) - manipulability
            self.q_example = mat_data['qExample']      # (nx, ny, nz, ndof) - best joint configs
        
        # Build list of reachable voxel centers and their configs
        reachable_mask = self.reach_score > 0
        n_reachable = np.sum(reachable_mask)
        
        print(f"[ReachabilityMap] Grid: origin={self.grid_origin}, size={self.grid_size}, voxel={self.voxel_size}")
        print(f"[ReachabilityMap] Dimensions: {self.grid_dims} voxels")
        print(f"[ReachabilityMap] Reachable voxels: {n_reachable} / {np.prod(self.grid_dims)}")
        
        # Get indices of reachable voxels
        reachable_indices = np.argwhere(reachable_mask)  # (n_reachable, 3)
        
        # Compute voxel center positions in arm base frame
        voxel_centers = self.grid_origin + (reachable_indices + 0.5) * self.voxel_size
        
        # Extract corresponding joint configurations
        # Note: qExample only stores ARM joints (6), not base joints (3)
        reachable_configs = self.q_example[reachable_mask]  # (n_reachable, 6)
        
        # Build KD-tree for fast nearest-neighbor queries
        self.kdtree = cKDTree(voxel_centers)
        
        # Store as tensors for GPU
        self.reachable_positions = torch.from_numpy(voxel_centers).float().to(device)
        self.reachable_configs = torch.from_numpy(reachable_configs).float().to(device)  # Only arm joints!
        
        print(f"[ReachabilityMap] KD-tree built with {len(self.reachable_positions)} points")
        print(f"[ReachabilityMap] Ready for queries!")
    
    def query(
        self, 
        positions: torch.Tensor,
        tolerance: float = None
    ) -> torch.Tensor:
        """
        Check if positions are reachable.
        
        Args:
            positions: Target positions in ARM BASE FRAME, shape (N, 3) or (3,)
            tolerance: Distance tolerance in meters (default: voxel_size)
        
        Returns:
            is_reachable: Boolean tensor, shape (N,) or scalar
        """
        if tolerance is None:
            tolerance = self.voxel_size
        
        # Handle single position
        single_input = positions.ndim == 1
        if single_input:
            positions = positions.unsqueeze(0)
        
        # Convert to numpy for KD-tree query
        positions_np = positions.detach().cpu().numpy()
        
        # Find nearest reachable voxel for each query position
        distances, indices = self.kdtree.query(positions_np, k=1)
        
        # Check if within tolerance
        is_reachable = torch.from_numpy(distances < tolerance).to(self.device)
        
        if single_input:
            return is_reachable.squeeze(0)
        return is_reachable
    
    def get_best_configs(
        self, 
        positions: torch.Tensor,
        tolerance: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get best joint configurations for reaching target positions.
        
        Args:
            positions: Target positions in ARM BASE FRAME, shape (N, 3) or (3,)
            tolerance: Distance tolerance in meters (default: voxel_size)
        
        Returns:
            configs: ARM joint configurations, shape (N, 6) or (6,) - [j1, j2, j3, j4, j5, j6]
            is_reachable: Whether position is reachable, shape (N,) or scalar
        """
        if tolerance is None:
            tolerance = self.voxel_size
        
        # Handle single position
        single_input = positions.ndim == 1
        if single_input:
            positions = positions.unsqueeze(0)
        
        # Convert to numpy for KD-tree query
        positions_np = positions.detach().cpu().numpy()
        
        # Find nearest reachable voxel
        distances, indices = self.kdtree.query(positions_np, k=1)
        
        # Get corresponding configs (only arm joints!)
        configs = self.reachable_configs[indices]  # (N, 6)
        is_reachable = torch.from_numpy(distances < tolerance).to(self.device)
        
        if single_input:
            return configs.squeeze(0), is_reachable.squeeze(0)
        return configs, is_reachable
    
    def distance_to_workspace(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from positions to nearest reachable voxel.
        
        Args:
            positions: Positions in ARM BASE FRAME, shape (N, 3) or (3,)
        
        Returns:
            distances: Distance to workspace boundary in meters, shape (N,) or scalar
        """
        single_input = positions.ndim == 1
        if single_input:
            positions = positions.unsqueeze(0)
        
        # Convert to numpy for KD-tree query
        positions_np = positions.detach().cpu().numpy()
        
        # Find nearest reachable voxel
        distances, _ = self.kdtree.query(positions_np, k=1)
        
        distances_tensor = torch.from_numpy(distances).float().to(self.device)
        
        if single_input:
            return distances_tensor.squeeze(0)
        return distances_tensor
    
    def world_to_arm_frame(
        self, 
        positions_world: torch.Tensor,
        base_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform positions from world frame to arm base frame.
        
        The arm base frame is offset from mobile base by:
        - Translation: [0.16, 0, 0.9465] meters
        - Rotation: -90° around Z-axis
        
        Args:
            positions_world: Positions in world frame, shape (N, 3)
            base_pose: Mobile base pose [x, y, theta], shape (N, 3)
        
        Returns:
            positions_arm: Positions in arm base frame, shape (N, 3)
        """
        # Arm mount offset from URDF (in mobile base frame)
        arm_translation = torch.tensor([0.16, 0.0, 0.9465], device=self.device)
        
        # Extract base pose
        base_x = base_pose[:, 0:1]  # (N, 1)
        base_y = base_pose[:, 1:2]  # (N, 1)
        base_theta = base_pose[:, 2:3]  # (N, 1)
        
        # Step 1: Transform from world to mobile base frame
        # Rotation: R(-theta)
        cos_theta = torch.cos(-base_theta)
        sin_theta = torch.sin(-base_theta)
        
        # Position relative to base
        pos_rel = positions_world - torch.cat([base_x, base_y, torch.zeros_like(base_x)], dim=1)
        
        # Rotate to base frame
        x_base = pos_rel[:, 0:1] * cos_theta - pos_rel[:, 1:2] * sin_theta
        y_base = pos_rel[:, 0:1] * sin_theta + pos_rel[:, 1:2] * cos_theta
        z_base = pos_rel[:, 2:3]
        
        pos_in_base_frame = torch.cat([x_base, y_base, z_base], dim=1)
        
        # Step 2: Transform from mobile base frame to arm base frame
        # Subtract arm mount translation
        pos_in_arm_mount = pos_in_base_frame - arm_translation
        
        # Rotation: -90° around Z-axis (from URDF: rpy="0 0 -1.5708")
        # R_z(-90°) = [0, 1, 0; -1, 0, 0; 0, 0, 1]
        x_arm = pos_in_arm_mount[:, 1:2]   # y_mount becomes x_arm
        y_arm = -pos_in_arm_mount[:, 0:1]  # -x_mount becomes y_arm
        z_arm = pos_in_arm_mount[:, 2:3]   # z unchanged
        
        positions_arm = torch.cat([x_arm, y_arm, z_arm], dim=1)
        
        return positions_arm
    
    def get_workspace_bounds_world(self, base_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get workspace bounds in world frame for given base pose.
        
        Args:
            base_pose: Mobile base pose [x, y, theta], shape (3,) or (N, 3)
        
        Returns:
            min_bounds: Minimum [x, y, z] in world frame
            max_bounds: Maximum [x, y, z] in world frame
        """
        # Workspace bounds in arm base frame
        arm_min = torch.tensor(self.grid_origin, device=self.device)
        arm_max = torch.tensor(self.grid_origin, device=self.device) + torch.tensor(self.grid_size, device=self.device)
        
        # Transform corners to world frame (simplified - just transform center and use bounds)
        # For visualization purposes
        center_arm = (arm_min + arm_max) / 2
        
        # This is approximate - for exact bounds, would need to transform all 8 corners
        # For now, just give rough bounds
        return arm_min, arm_max


def test_reachability_map():
    """Test function to verify reachability map loading and queries."""
    print("=" * 80)
    print("Testing Reachability Map")
    print("=" * 80)
    
    # Load map
    reach_map = ReachabilityMap('matlab/reach_map_mobile_mm_arm_only.mat')
    
    # Test queries
    test_positions = torch.tensor([
        [0.5, 0.0, 0.0],   # Should be reachable (forward from shoulder)
        [0.0, 0.5, 0.0],   # Should be reachable (side)
        [0.0, 0.0, 0.5],   # Should be reachable (up)
        [2.0, 2.0, 2.0],   # Should be unreachable (too far)
    ], device='cuda:0')
    
    print("\nTest 1: Reachability queries")
    is_reachable = reach_map.query(test_positions)
    for i, (pos, reach) in enumerate(zip(test_positions, is_reachable)):
        print(f"  Position {i}: {pos.cpu().numpy()} -> Reachable: {reach.item()}")
    
    print("\nTest 2: Get best configs")
    configs, is_reachable = reach_map.get_best_configs(test_positions)
    for i, (pos, cfg, reach) in enumerate(zip(test_positions, configs, is_reachable)):
        if reach:
            print(f"  Position {i}: Best arm config = {cfg.cpu().numpy()}")  # 6 arm joints
    
    print("\nTest 3: Distance to workspace")
    distances = reach_map.distance_to_workspace(test_positions)
    for i, (pos, dist) in enumerate(zip(test_positions, distances)):
        print(f"  Position {i}: Distance = {dist.item():.3f} m")
    
    print("\nTest 4: World to arm frame transform")
    world_pos = torch.tensor([[1.0, 0.5, 1.5]], device='cuda:0')  # Some position in world
    base_pose = torch.tensor([[0.0, 0.0, 0.0]], device='cuda:0')  # Base at origin
    arm_pos = reach_map.world_to_arm_frame(world_pos, base_pose)
    print(f"  World position: {world_pos[0].cpu().numpy()}")
    print(f"  Arm frame position: {arm_pos[0].cpu().numpy()}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_reachability_map()
