"""Verify what trajectories the environment is actually loading and using.

This will check:
1. What trajectory type is configured
2. How many trajectories are loaded
3. Sample some trajectory targets to see if they require chassis movement
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    print("="*80)
    print("TRAJECTORY LOADING VERIFICATION")
    print("="*80)
    
    # Initialize Isaac Sim
    print("\n[1/3] Initializing Isaac Sim...")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    import gymnasium as gym
    import torch
    
    # Register tasks
    print("[2/3] Registering tasks...")
    from task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    
    # Create environment with trajectory loading
    print("[3/3] Creating environment with multi_recorded trajectories...")
    env = gym.make(
        "MobileMMTrackEE-v0",
        num_envs=4,
        headless=True,
        trajectory_type="multi_recorded",
        use_all_trajectories=True,
    )
    
    unwrapped = env.unwrapped
    
    print("\n" + "="*80)
    print("ENVIRONMENT CONFIGURATION")
    print("="*80)
    
    # Check trajectory manager
    if hasattr(unwrapped, 'trajectory_manager'):
        traj_mgr = unwrapped.trajectory_manager
        print(f"✅ Trajectory manager found: {type(traj_mgr).__name__}")
        
        # Check what type it is
        if hasattr(traj_mgr, 'trajectory_type'):
            print(f"   Trajectory type: {traj_mgr.trajectory_type}")
        
        # Check if multi-trajectory
        if hasattr(traj_mgr, 'num_trajectories'):
            print(f"   Number of trajectories loaded: {traj_mgr.num_trajectories}")
        
        if hasattr(traj_mgr, 'trajectories'):
            print(f"   Trajectories list length: {len(traj_mgr.trajectories)}")
        
        if hasattr(traj_mgr, 'trajectory_dir'):
            print(f"   Trajectory directory: {traj_mgr.trajectory_dir}")
        
        if hasattr(traj_mgr, 'use_all_trajectories'):
            print(f"   Use all trajectories: {traj_mgr.use_all_trajectories}")
            
        if hasattr(traj_mgr, 'use_chassis_only'):
            print(f"   Use chassis only: {traj_mgr.use_chassis_only}")
    else:
        print("❌ No trajectory manager found!")
    
    print("\n" + "="*80)
    print("SAMPLING TRAJECTORIES")
    print("="*80)
    
    # Reset and sample trajectory targets
    obs = env.reset()
    
    print("\nSampling 100 timesteps from current trajectories:")
    print("-"*80)
    
    initial_ee_pos = unwrapped.robot.data.body_pos_w[0, unwrapped._ee_body_idx, :].cpu().numpy()
    print(f"Initial EE position: {initial_ee_pos}")
    
    target_positions = []
    for step in range(100):
        target_pos, _ = unwrapped.trajectory_manager.get_target_pose()
        target_positions.append(target_pos[0].cpu().numpy())
    
    import numpy as np
    target_positions = np.array(target_positions)
    
    # Analyze target movement
    total_target_movement = np.linalg.norm(target_positions[-1] - target_positions[0])
    max_distance_from_start = np.max(np.linalg.norm(target_positions - target_positions[0], axis=1))
    
    print(f"\nTarget trajectory analysis:")
    print(f"  Start position: {target_positions[0]}")
    print(f"  End position:   {target_positions[-1]}")
    print(f"  Total movement: {total_target_movement:.4f} m")
    print(f"  Max distance from start: {max_distance_from_start:.4f} m")
    
    # Check if target goes far from robot
    distances_from_ee = np.linalg.norm(target_positions - initial_ee_pos, axis=1)
    far_targets = np.sum(distances_from_ee > 1.0)  # More than 1m away
    
    print(f"\n  Targets requiring significant reach:")
    print(f"    > 0.5m away: {np.sum(distances_from_ee > 0.5)} / 100")
    print(f"    > 1.0m away: {np.sum(distances_from_ee > 1.0)} / 100")
    print(f"    > 1.5m away: {np.sum(distances_from_ee > 1.5)} / 100")
    
    if max_distance_from_start < 0.2:
        print("\n  ⚠️  WARNING: Trajectory stays very close to start position!")
        print("      This might be a simple circle/line, not recorded trajectories")
    elif far_targets > 20:
        print("\n  ✅ Trajectory requires significant movement")
        print("      Likely needs chassis movement to track")
    
    # Try resetting to see if different trajectories are loaded
    print("\n" + "="*80)
    print("TESTING TRAJECTORY VARIETY")
    print("="*80)
    
    print("\nResetting environment 5 times to check trajectory sampling...")
    
    for reset_idx in range(5):
        obs = env.reset()
        target_pos, _ = unwrapped.trajectory_manager.get_target_pose()
        print(f"  Reset {reset_idx + 1}: Target = {target_pos[0].cpu().numpy()}")
        
        # If multi-trajectory, check which one was selected
        if hasattr(unwrapped.trajectory_manager, 'current_trajectory_indices'):
            idx = unwrapped.trajectory_manager.current_trajectory_indices[0].item()
            print(f"             Trajectory index = {idx}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if not hasattr(unwrapped, 'trajectory_manager'):
        print("❌ CRITICAL: No trajectory manager!")
    elif not hasattr(unwrapped.trajectory_manager, 'num_trajectories'):
        print("❌ Using simple trajectory (circle/line), NOT multi_recorded!")
        print("   -> Training did NOT see the 1,038 recorded trajectories")
    elif unwrapped.trajectory_manager.num_trajectories == 1:
        print("⚠️  Only 1 trajectory loaded - should be 1,038!")
    elif unwrapped.trajectory_manager.num_trajectories < 100:
        print(f"⚠️  Only {unwrapped.trajectory_manager.num_trajectories} trajectories loaded")
    else:
        print(f"✅ {unwrapped.trajectory_manager.num_trajectories} trajectories loaded")
        print("   -> Multi-trajectory system appears to be working")
    
    print("="*80)
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
