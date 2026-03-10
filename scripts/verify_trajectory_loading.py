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
    
    # Now kwargs work! The environment will honor them
    env = gym.make(
        "MobileMMTrackEE-v0",
        num_envs=4,
        trajectory_type="multi_recorded",
        trajectory_dir="trajectoryToLearn/world_json",
        use_all_trajectories=True,
    )
    
    # Unwrap to get the underlying environment
    unwrapped_env = env.unwrapped
    
    print("\n" + "="*80)
    print("ENVIRONMENT CONFIGURATION")
    print("="*80)
    
    # Check trajectory manager
    if hasattr(unwrapped_env, 'trajectory_manager'):
        traj_mgr = unwrapped_env.trajectory_manager
        print(f"✅ Trajectory manager found: {type(traj_mgr).__name__}")
        print(f"   Trajectory type: {getattr(traj_mgr, 'traj_type', 'UNKNOWN')}")
        
        if traj_mgr.traj_type == "multi_recorded" and getattr(traj_mgr, 'multi_loader', None) is not None:
            loader = traj_mgr.multi_loader
            print(f"   Trajectory directory: {loader.trajectory_dir}")
            print(f"   Loaded trajectories: {len(loader.trajectories)}")
            categories = loader.get_categories()
            print(f"   Categories sample: {categories[:5]}{'...' if len(categories) > 5 else ''}")
        elif traj_mgr.traj_type == "recorded" and traj_mgr.recorded_positions is not None:
            print(f"   Recorded waypoints: {traj_mgr.recorded_positions.shape[1]}")
    else:
        print("❌ No trajectory manager found!")
    
    print("\n" + "="*80)
    print("SAMPLING TRAJECTORIES")
    print("="*80)
    
    # Reset and sample trajectory targets
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    print("\nSampling 100 timesteps from current trajectories:")
    print("-"*80)
    
    initial_ee_pos = unwrapped_env.robot.data.body_pos_w[0, unwrapped_env._ee_body_idx, :].cpu().numpy()
    print(f"Initial EE position: {initial_ee_pos}")
    
    target_positions = []
    for step in range(100):
        target_pos, _ = unwrapped_env.trajectory_manager.get_target_pose()
        target_positions.append(target_pos[0].cpu().numpy())
        # IMPORTANT: Step through the trajectory!
        unwrapped_env.trajectory_manager.step()
    
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
        if isinstance(obs, tuple):
            obs, _ = obs
        target_pos, _ = unwrapped_env.trajectory_manager.get_target_pose()
        print(f"  Reset {reset_idx + 1}: Target = {target_pos[0].cpu().numpy()}")
        
        # If multi-trajectory, check which one was selected
        if hasattr(unwrapped_env.trajectory_manager, 'current_trajectory_indices'):
            idx = unwrapped_env.trajectory_manager.current_trajectory_indices[0].item()
            print(f"             Trajectory index = {idx}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if not hasattr(unwrapped_env, 'trajectory_manager'):
        print("❌ CRITICAL: No trajectory manager!")
    elif unwrapped_env.trajectory_manager.traj_type not in ["recorded", "multi_recorded"]:
        print(f"❌ Using simple trajectory ({unwrapped_env.trajectory_manager.traj_type}), NOT multi_recorded!")
        print("   -> Training did NOT see the 1,038 recorded trajectories")
    elif unwrapped_env.trajectory_manager.multi_loader is None:
        print("❌ Multi-loader not initialized!")
        print("   -> Trajectory type is multi_recorded but loader failed")
    else:
        num_trajs = len(unwrapped_env.trajectory_manager.multi_loader.trajectories)
        if num_trajs == 1:
            print("⚠️  Only 1 trajectory loaded - should be 1,038!")
        elif num_trajs < 100:
            print(f"⚠️  Only {num_trajs} trajectories loaded")
        else:
            print(f"✅ {num_trajs} trajectories loaded and advancing!")
            print(f"   -> Multi-trajectory system is WORKING")
            print(f"   -> Training with {num_trajs} diverse trajectories")
    
    print("="*80)
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
