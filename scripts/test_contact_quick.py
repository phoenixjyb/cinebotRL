"""Quick test to check contact forces in Isaac Lab environment.

Usage:
    I:\isaaclab\isaaclab.bat -p scripts/test_contact_quick.py
"""

import argparse
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    
    print(f"\\n{'='*70}")
    print("CONTACT FORCES QUICK TEST")
    print(f"{'='*70}\\n")
    
    # STEP 1: Initialize Isaac Sim FIRST (before importing Isaac Lab!)
    print("Initializing Isaac Sim...")
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    print("✓ Isaac Sim ready\\n")
    
    # STEP 2: NOW we can import Isaac Lab modules
    print("Importing Isaac Lab modules...")
    import torch
    import gymnasium as gym
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from task_spec import register_isaac_lab_tasks
    print("✓ Modules imported\\n")
    
    # Register and create environment
    print(f"Creating environment ({args.num_envs} envs)...")
    register_isaac_lab_tasks()
    env_cfg = parse_env_cfg("MobileMMTrackEE-v0", device="cuda:0", num_envs=args.num_envs)
    env = gym.make("MobileMMTrackEE-v0", cfg=env_cfg)
    print("✓ Environment created\\n")
    
    # Reset
    print("Resetting...")
    obs, _ = env.reset()
    print("✓ Reset complete\\n")
    
    # Access internal environment
    base_env = env.unwrapped
    robot = base_env._robot
    
    # Print available contact attributes
    print(f"{'='*70}")
    print("ROBOT CONTACT FORCE ATTRIBUTES")
    print(f"{'='*70}\\n")
    
    # Check robot.data attributes
    print("robot.data attributes with 'contact' or 'force':")
    for attr in dir(robot.data):
        if 'contact' in attr.lower() or 'force' in attr.lower():
            if not attr.startswith('_'):
                try:
                    val = getattr(robot.data, attr)
                    if isinstance(val, torch.Tensor):
                        print(f"  {attr:40s} shape={val.shape}")
                    else:
                        print(f"  {attr:40s} type={type(val).__name__}")
                except:
                    pass
    
    # Check robot.root_physx_view if available
    if hasattr(robot, 'root_physx_view'):
        print("\\nrobot.root_physx_view attributes with 'contact' or 'force':")
        for attr in dir(robot.root_physx_view):
            if 'contact' in attr.lower() or 'force' in attr.lower():
                if not attr.startswith('_') and not attr.startswith('get'):
                    print(f"  {attr}")
    
    # Try to get contact forces
    print(f"\\n{'='*70}")
    print("TESTING CONTACT FORCE RETRIEVAL")
    print(f"{'='*70}\\n")
    
    # Method 1: From robot.data
    if hasattr(robot.data, 'body_net_contact_force_w'):
        forces = robot.data.body_net_contact_force_w
        print(f"Method 1: robot.data.body_net_contact_force_w")
        print(f"  Shape: {forces.shape}")
        print(f"  Max force: {forces.abs().max().item():.3f} N")
        print(f"  Non-zero count: {(forces.abs() > 0.01).sum().item()}")
    
    # Method 2: From PhysX view
    if hasattr(robot, 'root_physx_view'):
        try:
            forces = robot.root_physx_view.get_net_contact_forces()
            print(f"\\nMethod 2: robot.root_physx_view.get_net_contact_forces()")
            print(f"  Shape: {forces.shape}")
            print(f"  Max force: {forces.abs().max().item():.3f} N")
            print(f"  Non-zero count: {(forces.abs() > 0.01).sum().item()}")
        except Exception as e:
            print(f"\\nMethod 2: Failed - {e}")
    
    # Run a few steps to generate forces
    print(f"\\n{'='*70}")
    print(f"RUNNING {args.steps} STEPS WITH RANDOM ACTIONS")
    print(f"{'='*70}\\n")
    
    max_force_seen = 0.0
    for step in range(args.steps):
        # Random actions
        action = torch.randn(args.num_envs, 8, device="cuda:0") * 0.5
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check forces
        if hasattr(robot.data, 'body_net_contact_force_w'):
            forces = robot.data.body_net_contact_force_w
            max_f = forces.abs().max().item()
            if max_f > max_force_seen:
                max_force_seen = max_f
            if max_f > 0.1:
                print(f"Step {step:3d}: Max force = {max_f:8.3f} N")
    
    print(f"\\nMax force over {args.steps} steps: {max_force_seen:.3f} N")
    
    if max_force_seen < 0.01:
        print("\\n❌ PROBLEM: Contact forces are reading ~0.0 N")
        print("   This confirms the bug - no collision feedback!")
    else:
        print("\\n✅ Contact forces ARE being detected")
        print(f"   Max: {max_force_seen:.3f} N")
    
    print(f"\\n{'='*70}")
    
    # Cleanup
    env.close()
    simulation_app.close()
    print("Done!")

if __name__ == "__main__":
    main()
