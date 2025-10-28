#!/usr/bin/env python3
"""
Simple test script to verify trajectory visualization markers work.
Tests the lazy initialization pattern.
"""

import argparse
import sys
from pathlib import Path

# Add workspace root to Python path so we can import rl_platform
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root / "src"))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test trajectory visualization markers")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("✓ Isaac Sim initialized")

# Now import after Isaac Sim is initialized
import gymnasium as gym
import torch

# Register custom tasks
print("Registering custom tasks...")
from src.task_spec import register_isaac_lab_tasks
register_isaac_lab_tasks()
print("✓ Registered custom tasks")

# Create environment
print("\nCreating environment...")
env = gym.make(
    "MobileMMTrackEE-v0",
    num_envs=1,
)

# Enable debug visualization
if not args_cli.headless and hasattr(env.unwrapped, 'set_debug_vis'):
    print("\nEnabling debug visualization...")
    env.unwrapped.set_debug_vis(True)
    print("✓ Debug visualization enabled")

# Reset environment
print("\nResetting environment...")
obs, info = env.reset()
print("✓ Environment reset complete")

# Run a few steps
print("\nRunning 10 steps to test marker updates...")
for i in range(10):
    # Sample action (numpy array)
    action_np = env.action_space.sample()
    # Convert to torch tensor
    action = torch.from_numpy(action_np).unsqueeze(0).to(device='cuda:0')
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {i+1}: reward={reward[0]:.2f}")

print("\n✓ Marker test complete!")
print("If running in GUI mode, you should see colored spheres:")
print("  - Red = Current target waypoint")
print("  - Green = Future waypoints")
print("  - Blue = Past waypoints")
print("  - Yellow = End-effector")

# Cleanup
env.close()
simulation_app.close()
