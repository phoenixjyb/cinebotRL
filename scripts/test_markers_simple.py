#!/usr/bin/env python3
"""
Simple test script to verify trajectory visualization markers work.
Tests the lazy initialization pattern.
"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test trajectory visualization markers")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("✓ Isaac Sim initialized")

# Now import after Isaac Sim is initialized
import gymnasium as gym
from src.task_spec import register_isaac_lab_tasks

# Register custom tasks
register_isaac_lab_tasks()
print("✓ Registered custom tasks")

# Create environment
print("\nCreating environment...")
env = gym.make(
    "MobileMMTrackEE-v0",
    num_envs=1,
    render_mode="rgb_array" if args_cli.headless else None,
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
    action = env.action_space.sample()
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
