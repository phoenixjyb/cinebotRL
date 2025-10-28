#!/usr/bin/env python3
"""Quick test for trajectory visualization markers.

This minimal script tests ONLY the marker visualization without running the full RL loop.
"""

import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Accept EULA
os.environ["ACCEPT_EULA"] = "YES"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
os.environ["GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS"] = "1"

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Test trajectory markers")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
args = parser.parse_args()

# Launch Isaac Sim
print("[1/4] Launching Isaac Sim...")
from isaaclab.app import AppLauncher
launcher = AppLauncher(args_cli=parser)
simulation_app = launcher.app

print("[2/4] Importing dependencies...")
import gymnasium as gym

# Import task directly and register
print("[3/4] Registering tasks and creating environment...")
from src.rl_platform.tasks.mobile_mm.env import MobileMMTrackEE
from src.rl_platform.tasks.mobile_mm.config import MobileMMTrackEECfg

gym.register(
    id="MobileMMTrackEE-v0",
    entry_point="src.rl_platform.tasks.mobile_mm.env:MobileMMTrackEE",
    disable_env_checker=True,
)
print("    ✓ Task registered")

env = gym.make(
    'MobileMMTrackEE-v0',
    num_envs=args.num_envs,
    headless=False,  # Must use GUI to see markers
)

# CRITICAL: Enable debug visualization!
if hasattr(env.unwrapped, 'set_debug_vis'):
    env.unwrapped.set_debug_vis(True)
    print("✓ Debug visualization enabled")
else:
    print("✗ set_debug_vis() not found - markers won't update!")

print("[4/4] Running simulation...")
print("    Watch for colored spheres:")
print("    - RED (large, 0.06m) = Current target waypoint")
print("    - GREEN (medium, 0.04m) = Future waypoints (up to 50)")
print("    - BLUE (small, 0.03m) = Past waypoints (up to 20)")
print("    - YELLOW (medium, 0.05m) = End-effector position")
print("")
print("    Press Ctrl+C to stop...")

try:
    obs, info = env.reset()
    
    for step in range(500):  # Run for 500 steps
        # Random actions (not important, just keep sim running)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 100 == 0:
            print(f"Step {step}/500 - Markers should be visible in viewport")
        
        # Reset if any env terminates
        if terminated.any() or truncated.any():
            obs, info = env.reset()
    
    print("\n✓ Test complete! Did you see colored spheres?")
    
except KeyboardInterrupt:
    print("\n✓ Interrupted by user")
finally:
    env.close()
    simulation_app.close()
