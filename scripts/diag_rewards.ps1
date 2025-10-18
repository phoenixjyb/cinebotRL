#!/usr/bin/env pwsh
"""Quick diagnostic: print reward components for 10 steps."""

$checkpoint = "H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_180217\final_model.zip"

# Create inline Python script for diagnostics
$python_code = @'
import sys
from pathlib import Path
import numpy as np
import torch

# Add paths
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "src"))

from omni.isaac.lab.app import AppLauncher
from stable_baselines3 import PPO

# Initialize app
app_launcher = AppLauncher(headless=True)
sim = app_launcher.app

# Import after app init
from src.rl_platform.tasks.mobile_mm.task_spec_manager import create_environment_from_spec
from src.task_spec import MobileMMTrackConfig

print("\n[1/3] Creating environment...")
cfg = MobileMMTrackConfig()
cfg.env.num_envs = 1
cfg.task.trajectory.type = "multi_recorded"
cfg.task.trajectory.use_all_trajectories = True

# Use the spec-based environment creation
spec = cfg.to_spec()
env = create_environment_from_spec(spec)
print("✓ Environment created")

print("[2/3] Loading model...")
model = PPO.load(r"$checkpoint", device="cuda:0")
print("✓ Model loaded")

print("[3/3] Running 10 diagnostic steps...")
print("=" * 120)

obs, _ = env.reset()

for step in range(10):
    # Get action
    action, _ = model.predict(obs, deterministic=False)
    
    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n🔹 STEP {step + 1}:")
    print(f"   Total Reward: {reward[0].item():+.2f}")
    
    # Print components
    if hasattr(env.unwrapped, 'reward_components') and env.unwrapped.reward_components:
        for key, val in sorted(env.unwrapped.reward_components.items()):
            v = val[0].item() if hasattr(val, 'item') else val[0]
            sign = "+" if v >= 0 else ""
            print(f"   {key:35s}: {sign}{v:+10.4f}")

print("\n" + "=" * 120)
print("Diagnostic complete!")

env.close()
app_launcher.close()
'@

# Write and run the script
$script_file = "c:\Users\yanbo\wSpace\cinebotRL\scripts\_diag_temp.py"
$python_code | Out-File -FilePath $script_file -Encoding UTF8

& "I:\isaaclab\isaaclab.bat" -p $script_file

# Clean up
Remove-Item $script_file -ErrorAction SilentlyContinue
