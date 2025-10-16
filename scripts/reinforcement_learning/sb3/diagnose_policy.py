"""
Diagnostic script to check what the trained policy is actually doing.
Prints action statistics to understand why the robot isn't moving.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

# Load trained model
model_path = r"H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251016_184941\final_model.zip"
model = PPO.load(model_path, device="cpu")

print("=" * 70)
print("POLICY DIAGNOSTICS")
print("=" * 70)

# Create dummy observations (normalized)
num_samples = 100
obs_dim = 70
dummy_obs = np.random.randn(num_samples, obs_dim).astype(np.float32) * 0.1  # Small random obs

# Get actions from policy
actions_list = []
for i in range(num_samples):
    action, _ = model.predict(dummy_obs[i:i+1], deterministic=False)
    actions_list.append(action[0])

actions = np.array(actions_list)  # [num_samples, 8]

print(f"\nAction Statistics (from {num_samples} random observations):")
print(f"Action space: 8 dimensions")
print(f"  - Dimensions 0-5: Arm joint targets")
print(f"  - Dimension 6: Base linear velocity (v_x)")
print(f"  - Dimension 7: Base angular velocity (omega_z)")
print()

for i in range(8):
    if i < 6:
        label = f"Arm Joint {i}"
    elif i == 6:
        label = "Base V_X"
    else:
        label = "Base OMEGA_Z"
    
    print(f"{label:15s}: mean={actions[:, i].mean():7.4f}, std={actions[:, i].std():7.4f}, "
          f"min={actions[:, i].min():7.4f}, max={actions[:, i].max():7.4f}")

print()
print("Analysis:")
print("-" * 70)

# Check if base commands are too small
base_vx_mean = abs(actions[:, 6].mean())
base_wz_mean = abs(actions[:, 7].mean())
base_vx_std = actions[:, 6].std()
base_wz_std = actions[:, 7].std()

if base_vx_mean < 0.01 and base_wz_mean < 0.01:
    print("⚠️  WARNING: Base velocity commands are VERY SMALL!")
    print(f"   Base V_X mean: {base_vx_mean:.6f}")
    print(f"   Base OMEGA_Z mean: {base_wz_mean:.6f}")
    print("   → Policy may not have learned to move the base")
elif base_vx_std > 1.0 or base_wz_std > 1.0:
    print("⚠️  WARNING: Base commands have VERY HIGH variance!")
    print(f"   Base V_X std: {base_vx_std:.4f}")
    print(f"   Base OMEGA_Z std: {base_wz_std:.4f}")
    print("   → Policy is very random (high entropy)")
else:
    print("✓  Base commands look reasonable")

print()
print("Policy Network Info:")
print(f"  - Input dim: {obs_dim}")
print(f"  - Output dim: 8 (actions)")
print(f"  - Architecture: {model.policy}")

print()
print("=" * 70)
