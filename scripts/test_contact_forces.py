"""
Test script to verify that Isaac Sim contact force APIs are working.

This script:
1. Creates the MobileMMTrackEE environment
2. Checks if contact force APIs are available in the environment
3. Monitors contact forces during random actions
4. Commands collision-prone actions to test detection
5. Reports if the API is working or returning zeros

Run with: isaaclab.bat -p scripts/test_contact_forces.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Accept EULA
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

import torch
import gymnasium as gym
import numpy as np

print("\n" + "=" * 80)
print("ISAAC SIM CONTACT FORCE API VERIFICATION TEST")
print("=" * 80)
print()

def test_contact_forces():
    """Test if contact force APIs are working."""
    
    # Step 1: Register and create environment
    print("-" * 80)
    print("STEP 1: Environment Creation")
    print("-" * 80)
    
    # Register task
    print("Registering MobileMMTrackEE-v0 task...")
    try:
        import omni.isaac.lab_tasks  # This triggers task registration
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        print("✓ Task registration imported\n")
    except ImportError as e:
        print(f"✗ Failed to import lab_tasks: {e}")
        # Try direct registration
        try:
            from rl_platform.tasks.mobile_mm import __init__
            print("✓ Task registered directly\n")
        except ImportError as e2:
            print(f"✗ Failed direct registration: {e2}")
            return False
    
    # Create environment with minimal settings
    print("Creating environment (1 env for testing)...")
    try:
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
        
        # Create config
        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.scene.num_envs = 1  # Just 1 env for testing
        env_cfg.sim.device = "cuda:0"
        
        # Create environment directly
        env = MobileMMTrackEEEnv(cfg=env_cfg)
        print("✓ Environment created successfully\n")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Check API availability
    print("-" * 80)
    print("STEP 2: Contact Force API Availability Check")
    print("-" * 80)
    
    # Access the robot from environment
    robot = env.robot
    
    api_available = False
    working_method = None
    
    # Method 1: PhysX view
    try:
        forces = robot.root_physx_view.get_net_contact_forces()
        print("✓ Method 1 (root_physx_view.get_net_contact_forces()): AVAILABLE")
        print(f"  Shape: {forces.shape}, Device: {forces.device}")
        api_available = True
        working_method = "root_physx_view"
    except AttributeError as e:
        print(f"✗ Method 1 (root_physx_view): NOT AVAILABLE")
        print(f"  Error: {e}")
    
    # Method 2: Robot data
    try:
        forces = robot.data.body_net_contact_force_w
        print("✓ Method 2 (data.body_net_contact_force_w): AVAILABLE")
        print(f"  Shape: {forces.shape}, Device: {forces.device}")
        if not api_available:
            api_available = True
            working_method = "data.body_net_contact_force_w"
    except AttributeError as e:
        print(f"✗ Method 2 (data.body_net_contact_force_w): NOT AVAILABLE")
        print(f"  Error: {e}")
    
    if not api_available:
        print("\n" + "=" * 80)
        print("❌ CRITICAL: NO CONTACT FORCE API AVAILABLE!")
        print("=" * 80)
        print("Self-collision detection will NOT work!")
        env.close()
        return False
    
    print(f"\n✓ Using working method: {working_method}\n")
    
    # Step 3: Test baseline forces (should be near zero)
    print("-" * 80)
    print("STEP 3: Baseline Contact Forces (Should Be Near Zero)")
    print("-" * 80)
    
    # Reset environment
    obs_dict, extras = env.reset()
    print("Environment reset complete")
    
    # Take a few random steps to settle physics
    print("Running 10 steps to settle physics...")
    for i in range(10):
        action = torch.zeros((1, env.num_actions), device=env.device)  # Zero actions
        obs_dict, rewards, dones, extras = env.step(action)
    
    # Get contact forces
    if working_method == "root_physx_view":
        forces = robot.root_physx_view.get_net_contact_forces()
    else:
        forces = robot.data.body_net_contact_force_w
    
    force_magnitudes = torch.norm(forces, dim=-1)
    max_force = force_magnitudes.max().item()
    mean_force = force_magnitudes.mean().item()
    
    print(f"\nBaseline forces (at rest):")
    print(f"  Max contact force:  {max_force:.4f} N")
    print(f"  Mean contact force: {mean_force:.4f} N")
    
    if max_force < 1.0:
        print("✓ Baseline forces are low (< 1.0 N) - as expected")
    else:
        print(f"⚠  Baseline forces are high (>= 1.0 N) - unexpected!")
    
    # Step 4: Test with random actions
    print("\n" + "-" * 80)
    print("STEP 4: Contact Forces During Random Actions")
    print("-" * 80)
    print("Running 50 steps with random actions...")
    
    max_forces_random = []
    for step in range(50):
        # Random actions
        action = torch.rand((1, env.num_actions), device=env.device) * 2 - 1  # [-1, 1]
        obs_dict, rewards, dones, extras = env.step(action)
        
        # Get contact forces
        if working_method == "root_physx_view":
            forces = robot.root_physx_view.get_net_contact_forces()
        else:
            forces = robot.data.body_net_contact_force_w
        
        force_magnitudes = torch.norm(forces, dim=-1)
        max_force = force_magnitudes.max().item()
        max_forces_random.append(max_force)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: Max force = {max_force:8.4f} N")
    
    peak_random = max(max_forces_random)
    mean_random = np.mean(max_forces_random)
    
    print(f"\nRandom action results:")
    print(f"  Peak force:  {peak_random:.4f} N")
    print(f"  Mean force:  {mean_random:.4f} N")
    
    # Step 5: Test with collision-prone actions
    print("\n" + "-" * 80)
    print("STEP 5: Forced Collision Test (Extreme Actions)")
    print("-" * 80)
    print("Commanding extreme actions to force collisions...")
    
    # Reset to clean state
    obs_dict, extras = env.reset()
    
    max_forces_collision = []
    for step in range(50):
        # Extreme actions targeting known collision configurations
        # Actions: [6 arm joints, 3 base DOF] = 9 total
        action = torch.tensor([[
            1.0,   # Arm joint 1: max
            -1.0,  # Arm joint 2: min (shoulder down)
            1.0,   # Arm joint 3: max (elbow up)
            1.0,   # Arm joint 4: max
            -1.0,  # Arm joint 5: min
            1.0,   # Arm joint 6: max
            0.0,   # Base x: no movement
            0.0,   # Base y: no movement
            0.0,   # Base theta: no rotation
        ]], device=env.device)
        
        obs_dict, rewards, dones, extras = env.step(action)
        
        # Get contact forces
        if working_method == "root_physx_view":
            forces = robot.root_physx_view.get_net_contact_forces()
        else:
            forces = robot.data.body_net_contact_force_w
        
        force_magnitudes = torch.norm(forces, dim=-1)
        max_force = force_magnitudes.max().item()
        max_forces_collision.append(max_force)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: Max force = {max_force:8.4f} N")
    
    peak_collision = max(max_forces_collision)
    mean_collision = np.mean(max_forces_collision)
    
    print(f"\nCollision test results:")
    print(f"  Peak force:  {peak_collision:.4f} N")
    print(f"  Mean force:  {mean_collision:.4f} N")
    
    # Step 6: Check reward components
    print("\n" + "-" * 80)
    print("STEP 6: Reward Component Analysis")
    print("-" * 80)
    
    if "reward_components" in extras:
        reward_comps = extras["reward_components"]
        print("Reward components from last step:")
        for key, value in reward_comps.items():
            if "collision" in key.lower():
                print(f"  {key}: {value:.4f}")
    else:
        print("⚠  No reward_components in extras dict")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    env.close()
    
    # Analysis
    if peak_collision > 5.0:
        print("✅ PASS: Contact forces API is WORKING!")
        print(f"   Peak force {peak_collision:.2f} N detected during collision test")
        print("   Self-collision detection is functional ✓")
        return True
    elif peak_collision > 1.0 or peak_random > 1.0:
        print("⚠️  PARTIAL: Contact forces detected but may be weak")
        print(f"   Peak force {max(peak_collision, peak_random):.2f} N")
        print("   API is working but sensitivity may be low")
        print("   Self-collision detection should work but may need tuning")
        return True
    else:
        print("❌ FAIL: Contact forces NOT detected!")
        print(f"   Peak force {max(peak_collision, peak_random):.2f} N (< 1.0 N)")
        print("   Contact force API is NOT working properly")
        print("   Self-collision detection is DISABLED in practice!")
        return False


def main():
    """Run the test."""
    try:
        print("Initializing test...")
        result = test_contact_forces()
        
        print("\n" + "=" * 80)
        if result:
            print("✅ Contact force API verification: SUCCESS")
            print("   Your self-collision detection should work during training!")
            print("\nRECOMMENDATION:")
            print("  - Add collision diagnostics to env.py as suggested in documentation")
            print("  - Monitor contact forces in early training to verify")
        else:
            print("❌ Contact force API verification: FAILED")
            print("   Self-collision detection will NOT work!")
            print("\nRECOMMENDATION:")
            print("  - Review CONTACT_FORCE_API_VERIFICATION.md for alternatives")
            print("  - Consider joint-limit based heuristics")
            print("  - Check robot USD file for contact sensor configuration")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
