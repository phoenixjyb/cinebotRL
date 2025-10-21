"""Quick test to verify base movement after action scaling fix.

This script:
1. Creates environment with 4 envs
2. Sends maximum base actions (1.0, 1.0) for 20 steps
3. Measures actual base displacement
4. Verifies it matches expected scaled velocity

Expected results (with fix @ 20Hz):
- Forward movement: ~1.5 m in 1 second (20 steps × 0.05s)
- Rotation: ~100° in 1 second (2.0 rad/s)

Without fix (bug):
- Forward movement: ~1.0 m in 1 second
- Rotation: ~57° in 1 second (1.0 rad/s)
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_base_movement():
    """Test that base actions are properly scaled."""
    
    print("=" * 70)
    print("BASE MOVEMENT VERIFICATION TEST")
    print("=" * 70)
    print("\nThis test verifies the critical bug fix for base action scaling.")
    print("Testing with maximum base actions (1.0, 1.0) for 1 second (50 steps)\n")
    
    # Import after path setup
    from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg
    
    # Create minimal config
    cfg = MobileMMTrackEEEnvCfg()
    cfg.num_envs = 4
    cfg.scene.num_envs = 4
    
    print(f"[1/4] Creating environment with {cfg.num_envs} envs...")
    env = MobileMMTrackEEEnv(cfg)
    
    print(f"[2/4] Resetting environment...")
    obs, _ = env.reset()
    
    # Get initial base positions
    initial_pos = env.robot.data.joint_pos[:, env._base_joint_ids].clone()
    initial_x = initial_pos[:, 0]
    initial_y = initial_pos[:, 1]
    initial_theta = initial_pos[:, 2]
    
    print(f"\nInitial positions (env 0):")
    print(f"  X: {initial_x[0].item():.4f} m")
    print(f"  Y: {initial_y[0].item():.4f} m")
    print(f"  Theta: {initial_theta[0].item():.4f} rad ({torch.rad2deg(initial_theta[0]).item():.1f}°)")
    
    print(f"\n[3/4] Sending maximum base actions for 50 steps (1.0 second)...")
    print("  Actions: [0,0,0,0,0,0, 1.0, 1.0]  (arm zeros, base max)")
    
    # Send maximum base actions for 50 steps
    num_steps = 50
    for step in range(num_steps):
        # Actions: [6 arm joints, vx, wz]
        # Set arm to zeros, base to maximum forward + rotation
        action = torch.zeros(cfg.num_envs, 8, device=env.device)
        action[:, 6] = 1.0  # Maximum forward velocity
        action[:, 7] = 1.0  # Maximum rotation
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            current_pos = env.robot.data.joint_pos[:, env._base_joint_ids]
            dx = (current_pos[0, 0] - initial_x[0]).item()
            dy = (current_pos[0, 1] - initial_y[0]).item()
            dtheta = (current_pos[0, 2] - initial_theta[0]).item()
            print(f"  Step {step:2d}: Δx={dx:+.3f}m, Δy={dy:+.3f}m, Δθ={torch.rad2deg(torch.tensor(dtheta)).item():+.1f}°")
    
    # Get final base positions
    final_pos = env.robot.data.joint_pos[:, env._base_joint_ids]
    final_x = final_pos[:, 0]
    final_y = final_pos[:, 1]
    final_theta = final_pos[:, 2]
    
    # Compute displacements
    dx = (final_x - initial_x)[0].item()
    dy = (final_y - initial_y)[0].item()
    dtheta = (final_theta - initial_theta)[0].item()
    dtheta_deg = torch.rad2deg(torch.tensor(dtheta)).item()
    
    print(f"\n[4/4] Results Analysis (env 0):")
    print(f"  Final X: {final_x[0].item():.4f} m")
    print(f"  Final Y: {final_y[0].item():.4f} m")
    print(f"  Final Theta: {final_theta[0].item():.4f} rad ({torch.rad2deg(final_theta[0]).item():.1f}°)")
    print(f"\n  Total displacement:")
    print(f"    ΔX: {dx:+.4f} m")
    print(f"    ΔY: {dy:+.4f} m")
    print(f"    Δθ: {dtheta:+.4f} rad ({dtheta_deg:+.1f}°)")
    
    # Expected values (with fix)
    dt = 0.005 * 10  # 5ms physics × 10 decimation = 50ms per step (20Hz control)
    expected_vx = 1.5  # max_linear_velocity
    expected_wz = 2.0  # max_angular_velocity
    expected_distance = expected_vx * dt * num_steps  # Should be ~1.5m
    expected_rotation = expected_wz * dt * num_steps  # Should be ~2.0 rad (~115°)
    expected_rotation_deg = torch.rad2deg(torch.tensor(expected_rotation)).item()
    
    print(f"\n  Expected (with fix):")
    print(f"    Forward: {expected_distance:.4f} m  (1.5 m/s × 1.0s)")
    print(f"    Rotation: {expected_rotation:.4f} rad ({expected_rotation_deg:.1f}°)  (2.0 rad/s × 1.0s)")
    
    # Without fix expected
    buggy_distance = 1.0 * dt * num_steps  # Would be ~1.0m
    buggy_rotation = 1.0 * dt * num_steps  # Would be ~1.0 rad (~57°)
    buggy_rotation_deg = torch.rad2deg(torch.tensor(buggy_rotation)).item()
    
    print(f"\n  Expected (WITHOUT fix - bug):")
    print(f"    Forward: {buggy_distance:.4f} m  (1.0 m/s × 1.0s)")
    print(f"    Rotation: {buggy_rotation:.4f} rad ({buggy_rotation_deg:.1f}°)  (1.0 rad/s × 1.0s)")
    
    # Compute actual distance (Pythagorean)
    actual_distance = (dx**2 + dy**2)**0.5
    
    print(f"\n  Measured:")
    print(f"    Distance: {actual_distance:.4f} m")
    print(f"    Rotation: {abs(dtheta):.4f} rad ({abs(dtheta_deg):.1f}°)")
    
    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT:")
    print("=" * 70)
    
    distance_error = abs(actual_distance - expected_distance)
    rotation_error = abs(abs(dtheta) - expected_rotation)
    
    if distance_error < 0.1 and rotation_error < 0.2:
        print("✅ BASE MOVEMENT CORRECT - Bug is FIXED!")
        print(f"   Distance within 10cm of expected: {distance_error*100:.1f}cm error")
        print(f"   Rotation within 11° of expected: {torch.rad2deg(torch.tensor(rotation_error)).item():.1f}° error")
        print("\n   Base actions are properly scaled to velocity limits.")
        print("   Policy can now learn to use base for tracking!")
        return True
    
    elif abs(actual_distance - buggy_distance) < 0.1:
        print("❌ BASE MOVEMENT WEAK - Bug still present!")
        print(f"   Distance matches buggy behavior: ~{actual_distance:.2f}m (expected {buggy_distance:.2f}m)")
        print(f"   Should be: ~{expected_distance:.2f}m (1.5× faster)")
        print("\n   Base actions NOT scaled - still using raw [-1,1] values!")
        print("   Policy will learn to ignore base (too slow to help).")
        return False
    
    else:
        print("⚠️  UNEXPECTED BEHAVIOR")
        print(f"   Distance: {actual_distance:.2f}m (expected {expected_distance:.2f}m, bug would give {buggy_distance:.2f}m)")
        print(f"   Rotation: {abs(dtheta_deg):.1f}° (expected {expected_rotation_deg:.1f}°, bug would give {buggy_rotation_deg:.1f}°)")
        print("\n   Values don't match either scenario - investigate further!")
        return None
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        result = test_base_movement()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
