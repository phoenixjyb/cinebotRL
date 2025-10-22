#!/usr/bin/env python3
"""
Verify USD Joint Limits at Runtime

This script loads the robot USD asset in Isaac Sim/Lab and prints the actual
joint limits that PhysX will use during simulation. This is critical because:

1. URDF has joint_theta with limits: ±6.283185 rad (±2π)
2. USD converter might collapse infinite limits to [0, 0] (locked joint)
3. Need to verify USD actually has ±6.28 rad loaded correctly

Usage:
    python scripts/verify_usd_limits.py

Expected Output (CORRECT):
    joint_theta: soft_joint_pos_limits = [-6.283185, 6.283185]

Bad Output (BUG):
    joint_theta: soft_joint_pos_limits = [0.0, 0.0]  # LOCKED JOINT!

Reference: docs/_CODE_REVIEW_VALIDATION.md (Issue #3 - CRITICAL)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def main():
    """Verify USD joint limits are loaded correctly."""
    
    # Initialize Isaac Sim/Lab (minimal setup)
    from omni.isaac.kit import SimulationApp
    
    print("Initializing Isaac Sim (headless)...")
    simulation_app = SimulationApp({"headless": True})
    
    import omni
    from pxr import Usd, UsdPhysics
    
    # Path to USD asset
    usd_path = project_root / "assets_own" / "usd" / "mobile_manipulator_PPR_base_corrected.usd"
    
    if not usd_path.exists():
        print(f"❌ ERROR: USD file not found at {usd_path}")
        simulation_app.close()
        sys.exit(1)
    
    print(f"\nLoading USD: {usd_path}")
    
    # Open USD stage
    stage = Usd.Stage.Open(str(usd_path))
    
    if not stage:
        print("❌ ERROR: Failed to open USD stage")
        simulation_app.close()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("JOINT LIMITS VERIFICATION")
    print("="*70)
    
    # Find all joints with limits
    joints_found = []
    
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            joint_name = prim.GetName()
            joint_path = prim.GetPath()
            
            # Get limit attributes
            limit_api = UsdPhysics.DriveAPI(prim, "angular") if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.DriveAPI(prim, "linear")
            
            # Try to get joint limits
            has_limits = False
            lower_limit = None
            upper_limit = None
            
            # Check for limit attributes (different APIs might use different names)
            if prim.HasAttribute("physics:lowerLimit"):
                lower_limit = prim.GetAttribute("physics:lowerLimit").Get()
                has_limits = True
            if prim.HasAttribute("physics:upperLimit"):
                upper_limit = prim.GetAttribute("physics:upperLimit").Get()
                has_limits = True
            
            # Alternative: soft limits
            if prim.HasAttribute("drive:angular:physics:lowerLimit"):
                lower_limit = prim.GetAttribute("drive:angular:physics:lowerLimit").Get()
                has_limits = True
            if prim.HasAttribute("drive:angular:physics:upperLimit"):
                upper_limit = prim.GetAttribute("drive:angular:physics:upperLimit").Get()
                has_limits = True
            
            if has_limits:
                joints_found.append({
                    'name': joint_name,
                    'path': str(joint_path),
                    'lower': lower_limit,
                    'upper': upper_limit,
                    'type': 'Revolute' if prim.IsA(UsdPhysics.RevoluteJoint) else 'Prismatic'
                })
    
    # Print results
    if not joints_found:
        print("\n⚠️  WARNING: No joints with limits found in USD!")
        print("This might indicate:")
        print("  1. USD format doesn't store limits as expected")
        print("  2. Limits are stored under different attributes")
        print("  3. Need to check Isaac Lab's articulation API instead")
    else:
        print(f"\nFound {len(joints_found)} joints with limits:\n")
        
        # Check for critical joints
        critical_joints = ['joint_theta', 'joint_x', 'joint_y']
        
        for joint in joints_found:
            status = ""
            if joint['name'] in critical_joints:
                if joint['name'] == 'joint_theta':
                    # Check if joint_theta is locked
                    if joint['lower'] == 0.0 and joint['upper'] == 0.0:
                        status = "  ❌ LOCKED (BUG!)"
                    elif abs(joint['lower'] + 6.283185) < 0.01 and abs(joint['upper'] - 6.283185) < 0.01:
                        status = "  ✅ CORRECT (±2π rad)"
                    else:
                        status = f"  ⚠️  UNEXPECTED"
                elif joint['name'] in ['joint_x', 'joint_y']:
                    # Check prismatic joints (should be ~±5m)
                    if abs(joint['lower']) > 1.0 and abs(joint['upper']) > 1.0:
                        status = "  ✅ CORRECT"
                    else:
                        status = "  ⚠️  TOO RESTRICTIVE"
            
            print(f"{joint['name']:20s} ({joint['type']:10s}): [{joint['lower']:8.3f}, {joint['upper']:8.3f}]{status}")
    
    print("\n" + "="*70)
    print("\nCRITICAL CHECK: joint_theta limits")
    print("="*70)
    
    # Find joint_theta specifically
    joint_theta = next((j for j in joints_found if j['name'] == 'joint_theta'), None)
    
    if joint_theta:
        lower = joint_theta['lower']
        upper = joint_theta['upper']
        
        print(f"\njoint_theta limits: [{lower:.6f}, {upper:.6f}]")
        
        if lower == 0.0 and upper == 0.0:
            print("\n❌ CRITICAL BUG DETECTED!")
            print("   joint_theta is LOCKED at [0, 0]")
            print("   This prevents base rotation!")
            print("\n   FIX: Re-export USD from URDF with:")
            print("        - Ensure URDF has finite limits: ±6.283185 rad")
            print("        - Check USD converter handles limits correctly")
            print("        - Verify PhysX articulation settings")
        elif abs(lower + 6.283185) < 0.01 and abs(upper - 6.283185) < 0.01:
            print("\n✅ CORRECT! joint_theta has proper limits (±2π rad)")
            print("   Base can rotate freely within ±360° range")
        else:
            print(f"\n⚠️  UNEXPECTED LIMITS!")
            print(f"   Expected: [-6.283185, 6.283185] (±2π rad)")
            print(f"   Got:      [{lower:.6f}, {upper:.6f}]")
            print("   Check URDF → USD conversion process")
    else:
        print("\n⚠️  WARNING: joint_theta NOT FOUND in USD!")
        print("   This might mean:")
        print("     1. Joint name mismatch (check USD prim names)")
        print("     2. PPR joints configured differently")
        print("     3. Need to inspect USD with Isaac Sim GUI")
        print("\n   ACTION: Open USD in Isaac Sim and inspect joint tree")
    
    # Cleanup
    print("\n" + "="*70)
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()
