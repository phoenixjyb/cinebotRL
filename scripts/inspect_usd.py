"""Quick USD inspection script to find end-effector link and joint info.

This script uses pxr (USD Python bindings) to inspect the robot USD file
and print out the link hierarchy and joint information.

Usage:
    python scripts/inspect_usd.py assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
"""

import argparse
import sys
from pathlib import Path


def inspect_usd(usd_path: Path):
    """Inspect USD file and print robot structure."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        print("ERROR: pxr (USD Python bindings) not available")
        print("This needs to run in an environment with USD installed")
        print("(e.g., Isaac Sim Python or Isaac Lab environment)")
        sys.exit(1)
    
    if not usd_path.exists():
        print(f"ERROR: USD file not found: {usd_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Inspecting USD: {usd_path}")
    print(f"{'='*70}\n")
    
    # Open stage
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print("ERROR: Could not open USD stage")
        sys.exit(1)
    
    # Find all prims
    print("🔍 PRIM HIERARCHY")
    print("-" * 70)
    
    def print_prim_tree(prim, indent=0):
        """Recursively print prim tree."""
        prefix = "  " * indent
        prim_type = prim.GetTypeName()
        
        # Highlight interesting types
        if prim_type in ["Xform", "Mesh", "Skeleton"]:
            icon = "📦"
        elif "Joint" in prim_type:
            icon = "🔗"
        elif "Collision" in str(prim.GetPath()) or "collision" in str(prim.GetPath()).lower():
            icon = "💥"
        else:
            icon = "  "
        
        print(f"{prefix}{icon} {prim.GetName()} ({prim_type})")
        
        # Print children
        for child in prim.GetChildren():
            print_prim_tree(child, indent + 1)
    
    root = stage.GetPseudoRoot()
    for prim in root.GetChildren():
        print_prim_tree(prim)
    
    # Find joints
    print(f"\n{'='*70}")
    print("🔗 JOINTS FOUND")
    print("-" * 70)
    
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_name = str(prim.GetPath())
            # Try to get body0 and body1
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            
            joints.append({
                "path": joint_name,
                "body0": str(body0[0]) if body0 else "None",
                "body1": str(body1[0]) if body1 else "None",
            })
            
            print(f"Joint: {joint_name}")
            print(f"  ├─ Body 0: {body0[0] if body0 else 'None'}")
            print(f"  └─ Body 1: {body1[0] if body1 else 'None'}")
    
    # Find likely end-effector
    print(f"\n{'='*70}")
    print("🎯 LIKELY END-EFFECTOR LINKS")
    print("-" * 70)
    
    ee_keywords = ["ee", "end_effector", "gripper", "tool", "tcp", "flange", "wrist", "hand"]
    ee_candidates = []
    
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath()).lower()
        prim_name = prim.GetName().lower()
        
        # Check if any keyword matches
        for keyword in ee_keywords:
            if keyword in prim_path or keyword in prim_name:
                ee_candidates.append({
                    "path": str(prim.GetPath()),
                    "name": prim.GetName(),
                    "type": prim.GetTypeName(),
                })
                break
    
    if ee_candidates:
        for candidate in ee_candidates:
            print(f"✓ {candidate['path']}")
            print(f"  Type: {candidate['type']}")
    else:
        print("❌ No obvious end-effector links found")
        print("\nLooking for last joint in arm chain...")
        
        # Try to find arm joints
        arm_joints = [j for j in joints if "arm" in j["path"].lower()]
        if arm_joints:
            last_joint = arm_joints[-1]
            print(f"\n✓ Last arm joint: {last_joint['path']}")
            print(f"  Body 1 (child link): {last_joint['body1']}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print("-" * 70)
    print(f"Total prims: {len(list(stage.Traverse()))}")
    print(f"Total joints: {len(joints)}")
    print(f"EE candidates: {len(ee_candidates)}")
    
    if joints:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   For env.py, update the end-effector body index or name.")
        print(f"   Currently using: robot.data.body_pos_w[:, -1, :]  # Last body")
        print(f"   Should use specific link name from above")
    
    print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inspect USD robot file")
    parser.add_argument(
        "usd_path",
        type=Path,
        help="Path to USD file to inspect",
    )
    
    args = parser.parse_args()
    inspect_usd(args.usd_path)


if __name__ == "__main__":
    main()
