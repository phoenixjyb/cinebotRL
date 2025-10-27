#!/usr/bin/env python3
"""Convert URDF to USD using Isaac Sim's asset converter.

This uses the omni.kit.asset_converter which is available in Isaac Lab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Isaac Sim app launcher must be imported first
from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Convert URDF to USD")
parser.add_argument(
    "--urdf",
    type=str,
    default="assets_own/mobile_manipulator_PPR_theta_x_y.urdf",
    help="Path to input URDF file (relative to project root)",
)
parser.add_argument(
    "--usd",
    type=str,
    default="assets_own/usd/mobile_manipulator_PPR_theta_x_y.usd",
    help="Path to output USD file (relative to project root)",
)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Sim modules
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdPhysics

def main():
    """Convert URDF to USD."""
    
    project_root = Path(__file__).parent.parent
    urdf_path = (project_root / args_cli.urdf).resolve()
    usd_path = (project_root / args_cli.usd).resolve()
    
    print("=" * 80)
    print("URDF to USD Conversion (Asset Converter)")
    print("=" * 80)
    print(f"Input URDF:  {urdf_path}")
    print(f"Output USD:  {usd_path}")
    print()
    
    if not urdf_path.is_file():
        print(f"❌ ERROR: URDF file not found: {urdf_path}")
        return 1
    
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Converting URDF to USD using asset_converter...")
    try:
        # Use asset_converter
        import omni.kit.asset_converter as converter
        
        context = converter.AssetConverterContext()
        # Configure for mobile base (not fixed)
        context.ignore_materials = False
        context.ignore_animations = False
        context.ignore_cameras = False
        context.single_mesh = False
        context.smooth_normals = True
        context.preview_surface = True
        context.support_point_instancer = False
        context.embed_mdl_in_usd = False
        context.use_meter_as_world_unit = True
        context.create_world_as_default_root_prim = False
        
        # Convert
        task = converter.create_converter_task(
            str(urdf_path),
            str(usd_path),
            None,
            context
        )
        
        success = True
        while True:
            success = task.wait_until_finished()
            if not success:
                error_msg = task.get_status()
                print(f"❌ Conversion failed: {error_msg}")
                return 1
            
            if task.is_finished():
                break
        
        print(f"✅ USD created: {usd_path}")
        
        # Open and verify
        print("🔄 Verifying USD...")
        stage = Usd.Stage.Open(str(usd_path))
        
        if not stage:
            print(f"❌ Failed to open USD: {usd_path}")
            return 1
        
        # Count elements
        all_prims = [p for p in stage.Traverse()]
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        
        print()
        print("=" * 80)
        print("USD Verification")
        print("=" * 80)
        print(f"Total prims: {len(all_prims)}")
        print(f"Total joints: {len(joints)}")
        print()
        
        # Print joint hierarchy
        print("Joint hierarchy:")
        for j in joints:
            joint_api = UsdPhysics.Joint(j)
            body0 = joint_api.GetBody0Rel().GetTargets()
            body1 = joint_api.GetBody1Rel().GetTargets()
            b0_name = body0[0].name if body0 else "None"
            b1_name = body1[0].name if body1 else "None"
            print(f"  {j.GetName()}: {b0_name} → {b1_name}")
        
        print()
        print("✅ Conversion complete!")
        print(f"📁 Output: {usd_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        simulation_app.close()


if __name__ == "__main__":
    exit(main())
