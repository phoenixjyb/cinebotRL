#!/usr/bin/env python3
"""Convert URDF to USD using Isaac Sim's URDF importer.

This script must be run through Isaac Lab's launcher (isaaclab.bat) to access
the Isaac Sim Python environment and URDF importer.

Usage:
    I:\isaaclab\isaaclab.bat -p scripts/convert_urdf_to_usd.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Isaac Sim app launcher must be imported first
from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Convert URDF to USD using Isaac Sim")
parser.add_argument(
    "--urdf",
    type=str,
    default="assets_own/mobile_manipulator_PPR_theta_before_x.urdf",
    help="Path to input URDF file (relative to project root)",
)
parser.add_argument(
    "--usd",
    type=str,
    default="assets_own/usd/mobile_manipulator_PPR_theta_before_x.usd",
    help="Path to output USD file (relative to project root)",
)
parser.add_argument(
    "--mesh-scale",
    type=float,
    default=0.001,
    help="Scale factor for mesh geometry (default: 0.001 for mm→m conversion)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import Isaac Sim modules (must be after app launch)
import carb
import sys, glob
import omni.kit.app
import omni.kit.commands

# AppLauncher does not always enable the URDF importer extension for bare
# conversion scripts, so enable it explicitly before acquiring the interface.
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
simulation_app.update()

# Add URDF importer extension to path (needed when extension is copied, not junctioned)
urdf_ext_dirs = glob.glob(r"G:\isaaclab_venv\Lib\site-packages\isaacsim\kit\data\kit\isaac-sim\5.1\exts\3\isaacsim.asset.importer.urdf-*")
for d in urdf_ext_dirs:
    if d not in sys.path:
        sys.path.insert(0, d)
try:
    from isaacsim.asset.importer.urdf import _urdf
except ImportError:
    try:
        from omni.isaac.urdf import _urdf
    except ImportError:
        # Direct import from extension directory
        import importlib.util
        pyd_path = glob.glob(r"G:\isaaclab_venv\Lib\site-packages\isaacsim\kit\data\kit\isaac-sim\5.1\exts\3\isaacsim.asset.importer.urdf-*\isaacsim\asset\importer\urdf\_urdf*.pyd")[0]
        spec = importlib.util.spec_from_file_location("_urdf", pyd_path)
        _urdf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_urdf)
from pxr import Usd, UsdPhysics

def set_if_available(obj, name: str, value) -> None:
    """Set an Isaac Sim importer option only if this version exposes it."""
    if hasattr(obj, name):
        setattr(obj, name, value)
    else:
        print(f"Skipping unsupported URDF importer option: {name}")


def main():
    """Convert URDF to USD with proper configuration."""
    
    # Get absolute paths
    project_root = Path(__file__).parent.parent
    urdf_path = (project_root / args_cli.urdf).resolve()
    usd_path = (project_root / args_cli.usd).resolve()
    
    print("=" * 80)
    print("URDF to USD Conversion")
    print("=" * 80)
    print(f"Input URDF:  {urdf_path}")
    print(f"Output USD:  {usd_path}")
    print(f"Mesh scale:  {args_cli.mesh_scale}")
    print()
    
    # Validate input file exists
    if not urdf_path.is_file():
        print(f"ERROR: URDF file not found: {urdf_path}")
        return 1
    
    # Create output directory if needed
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Import URDF
    print("Importing URDF...")
    try:
        # Configure import settings
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            print("Failed to create URDF import config")
            return 1
        set_if_available(import_config, "merge_fixed_joints", False)  # Keep all joints
        set_if_available(import_config, "convex_decomp", False)  # Use original collision meshes
        set_if_available(import_config, "import_inertia_tensor", True)  # Import inertia from URDF
        set_if_available(import_config, "fix_base", False)  # Mobile base, not fixed
        set_if_available(import_config, "self_collision", False)  # Disable for now
        if hasattr(_urdf, "UrdfJointTargetType"):
            set_if_available(import_config, "default_drive_type", _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
        set_if_available(import_config, "default_position_drive_damping", 1000.0)
        set_if_available(import_config, "default_position_drive_stiffness", 10000.0)
        set_if_available(import_config, "distance_scale", args_cli.mesh_scale)  # Convert mm->m
        
        # Import URDF to USD
        success, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
            dest_path=str(usd_path),
            get_articulation_root=True,
        )
        
        if not success:
            print("URDF import failed!")
            return 1

        print(f"URDF imported successfully to: {prim_path}")
        
        # Open the stage to verify
        print("Opening USD stage for verification...")
        stage = Usd.Stage.Open(str(usd_path))
        
        if not stage:
            print(f"Failed to open USD stage: {usd_path}")
            return 1

        if not stage.GetDefaultPrim():
            root_prims = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsActive()]
            if not root_prims:
                print(f"Failed to find a root prim in USD stage: {usd_path}")
                return 1
            stage.SetDefaultPrim(root_prims[0])
            stage.GetRootLayer().Save()
            print(f"Set USD defaultPrim to: {root_prims[0].GetPath()}")
        
        # Print summary
        print()
        print("=" * 80)
        print("USD Stage Summary")
        print("=" * 80)
        
        # Count prims
        all_prims = [p for p in stage.Traverse()]
        print(f"Total prims: {len(all_prims)}")
        
        # Count articulation links
        articulation_prims = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RigidBodyAPI)]
        print(f"Rigid bodies (links): {len(articulation_prims)}")
        
        # Count joints
        joint_prims = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        print(f"Joints: {len(joint_prims)}")
        
        print()
        print("USD conversion complete!")
        print(f"Output saved to: {usd_path}")
        print()
        print("Next steps:")
        print("  1. Verify USD with: I:\\isaaclab\\isaaclab.bat -p scripts/test_mobile_mm_env.py")
        print("  2. Start training with: .\\scripts\\launch_training_windows.ps1")
        print()
        
        return 0
        
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        simulation_app.close()


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
