#!/usr/bin/env python3
"""Update project to use theta_before_x URDF with correct mobile base joint order.

This script:
1. Converts the new URDF to USD using Isaac Sim
2. Updates Python configuration to use new USD
3. Verifies all assets are in place

Usage:
    # Step 1: Convert URDF to USD (requires Isaac Lab environment)
    I:\isaaclab\isaaclab.bat -p scripts/update_to_theta_before_x.py --convert

    # Step 2: Update Python config (can run in any Python environment)
    python scripts/update_to_theta_before_x.py --update-config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert_urdf_to_usd():
    """Convert theta_before_x URDF to USD using Isaac Sim."""
    print("\n" + "=" * 80)
    print("Step 1: Converting URDF to USD")
    print("=" * 80)
    
    # Import Isaac Sim modules (must be in Isaac Lab environment)
    try:
        from isaaclab.app import AppLauncher
        import carb
        from omni.isaac.urdf import _urdf
        from pxr import Usd, UsdPhysics
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  This step must be run through Isaac Lab:")
        print("    I:\\isaaclab\\isaaclab.bat -p scripts/update_to_theta_before_x.py --convert")
        sys.exit(1)
    
    # Launch Isaac Sim
    app_launcher = AppLauncher(argparse.Namespace(headless=True))
    simulation_app = app_launcher.app
    
    # Get paths
    project_root = Path(__file__).parent.parent
    urdf_path = (project_root / "assets_own" / "mobile_manipulator_PPR_theta_before_x.urdf").resolve()
    usd_path = (project_root / "assets_own" / "usd" / "mobile_manipulator_PPR_theta_before_x.usd").resolve()
    
    if not urdf_path.exists():
        print(f"❌ URDF not found: {urdf_path}")
        sys.exit(1)
    
    # Create output directory
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📄 Input URDF: {urdf_path}")
    print(f"📦 Output USD: {usd_path}")
    
    # Configure URDF importer
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False  # Mobile base should NOT be fixed
    import_config.make_default_prim = True
    import_config.create_physics_scene = True
    import_config.distance_scale = 1.0  # URDF already in meters
    import_config.density = 0.0  # Use masses from URDF
    
    print("\n🔧 Import configuration:")
    print(f"  - merge_fixed_joints: {import_config.merge_fixed_joints}")
    print(f"  - fix_base: {import_config.fix_base}")
    print(f"  - import_inertia: {import_config.import_inertia_tensor}")
    print(f"  - distance_scale: {import_config.distance_scale}")
    
    # Perform conversion
    print("\n⚙️  Converting URDF to USD...")
    try:
        success, stage_path = _urdf.acquire_urdf_interface().parse_urdf(
            str(urdf_path),
            str(usd_path),
            import_config
        )
        
        if not success:
            print("❌ URDF import failed!")
            sys.exit(1)
        
        print(f"✅ USD created: {usd_path}")
        print(f"   Stage path: {stage_path}")
        
        # Verify USD can be opened
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            print("❌ Failed to open generated USD!")
            sys.exit(1)
        
        # Check for configuration directory (created by importer)
        config_dir = usd_path.parent / "configuration"
        if config_dir.exists():
            print(f"✅ Configuration directory: {config_dir}")
            print(f"   Files: {list(config_dir.glob('*.usd'))}")
        
        print("\n✅ Conversion complete!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)
    finally:
        simulation_app.close()


def update_python_config():
    """Update Python robot configuration to use new USD."""
    print("\n" + "=" * 80)
    print("Step 2: Updating Python Configuration")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    robot_config_file = project_root / "src" / "rl_platform" / "robots" / "mobile_mm.py"
    
    if not robot_config_file.exists():
        print(f"❌ Config file not found: {robot_config_file}")
        sys.exit(1)
    
    # Read current config
    content = robot_config_file.read_text()
    
    # Check if already updated
    if "mobile_manipulator_PPR_theta_before_x.usd" in content:
        print("✅ Configuration already uses theta_before_x USD")
        return
    
    # Update USD path
    old_line = 'usd_path=usd_dir / "mobile_manipulator_PPR_base_corrected.usd",'
    new_line = 'usd_path=usd_dir / "mobile_manipulator_PPR_theta_before_x.usd",'
    
    if old_line not in content:
        print("⚠️  Could not find expected USD path in config file")
        print("    Please update manually:")
        print(f"    File: {robot_config_file}")
        print(f"    Change: base_corrected.usd → theta_before_x.usd")
        return
    
    # Perform replacement
    updated_content = content.replace(old_line, new_line)
    robot_config_file.write_text(updated_content)
    
    print(f"✅ Updated: {robot_config_file}")
    print("   Changed: base_corrected.usd → theta_before_x.usd")


def verify_assets():
    """Verify all required assets are in place."""
    print("\n" + "=" * 80)
    print("Step 3: Verifying Assets")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "assets_own/mobile_manipulator_PPR_theta_before_x.urdf",
        "assets_own/usd/mobile_manipulator_PPR_theta_before_x.usd",
        "src/rl_platform/robots/mobile_mm.py",
    ]
    
    all_present = True
    for rel_path in required_files:
        file_path = project_root / rel_path
        if file_path.exists():
            print(f"✅ {rel_path}")
        else:
            print(f"❌ {rel_path} (MISSING)")
            all_present = False
    
    # Check mesh files
    mesh_dir = project_root / "assets_own" / "meshes" / "stl_output"
    if mesh_dir.exists():
        mesh_count = len(list(mesh_dir.glob("*.STL")))
        print(f"✅ Mesh files: {mesh_count} STL files in {mesh_dir.name}/")
    else:
        print(f"❌ Mesh directory not found: {mesh_dir}")
        all_present = False
    
    if all_present:
        print("\n✅ All assets verified!")
    else:
        print("\n⚠️  Some assets are missing. Please complete the conversion steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Update project to use theta_before_x URDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full update (run from Isaac Lab environment):
  I:\\isaaclab\\isaaclab.bat -p scripts/update_to_theta_before_x.py --convert --update-config

  # Just update Python config (any Python):
  python scripts/update_to_theta_before_x.py --update-config

  # Just verify:
  python scripts/update_to_theta_before_x.py --verify
        """
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert URDF to USD (requires Isaac Lab environment)"
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update Python robot configuration to use new USD"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all assets are in place"
    )
    
    args = parser.parse_args()
    
    # If no flags, show help
    if not (args.convert or args.update_config or args.verify):
        parser.print_help()
        print("\n⚠️  Please specify at least one action flag")
        sys.exit(1)
    
    try:
        if args.convert:
            convert_urdf_to_usd()
        
        if args.update_config:
            update_python_config()
        
        if args.verify or (args.convert and args.update_config):
            verify_assets()
        
        print("\n" + "=" * 80)
        print("✅ Update Complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Test with: I:\\isaaclab\\isaaclab.bat -p scripts/test_mobile_mm_env.py")
        print("  2. If successful, commit changes to git")
        print("  3. Rebuild MATLAB reachability map if needed")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
