"""
URDF to USD Conversion Script - Run in Isaac Sim Script Editor

Instructions:
1. Open Isaac Sim GUI
2. Go to Window → Script Editor
3. Copy and paste this entire script
4. Click "Run" button
5. The USD will be created in assets_own/usd/mobile_manipulator_PPR_theta_x_y.usd

This script uses the URDF importer available in Isaac Sim to convert the URDF.
"""

import carb
from pathlib import Path

# Define paths
project_root = Path("C:/Users/yanbo/wSpace/cinebotRL")
urdf_path = project_root / "assets_own" / "mobile_manipulator_PPR_theta_x_y.urdf"
usd_path = project_root / "assets_own" / "usd" / "mobile_manipulator_PPR_theta_x_y.usd"

print("=" * 80)
print("URDF to USD Conversion (Isaac Sim GUI)")
print("=" * 80)
print(f"Input URDF:  {urdf_path}")
print(f"Output USD:  {usd_path}")
print()

# Ensure output directory exists
usd_path.parent.mkdir(parents=True, exist_ok=True)

# Import the URDF using Isaac Sim's UI command
# This is the menu command: File → Import → URDF
import omni.kit.commands

print("🔄 Importing URDF...")
try:
    # Execute the URDF import command
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config={
            "merge_fixed_joints": False,
            "convex_decomposition": False,
            "import_inertia_tensor": True,
            "fix_base": False,  # Mobile base
            "make_default_prim": True,
            "create_physics_scene": True,
            "distance_scale": 1.0,
            "density": 0.0,  # Use URDF masses
        },
    )
    
    print("✅ URDF imported to stage")
    
    # Save the stage to USD file
    print(f"💾 Saving to: {usd_path}")
    import omni.usd
    context = omni.usd.get_context()
    context.save_as_stage(str(usd_path))
    
    print("✅ USD saved successfully!")
    print()
    print("Next steps:")
    print("  1. Verify USD with: I:\\isaaclab\\isaaclab.bat -p scripts/inspect_usd.py assets_own/usd/mobile_manipulator_PPR_theta_x_y.usd")
    print("  2. Update mobile_mm.py to use new USD")
    print("  3. Test with 4 envs first")
    
except Exception as e:
    print(f"❌ Error during import: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
