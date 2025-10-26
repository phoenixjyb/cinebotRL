# Update to theta_before_x URDF

This guide explains how to update the project to use the new `mobile_manipulator_PPR_theta_before_x.urdf` which has the correct mobile base joint order.

## Why This Update?

**Old URDF** (`base_corrected.urdf`):
- Joint order: base → joint_x → joint_y → joint_theta
- Problem: Mobile base joints move in **global frame** instead of robot local frame
- Impact: Mobile base planning doesn't work correctly

**New URDF** (`theta_before_x.urdf`):
- Joint order: base → joint_theta → joint_x → joint_y  
- Fix: Mobile base joints now move in **robot local frame**
- Impact: Mobile base can rotate first, then translate in its own coordinate system

## Step 1: Convert URDF to USD (Manual in Isaac Sim)

### 1.1 Open Isaac Sim

Launch Isaac Sim (standalone, not through Isaac Lab).

### 1.2 Import URDF

1. **File** → **Import**
2. Navigate to: `C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_theta_before_x.urdf`
3. **Import Settings**:
   ```
   ☑ Import Inertia Tensor
   ☑ Create Physics Scene
   ☐ Fix Base Link (IMPORTANT: keep UNCHECKED for mobile base!)
   ☐ Merge Fixed Joints
   ☐ Self Collision
   Distance Scale: 1.0 (URDF already in meters)
   ```
4. Click **Import**

### 1.3 Verify Import

Check in the Stage panel:
- Root prim: `/mobile_manipulator_PPR_theta_before_x`
- Mobile base joints: `joint_theta`, `joint_x`, `joint_y` (in that order)
- Arm joints: `left_arm_joint_1` through `left_arm_joint_6`
- All meshes loaded correctly

### 1.4 Save USD

1. **File** → **Save As**
2. Save to: `C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd\mobile_manipulator_PPR_theta_before_x.usd`
3. **Also save the configuration directory**:
   - Isaac Sim creates a `configuration/` folder next to the USD
   - Make sure it's in: `assets_own/usd/configuration/`

## Step 2: Update Python Configuration

### 2.1 Update Robot Assets Path

Edit `src/rl_platform/robots/mobile_mm.py`:

**OLD**:
```python
def get_mobile_mm_assets() -> MobileManipulatorAssets:
    """Return paths to the mobile manipulator USD and supporting configuration."""
    usd_dir = assets_root() / "usd"
    return MobileManipulatorAssets(
        usd_path=usd_dir / "mobile_manipulator_PPR_base_corrected.usd",
        config_dir=usd_dir / "configuration",
    )
```

**NEW**:
```python
def get_mobile_mm_assets() -> MobileManipulatorAssets:
    """Return paths to the mobile manipulator USD and supporting configuration."""
    usd_dir = assets_root() / "usd"
    # Use theta_before_x version with correct mobile base joint order
    return MobileManipulatorAssets(
        usd_path=usd_dir / "mobile_manipulator_PPR_theta_before_x.usd",
        config_dir=usd_dir / "configuration",
    )
```

### 2.2 Update URDF Conversion Script (Optional)

Edit `scripts/convert_urdf_to_usd.py` defaults:

```python
parser.add_argument(
    "--urdf",
    type=str,
    default="assets_own/mobile_manipulator_PPR_theta_before_x.urdf",  # Updated
    help="Path to input URDF file (relative to project root)",
)
parser.add_argument(
    "--usd",
    type=str,
    default="assets_own/usd/mobile_manipulator_PPR_theta_before_x.usd",  # Updated
    help="Path to output USD file (relative to project root)",
)
```

## Step 3: Verify Assets

Check that all files exist:

```
✅ assets_own/mobile_manipulator_PPR_theta_before_x.urdf
✅ assets_own/usd/mobile_manipulator_PPR_theta_before_x.usd
✅ assets_own/usd/configuration/
✅ assets_own/meshes/stl_output/*.STL
```

## Step 4: Test with Isaac Lab

### 4.1 Test Environment

```powershell
I:\isaaclab\isaaclab.bat -p scripts/test_mobile_mm_env.py --num_envs 1
```

Expected output:
- Environment loads without errors
- Robot spawns correctly
- Mobile base joints: `joint_theta`, `joint_x`, `joint_y`
- Arm joints: `left_arm_joint_1` through `left_arm_joint_6`

### 4.2 Test Training

```powershell
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 4 -Test
```

Should complete without USD loading errors.

## Step 5: Update MATLAB (Already Done)

MATLAB scripts already updated to use `theta_before_x.urdf`:
- ✅ `build_reachability_map_FK.m`
- ✅ `visualize_fk_map.m`
- ✅ `visualize_stored_configs.m`

## Step 6: Commit Changes

```powershell
git add assets_own/usd/mobile_manipulator_PPR_theta_before_x.usd
git add assets_own/usd/configuration/
git add src/rl_platform/robots/mobile_mm.py
git add scripts/convert_urdf_to_usd.py

git commit -m "Add theta_before_x USD and update Isaac Lab to use correct mobile base joint order"
git push origin train-windows
```

## Troubleshooting

### USD file not loading
- Verify meshes are in `assets_own/meshes/stl_output/`
- Check USD file size (should be ~100KB, not empty)
- Try re-importing URDF with "Fix Base Link" **UNCHECKED**

### Wrong joint order in Isaac Lab
- Check USD stage in Isaac Sim: joints should be in theta→x→y order
- Re-import URDF if joints are out of order
- Clear Isaac Sim cache: `%USERPROFILE%\.nvidia-omniverse\cache`

### Missing configuration directory
- Isaac Sim should create `configuration/` automatically during import
- If missing, re-import and make sure "Import Inertia" is checked
- Copy `configuration/` from old `base_corrected` if needed (contents should be similar)

## Verification Checklist

Before considering update complete:

- [ ] URDF converted to USD with correct settings
- [ ] Configuration directory exists and has USD files
- [ ] Python config updated to use new USD path
- [ ] Test environment runs without errors
- [ ] Joint order verified: theta→x→y for base
- [ ] MATLAB scripts use new URDF (already done)
- [ ] Changes committed to git

## What's Next?

With the corrected joint order:
1. Mobile base planning will work in robot local frame
2. Can implement full mobile manipulation tasks
3. Can train policies that use base rotation + translation
4. Reachability map remains valid (arm-only workspace unchanged)
