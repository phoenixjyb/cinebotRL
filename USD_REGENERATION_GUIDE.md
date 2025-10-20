# USD Regeneration Instructions

**Date**: October 21, 2025  
**Reason**: Critical physics fixes for base mobility and simulation stability

## Changes in URDF (5 Critical Fixes)

1. ✅ **base link**: 0.0 → 20.0 kg (was immovable/static, now movable)
2. ✅ **abstract_chassis_link**: 50.96 → 30.96 kg (fixed mass duplication)
3. ✅ **base_link_x/y**: 0.001 → 0.0 kg (eliminates numerical stiffness)
4. ✅ **joint_theta limits**: -inf/+inf → ±6.283185 rad (prevents yaw lock in USD)
5. ✅ **Inertia tensors**: Realistic values for 20kg base

**Total mobile platform**: 51 kg (20kg base + 31kg chassis, correct)

## Method 1: Isaac Sim GUI (RECOMMENDED)

1. **Open Isaac Sim** (not through Isaac Lab):
   ```powershell
   # Navigate to Isaac Sim installation
   cd "C:\Program Files\NVIDIA\Omniverse\pkg\isaac-sim-5.0.0"
   
   # Launch Isaac Sim GUI
   .\isaac-sim.bat
   ```

2. **Import URDF**:
   - File → Import → URDF
   - Select file: `C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_base_corrected.urdf`
   
3. **Configure Import Settings**:
   - ✅ **Mesh Scale**: `0.001` (convert mm to meters)
   - ✅ **Import Inertia Tensor**: Enabled
   - ✅ **Merge Fixed Joints**: Disabled
   - ❌ **Fix Base**: Disabled (mobile robot)
   - ❌ **Self Collision**: Disabled
   - **Drive Type**: Position
   - **Position Drive Damping**: `1000.0`
   - **Position Drive Stiffness**: `10000.0`

4. **Save USD**:
   - File → Save As
   - Location: `C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd\mobile_manipulator_PPR_base_corrected.usd`
   - ✅ Overwrite existing file

5. **Verify** (optional):
   - Check in viewport that robot looks correct
   - Check that base link has mass (not zero)
   - Check PPR joints exist (joint_x, joint_y, joint_theta)

## Method 2: Command Line (if available)

If Isaac Sim has command-line URDF converter:

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL\assets_own

# Check if tool exists
"C:\Program Files\NVIDIA\Omniverse\pkg\isaac-sim-5.0.0\isaac-sim.bat" --help

# Convert (if supported)
"C:\Program Files\NVIDIA\Omniverse\pkg\isaac-sim-5.0.0\isaac-sim.bat" convert urdf ` mobile_manipulator_PPR_base_corrected.urdf `
  usd/mobile_manipulator_PPR_base_corrected.usd `
  --mesh-scale 0.001
```

## Method 3: Python Script (Alternative)

If you prefer automation, use the conversion script:

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL

# Run converter through Isaac Lab
I:\isaaclab\isaaclab.bat -p scripts\convert_urdf_to_usd.py
```

**Note**: The Python script may need module path adjustments for Isaac Sim 5.0.

## Verification Checklist

After regeneration, verify:

- [ ] USD file exists: `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`
- [ ] USD file size reasonable (~2-5 MB)
- [ ] Configuration directory exists: `assets_own/usd/configuration/`

### Critical Physics Verification

**Run environment test**:
```powershell
I:\isaaclab\isaaclab.bat -p scripts\test_mobile_mm_env.py
```

**Check for these issues** (from logs):

1. ✅ **No zero-mass warnings**: 
   - Search log for: "invalid inertia" or "zero mass" 
   - Should see: base mass = 20.0 kg, chassis mass = 31.0 kg

2. ✅ **joint_theta NOT locked**:
   - Search USD/log for: `joint_theta` limits
   - Should be: `lower=-6.283185, upper=6.283185` (NOT `lower=0, upper=0`)

3. ✅ **Collision geometry enabled**:
   - Open USD in Isaac Sim
   - Select `abstract_chassis_link` mesh
   - Property panel should show: **PhysicsCollisionAPI** enabled
   - Approximation: Convex Hull or Mesh (NOT "None" or visual-only)

4. ✅ **No numerical warnings**:
   - No "solver failed" or "joint explosion" errors during simulation
   - Base should move smoothly when commanded

**If any checks fail**: See [docs/URDF_PHYSICS_ISSUES_REMAINING.md](docs/URDF_PHYSICS_ISSUES_REMAINING.md) for debugging

## Next Steps After USD Regeneration

1. **Test Environment**:
   ```powershell
   I:\isaaclab\isaaclab.bat -p scripts\test_mobile_mm_env.py
   ```
   - Verify robot spawns correctly
   - Check base is movable (not fixed)
   - Verify PPR joints functional

2. **Start Training**:
   ```powershell
   .\scripts\launch_training_windows.ps1 -Headless -NumEnvs 4096
   ```
   - Monitor first 100K-1M steps
   - Watch for base world position changing
   - Expect base mobilization to emerge quickly

3. **Monitor Diagnostics**:
   - Check TensorBoard logs
   - Look for base movement in world frame
   - Distance penalty should activate when target far
   - Base should approach target to reduce penalty

## Expected Training Behavior

With fixed URDF (20kg movable base):

- **First 100K steps**: Policy explores, base starts moving
- **500K-1M steps**: Coordinated arm+base movement emerges  
- **5M+ steps**: Smooth trajectory tracking with mobile base

Old behavior (zero-mass base): Base frozen at spawn, no movement ever.

---

**Status**: ✅ URDF fixed, ready for USD regeneration  
**Last URDF Update**: commit 02b3e2f (mass distribution fix)
