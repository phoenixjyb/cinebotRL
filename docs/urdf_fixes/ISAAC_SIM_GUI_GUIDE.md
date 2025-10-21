# How to Open Isaac Sim GUI for URDF Conversion

## Step-by-Step Instructions

### 1. Open Isaac Sim GUI

Run this command in PowerShell:

```powershell
cd I:\isaaclab\_isaac_sim
.\isaac-sim.bat
```

**Or** create a shortcut and double-click it:
- Target: `I:\isaaclab\_isaac_sim\isaac-sim.bat`
- Start in: `I:\isaaclab\_isaac_sim`

**Wait time**: Isaac Sim takes ~1-2 minutes to fully load.

### 2. Import URDF

Once Isaac Sim GUI opens:

1. **Locate URDF Importer** - Try these menu paths (varies by Isaac Sim version):

   **Option A: Window Menu (most common)**
   - `Window` → `Extensions` 
   - Search for: `urdf` or `URDF Importer`
   - Enable the extension if not already enabled
   - Then go to: `Window` → `Isaac Utils` → `URDF Importer`

   **Option B: Direct Menu**
   - `Isaac Utils` → `URDF Importer` (if Isaac Utils visible in top menu)
   
   **Option C: File Import**
   - `File` → `Import` → look for URDF option

2. **URDF Import Dialog Opens**

### 3. Configure Import Settings

In the URDF Importer dialog:

#### **Input Settings**:
- **Input File**: Browse to:
  ```
  C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_base_corrected.urdf
  ```

#### **Import Configuration** (CRITICAL - Match Screenshot Settings):

**Section: Model**
- ⚪ **Model Type**: Select "Referenced Model" (not "Create in Stage")
- **USD Output**: Keep as "Same as Imported Model (Default)"

**Section: Links**  
- ⚪ **Base Type**: Select "Moveable Base" ✅ (NOT "Static Base"!)
- **Default Density**: Leave as `0.0` (uses URDF inertia)

**Section: Joints & Drives**
- ☐ **Ignore Mimic**: Leave unchecked
- ⚪ **Joint Configuration**: Select "Stiffness" (not "Natural Frequency")
- ⚪ **Drive Type**: Select "Force" (not "Acceleration")

**Critical Joint Settings** (in the table at bottom):
- Row 1: `joint_theta` → **Target: "Velocity"** (allows rotation commands)
- Row 2-3: `joint_x`, `joint_y` → **Target: "Position"** (translation commands)
- Row 4+: All arm joints → **Target: "Position"**

**Values to verify**:
- Natural Frequency: `25.0` (already shown correctly)
- Damping Ratio: `0.005` (already shown correctly)

**MISSING FROM SCREENSHOT - Check for these options elsewhere**:
- ✅ **Mesh Scale**: Must set to `0.001` (millimeters → meters)
- ✅ **Import Inertia Tensor**: Must enable
- ❌ **Fix Base Link**: Must disable (mobile robot!)
- ❌ **Self Collision**: Disable (can enable later)

#### **Joint Settings**:
- **Default Drive Type**: `Position`
- **Position Drive Stiffness**: `10000.0`
- **Position Drive Damping**: `1000.0`

#### **Output Settings**:
- **Output Directory**: Leave default or specify:
  ```
  C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd
  ```

### 4. Click "Import"

- Wait 10-30 seconds for conversion
- Robot should appear in viewport when done

### 5. Verify Import (CRITICAL PHYSICS CHECKS)

#### Visual Check (Isaac Sim viewport):
- ✅ Robot appears (not giant or tiny - should be ~1m tall)
- ✅ Base link exists (check scene hierarchy)
- ✅ PPR joints visible: `joint_x`, `joint_y`, `joint_theta`
- ✅ Arm joints visible: `left_arm_joint1` through `left_arm_joint6`

#### Physics Check (CRITICAL - DO THIS):

1. **Select** `abstract_chassis_link` in scene hierarchy
2. **Property Panel** → Check for:
   - ✅ **PhysicsRigidBodyAPI**: Should be present
   - ✅ **PhysicsCollisionAPI**: Should be present
   - ✅ **Physics:mass**: Should show ~30.96 kg

3. **Check joint_theta limits**:
   - Select `joint_theta` in hierarchy
   - Property panel → Physics Drive
   - ✅ Lower limit: Should be `-6.283185` (NOT 0.0!)
   - ✅ Upper limit: Should be `6.283185` (NOT 0.0!)
   - ❌ If both are 0.0 → **IMPORT FAILED** - theta is locked

4. **Check base mass**:
   - Select `base` link
   - Property panel → Physics
   - ✅ Mass: Should show ~20.0 kg (NOT 0.0!)

**If any checks fail**: See [URDF_FIXES_APPLIED.md](URDF_FIXES_APPLIED.md) for debugging

### 6. Save USD File

**Method A - If Auto-Saved**:
- Check: `I:\isaaclab\_isaac_sim\usd\` for generated files
- Copy to: `C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd\mobile_manipulator_PPR_base_corrected.usd`

**Method B - Manual Save**:
1. `File` → `Save As`
2. Navigate to: `C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd\`
3. Filename: `mobile_manipulator_PPR_base_corrected.usd`
4. Click `Save` (overwrite if prompted)

### 7. Verify USD File

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL

# Check file exists and size
ls assets_own\usd\mobile_manipulator_PPR_base_corrected.usd

# Should be ~2-5 MB
```

---

## Troubleshooting

### Problem: "Isaac Utils menu not found"

**Solution 1**: Check under `Window` → `Extensions`:
- Search for "URDF"
- Enable "omni.isaac.urdf_importer" extension
- Restart Isaac Sim

**Solution 2**: Use Python Console in Isaac Sim:
```python
from omni.isaac.urdf import _urdf

urdf_interface = _urdf.acquire_urdf_interface()
urdf_interface.parse_urdf(
    "C:/Users/yanbo/wSpace/cinebotRL/assets_own/mobile_manipulator_PPR_base_corrected.urdf",
    _urdf.ImportConfig(),
    "C:/Users/yanbo/wSpace/cinebotRL/assets_own/usd/mobile_manipulator_PPR_base_corrected.usd"
)
```

### Problem: "Robot appears HUGE in viewport"

**Cause**: Forgot mesh scale 0.001  
**Fix**: Re-import with correct mesh scale

### Problem: "Mass warnings in log"

**Check**: Open generated USD in text editor, search for mass values:
```powershell
Select-String -Path "assets_own\usd\mobile_manipulator_PPR_base_corrected.usd" -Pattern "mass"
```

Should see:
- base: 20.0
- abstract_chassis_link: ~31.0

---

## After USD Conversion

### Test Environment

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL

# Test robot loads correctly
I:\isaaclab\isaaclab.bat -p scripts\test_mobile_mm_env.py
```

**Look for**:
- ✅ Robot spawns without errors
- ✅ No "invalid inertia" warnings
- ✅ No "zero mass" or "fixed link" warnings  
- ✅ PPR joints functional

### Start Training

```powershell
# Launch training with new USD
.\scripts\launch_training_windows.ps1 -Headless -NumEnvs 4096
```

**Monitor**:
- First 100K steps: Base should start moving
- TensorBoard: Watch `root_pos_w` changing
- Distance penalty activating when target far

---

## Quick Reference Commands

```powershell
# 1. Open GUI
cd I:\isaaclab\_isaac_sim
.\isaac-sim.bat

# 2. After conversion, test
cd C:\Users\yanbo\wSpace\cinebotRL
I:\isaaclab\isaaclab.bat -p scripts\test_mobile_mm_env.py

# 3. Start training
.\scripts\launch_training_windows.ps1 -Headless -NumEnvs 4096
```

---

**Expected Result**: With 20kg movable base, robot should move to track trajectories! 🚀
