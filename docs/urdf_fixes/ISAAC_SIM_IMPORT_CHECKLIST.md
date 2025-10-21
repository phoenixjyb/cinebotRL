# Isaac Sim URDF Import - Settings Checklist

**Reference**: Screenshot from Isaac Sim 5.0 URDF Importer dialog

## Configuration Sections (Top to Bottom)

### 1. Model Section
- [ ] **Model Type**: ⚪ Select "Referenced Model"
  - *Not "Create in Stage"*
- [ ] **USD Output**: Keep default "Same as Imported Model (Default)"

---

### 2. Links Section ⚠️ CRITICAL
- [ ] **Base Type**: ⚪ Select **"Moveable Base"** ✅
  - **NOT "Static Base"** - This would freeze the robot!
  - Screenshot shows this correctly selected
- [ ] **Default Density**: Keep as `0.0` Kg/m³
  - *Uses URDF inertia data instead of density*

---

### 3. Joints & Drives Section

#### Checkboxes:
- [ ] **Ignore Mimic**: ☐ Leave **unchecked**

#### Joint Configuration:
- [ ] ⚪ Select **"Stiffness"** (not "Natural Frequency")

#### Drive Type:
- [ ] ⚪ Select **"Force"** (not "Acceleration")

#### Joint Table (Bottom of Dialog):
Check that the table shows these **Target** types:

| Name | Target | Nat...ncy | Da...tio |
|------|--------|-----------|----------|
| **joint_theta** | **Position** ⬇️ | 25.0 | 0.005 |
| **joint_x** | **Position** ⬇️ | 25.0 | 0.005 |
| **joint_y** | **Position** ⬇️ | 25.0 | 0.005 |
| left_arm_joint1 | Position ⬇️ | 25.0 | 0.005 |
| left_arm_joint2 | Position ⬇️ | 25.0 | 0.005 |
| *(continue for all arm joints)* | Position ⬇️ | 25.0 | 0.005 |

**CRITICAL - Your Screenshot Shows joint_theta as "Velocity" but this is WRONG**:

Your `env.py` (line 720-724) uses:
```python
robot.set_joint_position_target(  # ← POSITION target!
    target=new_base_targets,      # Includes [x, y, theta]
    joint_ids=[joint_x, joint_y, joint_theta]
)
```

**All PPR joints receive POSITION commands**, even though policy outputs velocities (v_x, ω_z).  
The velocities are integrated to position deltas in `env.py` before sending to PhysX.

**Required Changes**:
- ❌ `joint_theta`: Change "Velocity" → **"Position"** in Isaac Sim dialog
- ✅ `joint_x` and `joint_y`: Already correct as "Position"
- ✅ All arm joints: Already correct as "Position"

---

### 4. Settings NOT Visible in Screenshot (Check Other Tabs/Sections)

#### **CRITICAL - Mesh Scale**:
- [ ] **Mesh Scale**: Set to `0.001`
  - *Converts millimeters (CAD) to meters (Isaac Sim)*
  - *Without this, robot will be 1000× too large!*
  - **Location**: May be in "Import" or "Mesh" section

#### **Physics Settings**:
- [ ] **Import Inertia Tensor**: ☑ **Enable**
  - *Preserves mass/inertia from URDF*
- [ ] **Fix Base Link**: ☐ **Disable**
  - *Mobile robot - base must move!*
- [ ] **Merge Fixed Joints**: ☐ **Disable**
  - *Keep all joints*
- [ ] **Self Collision**: ☐ **Disable** (optional, can enable later)
- [ ] **Create Physics Scene**: ☑ **Enable**
  - *Ensures collision meshes have PhysX*

---

## Why These Settings Matter

### Moveable Base (NOT Static)
- **Static Base** = robot welded to ground (training impossible)
- **Moveable Base** = robot can move freely ✅

### joint_theta as "Velocity" Target
- Allows continuous rotation (no wrapping at ±π)
- Policy can command angular velocity directly
- Prevents joint limit issues with ±2π bounds

### joint_x/joint_y as "Position" Target
- Direct translation control
- No velocity integration needed
- Matches PPR kinematic chain design

### Mesh Scale 0.001
- URDF meshes in millimeters (from CAD)
- Isaac Sim uses meters
- Scale 0.001 = divide by 1000 (mm → m)

### Force Drive Type
- More stable than acceleration drives
- Matches env.py spring-damper model (10kN/m stiffness)
- Better for contact-rich manipulation

---

## Post-Import Verification

After clicking "Import", check:

1. **Robot size**: Should be ~1 meter tall (not tiny or giant)
2. **Scene hierarchy**: All links/joints present
3. **Property panel** (select `abstract_chassis_link`):
   - Mass: ~30.96 kg ✅
   - PhysicsCollisionAPI: Present ✅
4. **joint_theta limits** (select joint):
   - Lower: -6.283185 ✅ (NOT 0.0)
   - Upper: 6.283185 ✅ (NOT 0.0)

If any verification fails → See [docs/URDF_PHYSICS_ISSUES_REMAINING.md](docs/URDF_PHYSICS_ISSUES_REMAINING.md)

---

## Common Import Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| Robot 1000× too large | Mesh scale = 1.0 | Re-import with scale = 0.001 |
| Robot 1000× too small | Mesh scale = 1000 | Re-import with scale = 0.001 |
| Base frozen at spawn | "Static Base" selected | Re-import with "Moveable Base" |
| Base won't rotate | joint_theta limits = 0 | Check URDF has ±6.28 limits |
| No ground contact | Collision not enabled | Check PhysicsCollisionAPI |
| Physics explosions | Drive type = Acceleration | Use "Force" drive type |

---

## Quick Reference: Correct Settings

```yaml
Model:
  Type: Referenced Model
  
Links:
  Base: Moveable Base ✅
  Density: 0.0
  
Joints:
  Configuration: Stiffness
  Drive Type: Force
  
  Joint Targets:
    joint_theta: Velocity   # ← Rotation
    joint_x: Position       # ← Translation X
    joint_y: Position       # ← Translation Y
    arm_joints: Position    # ← All arm DOF

Import:
  Mesh Scale: 0.001         # ← mm to meters
  Import Inertia: True
  Fix Base: False           # ← Mobile robot
  Self Collision: False
```

---

**Next**: After import succeeds → [Save USD and verify](#6-save-usd-file) in main guide
