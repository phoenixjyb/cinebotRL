# USD Regeneration Checklist - PPR Mass Fix

## ✅ Pre-Regeneration Status
- [x] URDF fixed: PPR helper masses 0.0 → 1.0 kg
- [x] Inertia updated: 0.0001 → 0.001
- [x] Changes committed: c59cda8
- [x] Documentation created: PPR_MASS_FIX_SUMMARY.md

## 📋 Isaac Sim Import Steps

### 1. Launch Isaac Sim
```powershell
# Navigate to Isaac Sim
cd I:\isaaclab\_isaac_sim
.\isaac-sim.bat
```

### 2. Import URDF
**Menu:** `File → Import → URDF`

**Settings (CRITICAL - match previous import):**

| Setting | Value | Reason |
|---------|-------|--------|
| **URDF File** | `C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_base_corrected.urdf` | Source |
| **Import Scale** | `0.001` | Meshes in millimeters → meters |
| **Fix Base** | `☐ Unchecked` | **MOVEABLE BASE!** |
| **Create Physics Scene** | `☑ Checked` | Add PhysicsScene |
| **Joint Drive Type** | `Position` | **ALL PPR joints!** |
| **Joint Stiffness** | `625` | (Will be overridden by env.py) |
| **Joint Damping** | `0.37` | (Will be overridden by env.py) |

**CRITICAL CHECKS:**
- ⚠️ **joint_theta MUST be "Position"** not "Velocity"
- ⚠️ **"Fix Base" MUST be unchecked** (moveable base)
- ⚠️ **Import Scale MUST be 0.001** (mm → m)

### 3. Verify Import
After import completes:

1. **Check Stage Hierarchy:**
   ```
   /World
     /mobile_manipulator_PPR_base_corrected
       base (root) ← Should have RigidBodyAPI
       base_link_x ← Should have mass=1.0
       base_link_y ← Should have mass=1.0
       abstract_chassis_link ← Should have geometry
       ...
   ```

2. **Verify PPR Helper Masses:**
   - Select `base_link_x` → Physics → Mass Properties
   - Should show: **Mass = 1.0 kg**
   - Repeat for `base_link_y`

3. **Verify Joint Controls:**
   - Select `joint_x` → Physics → Joint Drive
   - Should show: **Type = "Position"**, Stiffness = 625
   - Repeat for `joint_y`, `joint_theta`

4. **Check Base Mobility:**
   - Select `base` (root link)
   - Physics tab should show **RigidBodyAPI** (not ArticulationRootAPI with fixed base)

### 4. Save USD
**Menu:** `File → Save As`

**Output Path:**
```
C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd\mobile_manipulator_PPR_base_corrected.usd
```

**Notes:**
- Overwrite existing USD file
- This will update Isaac Lab to use new physics

### 5. Update Configuration USD (Optional)
If configuration folder exists:
```
assets_own/mobile_manipulator_PPR_base_corrected/configuration/Configuration.usd
```
May also need regeneration - check if Isaac Lab uses it.

## 🧪 Quick Test Before Full Training

After USD regeneration, test with minimal script:

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_mobile_mm_env.py
```

**Validation checks (first 100 steps):**
1. No physics warnings about masses or inertias
2. `root_pos_w` should change (not frozen at [1.05, 0.08, 0.0])
3. PPR joint values should accumulate
4. No explosions or NaN errors
5. Base position diagnostics show movement

**Expected output:**
```
[TRACKING Step 50]
  🚗 Base Pos (WORLD): [1.1XX, 0.0XX, 0.0XX]  ← Should be DIFFERENT!
  🔧 Base PPR offsets:   [0.0XX, -0.0XX, 0.0XX] ← Accumulating
```

## 🚀 Full Training Launch

If test passes, launch full training:

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 4096 -Headless
```

Or use direct command:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 4096 `
  --n_steps 128 `
  --batch_size 1024 `
  --total_timesteps 100000000 `
  --learning_rate 3e-4 `
  --ent_coef 0.001 `
  --enable_entropy_decay `
  --final_ent_coef 1e-4 `
  --decay_start_timestep 50000000 `
  --decay_duration_timesteps 50000000 `
  --enable_kl_schedule `
  --kl_warmup 0.25 `
  --kl_main 0.15 `
  --kl_finetune 0.07 `
  --target_kl 1.0 `
  --trajectory_type multi_recorded `
  --use_all_trajectories `
  --headless
```

## 📊 Training Validation (First 1M Steps)

**Monitor these metrics at steps 10K, 100K, 1M:**

| Metric | Training 1 (Phantom) | Training 2 (Expected) |
|--------|---------------------|---------------------|
| `root_pos_w` change | ❌ ~0 cm | ✅ >50 cm |
| PPR accumulation | ✅ Non-zero | ✅ Non-zero |
| EE-base distance | ❌ >4m (far) | ✅ <1m (reachable) |
| Base mobilization | ❌ 0.0000 | ✅ Positive |
| Physics explosions | ✅ None | ✅ None |

**Success Criteria:**
- `root_pos_w` should track `initial_pos + joint_pos` (within 10cm tolerance)
- Base should move when targets are >0.6m away
- No oscillations or instability
- EE stays within ~1m of base (arm reach + base motion)

## 🎯 Expected Training Outcomes

**100M timesteps with correct physics:**
- Policy learns to coordinate arm + base movement
- Base mobilizes for distant targets (>0.6m)
- Distance penalty decreases (base approaches targets)
- EE tracking improves (arm + base working together)
- Final performance >> Training 1 (phantom joints)

---

**Ready to proceed!** Launch Isaac Sim and follow the import checklist above.

**After USD regeneration:** Test → Train → Monitor base movement! 🚀
