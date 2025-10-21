# 🎯 IMMEDIATE ACTION REQUIRED - USD Import

## ✅ Status
- **URDF Fixed:** PPR helpers 0.0 → 1.0 kg (commit c59cda8)
- **Isaac Sim:** Launching now
- **Next:** Import URDF → Generate USD

## 📋 Quick Import Steps

### 1. Wait for Isaac Sim GUI
Isaac Sim is launching... wait for main window to appear (30-60 seconds)

### 2. Import URDF
**Menu:** `File → Import → URDF`

**Browse to:**
```
C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_base_corrected.urdf
```

### 3. CRITICAL SETTINGS ⚠️

| Setting | Value |
|---------|-------|
| **Import Scale** | `0.001` |
| **Fix Base** | `☐ UNCHECKED` |
| **Joint Drive** | `Position` |

### 4. Verify After Import

**Check PPR helper masses:**
- Select `base_link_x` → Physics panel
- Should show: **Mass = 1.0 kg** ✓
- Select `base_link_y` → Physics panel  
- Should show: **Mass = 1.0 kg** ✓

**Check joint controls:**
- Select `joint_theta` → Physics panel
- Should show: **Drive Type = Position** ✓

### 5. Save USD

**Menu:** `File → Save As`

**Save to:**
```
C:\Users\yanbo\wSpace\cinebotRL\assets_own\usd\mobile_manipulator_PPR_base_corrected.usd
```

**Overwrite:** Yes (replace old USD)

### 6. Test Immediately

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_mobile_mm_env.py
```

**Watch for:**
```
🚗 Base Pos (WORLD): [1.XXX, 0.XXX, 0.XXX]  ← Should CHANGE!
```

### 7. If Test Passes → Full Training

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 4096 --headless ...
```

---

**Critical Success Factors:**
- ✅ Import scale 0.001 (mm → m)
- ✅ Fix Base UNCHECKED (moveable!)
- ✅ All joints Position control
- ✅ Verify mass = 1.0 kg after import

**Expected Result:**
- `root_pos_w` will update when PPR joints move
- Base will physically move (no more phantom joints!)
- Training will show real coordinated motion

---

**Ready!** Proceed with import when Isaac Sim GUI appears! 🚀
