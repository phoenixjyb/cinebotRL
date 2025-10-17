# ✅ BASE MOVEMENT FIX - Action Checklist

**Date**: 2025-10-17  
**Status**: Code fixes complete, USD regeneration required

---

## Summary

Found **THREE critical bugs** preventing base movement:

1. ✅ **Action scaling missing** - Fixed in commit 2965c71
2. ✅ **URDF limits zero** - Fixed in commit d9304a0
3. ✅ **No actuator config** - Fixed in commit d9304a0

All code changes committed and documented.

---

## ⚠️ CRITICAL: USD Must Be Regenerated

The URDF file was updated but the USD file is still old!

**Why this matters:**
- Isaac Sim loads the USD file (not URDF directly)
- USD contains cached joint limits from old URDF
- Training will still use `effort=0, velocity=0` until USD regenerated

**Current files:**
- ✅ URDF updated: `assets_own/mobile_manipulator_PPR_base_corrected.urdf`
- ❌ USD outdated: `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`

---

## Required Actions Before Training

### Step 1: Regenerate USD from URDF 🔴 REQUIRED

You need to convert the updated URDF to USD format.

**Option A: Use Isaac Sim USD Converter**
```bash
# If you have Isaac Sim installed
# Open Isaac Sim GUI
# File → Import → URDF
# Select: assets_own/mobile_manipulator_PPR_base_corrected.urdf
# Save as: assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
```

**Option B: Use command line tool** (if available)
```bash
# Check if you have urdf2usd tool
isaacsim --help

# Convert URDF to USD
cd assets_own
isaacsim convert urdf mobile_manipulator_PPR_base_corrected.urdf usd/mobile_manipulator_PPR_base_corrected.usd
```

**Option C: Python script** (recommended if you have it)
```python
# scripts/convert_urdf_to_usd.py
from omni.isaac.urdf import _urdf

urdf_path = "C:/Users/yanbo/wSpace/cinebotRL/assets_own/mobile_manipulator_PPR_base_corrected.urdf"
usd_path = "C:/Users/yanbo/wSpace/cinebotRL/assets_own/usd/mobile_manipulator_PPR_base_corrected.usd"

_urdf.acquire_urdf_interface().import_urdf(urdf_path, usd_path)
print(f"✓ Converted {urdf_path} to {usd_path}")
```

### Step 2: Verify USD Contains New Limits

After regeneration, check the USD file:

```python
# Quick check script
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("assets_own/usd/mobile_manipulator_PPR_base_corrected.usd")

# Check joint_x
joint_x = stage.GetPrimAtPath("/mobile_manipulator/joint_x")
print(f"joint_x max force: {joint_x.GetAttribute('physics:maxForce').Get()}")
print(f"joint_x max velocity: {joint_x.GetAttribute('physics:maxVelocity').Get()}")

# Should show: maxForce=200.0, maxVelocity=2.0
# If still 0: USD not regenerated correctly!
```

### Step 3: Run Verification Test

After USD regeneration:

```bash
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_base_movement_fix.py
```

**Expected output:**
```
[4/4] Results Analysis (env 0):
  Total displacement:
    ΔX: +1.500 m
    Δθ: +2.000 rad (+114.6°)

✅ BASE MOVEMENT CORRECT - Bug is FIXED!
   Distance within 10cm of expected
   Rotation within 11° of expected
   Base actions are properly scaled to velocity limits.
```

**If still broken:**
```
❌ BASE MOVEMENT WEAK - Bug still present!
   Distance: ~0.00m (no movement)
   
   → USD file not regenerated correctly!
   → Still using old URDF with effort=0, velocity=0
```

### Step 4: Visual Inspection (Optional)

Load environment and manually command base:

```bash
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\test_mobile_mm_env.py
```

Watch the robot - base should physically move when actions sent.

---

## After Verification Passes

### Step 5: Commit Regenerated USD

```bash
git add assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
git commit -m "asset: Regenerate USD from updated URDF with working joint limits

Regenerated USD to include:
- joint_x: effort=200N, velocity=2.0 m/s (was 0)
- joint_y: effort=200N, velocity=2.0 m/s (was 0)
- joint_theta: effort=100Nm, velocity=2.5 rad/s (was 0)

Verified with test_base_movement_fix.py
Base now moves as expected!"
```

### Step 6: Start Fresh Training

**IMPORTANT**: Cannot use old checkpoints! Must retrain from scratch.

```bash
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --batch_size 1024 `
    --n_steps 128 `
    --total_timesteps 100000000 `
    --ent_coef 0.001 `
    --enable_entropy_decay `
    --final_ent_coef 0.0001 `
    --decay_start_timestep 50000000 `
    --decay_duration_timesteps 50000000 `
    --enable_kl_schedule `
    --kl_warmup 0.07 `
    --kl_main 0.02 `
    --kl_finetune 0.01 `
    --target_kl 0.07 `
    --headless
```

### Step 7: Monitor Training Metrics

Watch in TensorBoard for signs base is working:

**✅ Good signs:**
- Base velocity observations (dims 0-2): **Non-zero values**
- Base action outputs (actions 6-7): **Variation, not stuck at 0**
- Tracking error: **Decreases faster** (base helps arm reach)
- Episode length: **Robot repositions** to track distant targets

**❌ Bad signs (USD not regenerated):**
- Base velocity observations: Still all zeros
- Base actions: Policy outputs near-zero (learned it's useless)
- Tracking error: Only improves with arm movement
- Visual: Base completely static during evaluation

---

## Troubleshooting

### Issue: Verification test still shows no movement

**Likely cause**: USD file not regenerated

**Check:**
```bash
# Check USD file modification date
ls -l assets_own/usd/mobile_manipulator_PPR_base_corrected.usd

# Should be today's date!
# If old date: USD not regenerated
```

**Fix**: Repeat Step 1 (USD regeneration)

### Issue: USD regeneration fails

**Option 1**: Use Isaac Sim GUI (most reliable)
- Open Isaac Sim application
- File → Import → URDF
- Manual conversion

**Option 2**: Ask for help with conversion
- URDF file is correct
- Just needs USD format conversion
- Can share URDF if needed

### Issue: Training still shows frozen base

**Check:**
1. USD regenerated? (file date today)
2. USD loaded correctly? (check environment startup logs)
3. Actuator config working? (should see "Base actuator initialized")
4. Actions reaching physics? (add debug prints in _pre_physics_step)

---

## Files Changed

### Code Changes (✅ Committed)
1. `src/rl_platform/tasks/mobile_mm/env.py`
   - Added action scaling (lines 482-483)
   - Added base actuator config (lines 147-152)

2. `assets_own/mobile_manipulator_PPR_base_corrected.urdf`
   - Updated joint_x limits: effort 0→200, velocity 0→2.0
   - Updated joint_y limits: effort 0→200, velocity 0→2.0
   - Updated joint_theta limits: effort 0→100, velocity 1.6→2.5

### Asset Changes (⚠️ TODO)
3. `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`
   - **NOT YET REGENERATED**
   - Must regenerate from URDF before training!

### Documentation (✅ Complete)
4. `docs/BUG_REPORT_Frozen_Base.md` - Initial analysis
5. `docs/Frozen_Base_Investigation_Summary.md` - Full investigation
6. `docs/BASE_MOVEMENT_COMPREHENSIVE_ANALYSIS.md` - All three bugs
7. `scripts/test_base_movement_fix.py` - Verification script

---

## Success Criteria

Training is ready when:

- [x] Code fixes committed
- [x] Documentation complete
- [ ] USD file regenerated from updated URDF
- [ ] Verification test passes (base moves ~1.5m in 1 second)
- [ ] Visual check shows base physically moving
- [ ] Git history clean (USD changes committed)

**Current status**: 2/6 complete (code done, need USD regeneration)

---

## Timeline Estimate

- USD regeneration: 5-10 minutes
- Verification test: 2-3 minutes
- Visual check: 5 minutes
- **Total**: ~15-20 minutes before training ready

---

## Contact Points

If USD regeneration unclear:
1. Check Isaac Sim documentation for URDF import
2. Look for `urdf2usd` converter tool
3. Check if conversion script exists in project
4. Can provide Python script for conversion if needed

---

**Next immediate action**: Regenerate USD file from URDF! 🚀
