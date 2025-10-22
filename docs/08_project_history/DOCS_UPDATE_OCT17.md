# Documentation Update Summary: Trajectory Arguments Added

**Date:** October 17, 2025  
**Status:** ✅ **All Reference Docs Updated**

---

## 🎯 What Was Updated

All training command references have been updated to include the **new trajectory loading arguments**:

```powershell
--trajectory_type multi_recorded `
--use_all_trajectories `
```

And updated hyperparameters:
```powershell
--n_steps 128 `           # Increased from 32 for better GAE
--kl_warmup 0.25 `        # Increased from 0.07
--kl_main 0.15 `          # Increased from 0.02
--kl_finetune 0.07 `      # Increased from 0.01
--target_kl 1.0 `         # Increased from 0.07
```

---

## 📚 Updated Documents

### 1. **TRAINING_COMMAND_QUICK_REF.md** ✅
- **Already had** trajectory arguments
- Shows comparison: circle vs multi_recorded
- Includes test vs train distinction

### 2. **FINAL_TRAINING_COMMAND_With_All_Protections.md** ✅
**Changes:**
- Added `--trajectory_type multi_recorded`
- Added `--use_all_trajectories`
- Updated `--n_steps` from 32 → 128
- Updated `--learning_rate 0.0003` (was missing)
- Updated KL schedule parameters

**Before:**
```powershell
--n_steps 32 `
--target_kl 0.07 `
--headless
```

**After:**
```powershell
--n_steps 128 `
--kl_warmup 0.25 `
--kl_main 0.15 `
--kl_finetune 0.07 `
--target_kl 1.0 `
--trajectory_type multi_recorded `
--use_all_trajectories `
--headless
```

### 3. **QUICK_START.md** ✅
**Changes:**
- Updated main training command with all new arguments
- Updated all 3 phases (Conservative, Aggressive, Maximum)
- Added trajectory loading to each phase

**Before:**
```powershell
--n_steps 32 `
--headless
```

**After:**
```powershell
--n_steps 128 `
--trajectory_type multi_recorded `
--use_all_trajectories `
--headless
```

**All 3 Phases Now Include:**
- Phase 1: `--num_envs 2048 --batch_size 512 --n_steps 128 --trajectory_type multi_recorded --use_all_trajectories`
- Phase 2: `--num_envs 4096 --batch_size 1024 --n_steps 128 --trajectory_type multi_recorded --use_all_trajectories`
- Phase 3: `--num_envs 6144 --batch_size 1536 --n_steps 128 --trajectory_type multi_recorded --use_all_trajectories`

### 4. **READY_TO_TRAIN_Summary.md** ✅
**Changes:**
- Updated complete training command
- Added trajectory arguments
- Updated KL schedule parameters

### 5. **QUICK_START_Entropy_Decay_Training.md** ✅
**Changes:**
- Updated 100M training command
- Added all new arguments including trajectory loading

---

## 📋 Complete Reference Command

**This is the canonical command now referenced in all docs:**

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --batch_size 1024 `
    --n_steps 128 `
    --total_timesteps 100000000 `
    --learning_rate 0.0003 `
    --ent_coef 0.001 `
    --enable_entropy_decay `
    --final_ent_coef 0.0001 `
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

---

## 🔄 Documents That Already Had Trajectory Args

These documents were created with the new trajectory system and already include the correct arguments:

- ✅ `TRAINING_COMMAND_QUICK_REF.md` (created today)
- ✅ `TRAINING_WITH_RECORDED_TRAJECTORIES.md` (created today)
- ✅ `TRAJECTORY_LOADING_VERIFIED.md` (created today)
- ✅ `ALL_TESTS_PASSED.md` (created today)
- ✅ `TRAJECTORY_LOADING_INVESTIGATION.md` (created today)

---

## 📝 Other Documents (Not Updated)

These documents contain older training commands for historical reference or different purposes:

### Historical/Analysis Docs:
- `Policy_Divergence_Analysis_200M_Training.md` - Analysis of old run
- `Why_Both_Entropy_Decay_AND_KL_Schedule.md` - Theoretical analysis
- `Critical_Analysis_Playbook_vs_Reality.md` - Previous analysis
- `Frozen_Base_Investigation_Summary.md` - Debugging base movement
- `BASE_FIX_ACTION_CHECKLIST.md` - Historical checklist

**Reason:** These are historical documents analyzing past runs. Updating them would be confusing.

### Specialized Guides:
- `10_NNdesign/*` - Network architecture discussions
- `03_training/CPU_TRAINING_GUIDE.md` - CPU-specific guidance
- `VISUALIZATION_GUIDE.md` - Visualization commands (different purpose)

**Reason:** These have specific purposes and the commands are examples, not canonical references.

---

## ✅ Verification

To verify the updates, search for the training command in these files:

```powershell
# Should all show the new trajectory arguments
Select-String -Path "docs\TRAINING_COMMAND_QUICK_REF.md" -Pattern "trajectory_type"
Select-String -Path "docs\FINAL_TRAINING_COMMAND_With_All_Protections.md" -Pattern "trajectory_type"
Select-String -Path "docs\QUICK_START.md" -Pattern "trajectory_type"
Select-String -Path "docs\READY_TO_TRAIN_Summary.md" -Pattern "trajectory_type"
Select-String -Path "docs\QUICK_START_Entropy_Decay_Training.md" -Pattern "trajectory_type"
```

Expected output: All should show `--trajectory_type multi_recorded` and `--use_all_trajectories`

---

## 🎯 Impact

**Before Update:**
- Users would run training with default `circle` trajectory
- Only 1 synthetic trajectory used
- Low diversity, poor generalization

**After Update:**
- All reference docs point to multi-trajectory training
- 1,038 real trajectories loaded by default
- High diversity, better generalization
- 50/50 split (chassis-requiring vs arm-only)

---

## 📚 User Journey

When users look for training commands, they'll find consistent information:

1. **Quick start?** → `QUICK_START.md` ✅ Has trajectory args
2. **Full command?** → `FINAL_TRAINING_COMMAND_With_All_Protections.md` ✅ Has trajectory args
3. **Just the command?** → `TRAINING_COMMAND_QUICK_REF.md` ✅ Has trajectory args
4. **Understanding entropy?** → `QUICK_START_Entropy_Decay_Training.md` ✅ Has trajectory args
5. **Ready to train summary?** → `READY_TO_TRAIN_Summary.md` ✅ Has trajectory args

**All paths lead to the correct, up-to-date command!** ✅

---

## 🔍 Next Time Someone Asks

**"What's the command to train?"**

Point them to any of these (all consistent now):
- `docs/TRAINING_COMMAND_QUICK_REF.md` - Quick reference
- `docs/QUICK_START.md` - General quick start
- `docs/FINAL_TRAINING_COMMAND_With_All_Protections.md` - Full explanation

All will show:
✅ Multi-trajectory loading  
✅ Updated hyperparameters  
✅ All protections (entropy decay, KL schedule)  
✅ Correct n_steps (128)  

---

## Summary

✅ **5 key reference documents updated**  
✅ **All training commands now include trajectory loading**  
✅ **Consistent hyperparameters across all docs**  
✅ **Users will train on 1,038 trajectories by default**  
✅ **No confusion about which command to use**  

**The documentation is now aligned with the verified, working trajectory loading system!** 🎉
