# Contact Forces Debug Results

**Date**: October 22, 2025  
**Test**: `scripts/debug_contact_forces.py` with 16 envs, 100 steps  
**Status**: ✅ Environment working, ❌ Contact forces broken

---

## Summary

The debug script successfully ran and revealed that **contact forces are reading 0.0 N throughout the entire simulation**. This confirms the code review finding that self-collision detection is not working.

---

## ✅ Good News

1. **Environment Stable**
   - All 16 environments running without crashes
   - No "broken" environments (>2m error)
   - Simulation completed 100 steps successfully

2. **Base IS Moving!** 🎉
   - PPR joint positions changing over time
   - Base mobilization rewards positive (0.0-0.38)
   - Position deltas showing base motion:
     ```
     Step 0:   PPR offsets: [-0.013,  0.000, -0.068]
     Step 100: PPR offsets: [ 0.178, -0.000,  0.044]  (Env 7)
     ```

3. **Jerk Penalty Fix Working** ✅
   - Base mobilization rewards are positive
   - No catastrophic negative rewards
   - Policy can now learn to move base

---

## ❌ Critical Issue

### Contact Forces Reading Zero

**Observation:**
```
Contact forces shape: torch.Size([16, 10, 3])
Max contact force: 0.0000 N
⚠️  WARNING: Contact forces are zero!
```

**Root Cause:**
The environment prints this warning at line 970:
```
[WARNING] Contact forces API not found - collision detection disabled!
```

This means the code is hitting the fallback path in `env.py`:

```python
# env.py lines 960-976
try:
    # Try to get net contact forces from PhysX view
    net_contact_forces = self.robot.root_physx_view.get_net_contact_forces()
except AttributeError:
    # Fallback: try body_net_contact_force_w from robot data
    try:
        net_contact_forces = self.robot.data.body_net_contact_force_w
    except AttributeError:
        # Last resort: use zeros but warn once
        if not hasattr(self, '_contact_force_warning_shown'):
            print("[WARNING] Contact forces API not found - collision detection disabled!")
            self._contact_force_warning_shown = True
        net_contact_forces = torch.zeros(
            (self.num_envs, len(self.robot.body_names), 3),
            device=self.device
        )
```

**Impact:**
- `self_collision_penalty` is always 0.0 (no feedback)
- Policy has no incentive to avoid arm-chassis collisions
- Base repositioning to avoid collisions is not learned
- Arm may collide with chassis without penalty

---

## 🔧 Required Fix

### Investigation Needed

We need to find the correct Isaac Lab 2.2.0 API for contact forces. The current code tries:

1. ❌ `self.robot.root_physx_view.get_net_contact_forces()` - AttributeError
2. ❌ `self.robot.data.body_net_contact_force_w` - AttributeError
3. ✅ Falls back to zeros (current behavior)

### Possible Solutions

1. **Check Isaac Lab 2.2.0 Contact Sensor API**
   - Look for `ContactSensor` class or similar
   - Check if contact forces need explicit sensor setup
   - Review Isaac Lab examples for contact force usage

2. **Check Articulation Contact Methods**
   - `self.robot` is an `Articulation` object
   - Check for methods like:
     - `get_contact_forces()`
     - `contact_forces`
     - `body_contact_forces_w`
   - May need to enable contact reporting in scene config

3. **Add Contact Sensor to Scene**
   - Isaac Lab might require explicit `ContactSensor` addition
   - Check `MobileMMTrackEEEnvCfg` scene configuration
   - Add contact sensor if missing

### Testing Approach

1. Check Isaac Lab documentation/examples for contact forces
2. Try alternative APIs in a test script
3. If needed, add explicit contact sensor to scene config
4. Verify non-zero forces during arm-chassis collisions
5. Test self-collision penalty triggers correctly

---

## 📊 Environment Metrics (From Debug Run)

### Step 50 Statistics
- **Tracking Error**: 1.18-1.30m (mean 1.25m)
- **Base-Target Distance**: 0.44-0.54m (mean 0.48m)
- **Environment Health**: 100% "Poor" (0.3-2.0m error)
- **Base Mobilization**: 0.0-0.09 (positive! ✅)

### Step 100 Statistics
- **Tracking Error**: 1.51-1.73m (mean 1.64m)
- **Base-Target Distance**: 0.69-1.00m (mean 0.82m)
- **Environment Health**: 100% "Poor" (0.3-2.0m error)
- **Base Mobilization**: 0.0-0.38 (positive and increasing! ✅)

**Note**: Error increased from step 50→100 because:
- Random actions (not trained policy)
- No proper coordination
- Still validates base IS moving

---

## 🎯 Impact Assessment

### Without Contact Force Fix

**What Works:**
- ✅ Base can move (jerk penalty fixed)
- ✅ EE tracking rewards working
- ✅ Environment stable

**What's Missing:**
- ❌ No penalty for arm-chassis collisions
- ❌ No incentive to reposition base to avoid collisions
- ❌ Policy won't learn collision-aware behaviors
- ❌ May get stuck in configurations where arm hits chassis

### With Contact Force Fix

**Expected Improvements:**
- ✅ Self-collision penalty triggers
- ✅ Policy learns to avoid arm-chassis contact
- ✅ Base repositioning becomes valuable (avoids collisions)
- ✅ More robust whole-body coordination

---

## 🚀 Decision Point

### Option 1: Launch Session 6 WITHOUT Contact Fix

**Pros:**
- Jerk penalty fix alone might be enough
- Can see if base mobility improves tracking
- Faster to start training

**Cons:**
- No collision avoidance learned
- May develop bad habits (hitting chassis)
- Will need Session 7 with contact fix anyway

**Recommendation**: ⚠️ Not recommended - fix is critical for proper learning

### Option 2: Fix Contacts FIRST, Then Launch Session 6

**Pros:**
- Policy learns collision-aware behaviors from start
- Better training signal (no silent collisions)
- Avoids retraining to fix bad habits
- More complete fix set

**Cons:**
- Delays Session 6 by 1-2 hours
- Requires API investigation

**Recommendation**: ✅ **STRONGLY RECOMMENDED** - proper learning needs collision feedback

---

## 📝 Next Actions

### Immediate (30 minutes)

1. **Check Isaac Lab 2.2.0 Contact API** (15 mins)
   - Review `isaaclab/source/isaaclab/sensors/contact_sensor.py`
   - Check `Articulation` class for contact methods
   - Look at Isaac Lab examples using contact forces

2. **Test Alternative APIs** (15 mins)
   - Try `self.robot.contact_forces` (if exists)
   - Try adding explicit `ContactSensor` to scene
   - Document working approach

### Follow-up (1-2 hours)

3. **Implement Fix in env.py**
   - Update contact force retrieval code
   - Test with debug script (verify non-zero forces)
   - Commit fix

4. **Validate Fix**
   - Run debug script again
   - Confirm contact forces > 0 during collisions
   - Verify self-collision penalties trigger

### Then Launch Session 6! 🚀

With both fixes:
- ✅ Jerk penalty (base can move)
- ✅ Contact forces (collision awareness)
- 🎯 Expected: Coordinated whole-body motion with collision avoidance

---

## 📚 References

- **Debug Script**: `scripts/debug_contact_forces.py`
- **Environment Code**: `src/rl_platform/tasks/mobile_mm/env.py` (lines 960-976)
- **Code Review**: `docs/_CODE_REVIEW_VALIDATION.md` (Issue #2)
- **Fix Plan**: `docs/_FIX_IMPLEMENTATION_PLAN.md` (Phase 2)

---

**Conclusion**: The jerk penalty fix is working (base is moving!), but contact forces are broken. We should fix contact forces before Session 6 to ensure proper collision-aware learning from the start.
