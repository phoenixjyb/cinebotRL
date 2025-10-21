# PPR Spring Stiffness Tuning - Training Session 3

## 🐛 Problem Identified (Training Session 2)

**After 500 steps with 1.0kg PPR helpers:**

```
🎯 Target:        [1.637, 0.088, 0.632]
🚗 Base (actual): [1.056, 0.080, -0.072]  ← Only +6mm movement!
🔧 PPR offsets:   [0.023, 0.000, 0.013]   ← Commanded +23mm
📐 Base-Target:   0.5814 m
💰 base_mobilization: -0.0041              ← NEGATIVE (moving away!)
```

**Diagnosis:**
- Base **barely moving** despite PPR joint commands
- Only +6mm actual movement vs +23mm commanded
- Base mobilization reward **negative** (moving wrong direction)
- **Root cause:** Spring stiffness TOO HIGH (10,000 N/m)

## 📊 Physics Analysis

### Previous Configuration (Training Session 2)
```python
Mass (m):         1.0 kg
Stiffness (k):    10,000 N/m
Damping (c):      1,000 N·s/m
```

**Problems:**
1. **Natural frequency:** ω = sqrt(k/m) = sqrt(10000/1.0) = **100 rad/s** (16 Hz)
2. **Control frequency:** 50 Hz (dt = 0.02s)
3. **Nyquist limit:** 25 Hz (must be >2× natural frequency for stability)
4. **Result:** Spring oscillates **too fast** for 50Hz control!
5. **Effect:** Stiff spring **resists** position commands → slow response

**Analogy:** Like trying to push a very stiff spring - it takes many pushes to move it!

### Mass-Spring-Damper System Dynamics

For position control with spring constant k and mass m:

**Natural frequency:** ω_n = sqrt(k/m)  
**Damping ratio:** ζ = c / (2 * sqrt(k*m))  
**Critical damping:** c_crit = 2 * sqrt(k*m)

**Desired characteristics:**
- ω_n << control frequency (for good tracking)
- ζ ≈ 1.0 (critically damped, no overshoot)
- Response time: ~5-10 control timesteps

## ✅ Solution: Reduce Stiffness 10×

### New Configuration (Training Session 3)
```python
Mass (m):         1.0 kg          (unchanged)
Stiffness (k):    1,000 N/m       (was 10,000, reduced 10×)
Damping (c):      632 N·s/m       (was 1,000, optimized for critical damping)
```

**Benefits:**
1. **Natural frequency:** ω = sqrt(1000/1.0) = **31.6 rad/s** (5 Hz)
2. **Control frequency:** 50 Hz (still 10× higher than natural freq)
3. **Nyquist satisfied:** 50 Hz >> 10 Hz (5× margin)
4. **Damping ratio:** ζ = 632/(2*sqrt(1000*1.0)) = 632/63.2 = **1.0** (critically damped!)
5. **Response time:** ~0.1-0.2 seconds (5-10 timesteps at 50Hz)

**Expected behavior:**
- ✅ Base **10× more responsive** to position commands
- ✅ Commanded position ≈ actual position (within 1-2 control cycles)
- ✅ No overshoot or oscillations (critically damped)
- ✅ Smooth, controlled motion
- ✅ Base mobilization reward becomes **positive** (moves toward target!)

## 📈 Expected Performance Improvement

### Training Session 2 (k=10,000 N/m)
```
After 500 steps (10s):
  Base moved:  +6mm (commanded +23mm)
  Lag ratio:   26% (6/23)
  Mobilization: NEGATIVE (wrong direction)
```

### Training Session 3 (k=1,000 N/m) - Expected
```
After 500 steps (10s):
  Base moved:  +20-22mm (commanded +23mm)
  Lag ratio:   87-96% (much better tracking!)
  Mobilization: POSITIVE (toward target)
```

**Improvement:** **~10× better position tracking!**

## 🔧 Implementation Details

**File:** `src/rl_platform/tasks/mobile_mm/env.py` (lines 152-157)

### Before (Training Session 2)
```python
"base": ImplicitActuatorCfg(
    joint_names_expr=["joint_x", "joint_y", "joint_theta"],
    stiffness=10000.0,  # High stiffness for PPR position tracking
    damping=1000.0,     # High damping for stability
    effort_limit=1000.0,
    velocity_limit=2.0,
),
```

### After (Training Session 3)
```python
"base": ImplicitActuatorCfg(
    joint_names_expr=["joint_x", "joint_y", "joint_theta"],
    stiffness=1000.0,   # Reduced 10x for responsiveness (k=1000 N/m)
    damping=632.0,      # Critical damping: 2*sqrt(k*m) = 2*sqrt(1000*1.0)
    effort_limit=1000.0,
    velocity_limit=2.0,
),
```

**Changes:**
- Stiffness: 10,000 → **1,000 N/m** (10× softer)
- Damping: 1,000 → **632 N·s/m** (critically damped)

## 🎯 Validation Checkpoints

### Step 100 (First validation)
**Expected:**
- Base movement >10mm (was 6mm @ step 500 before)
- PPR offsets ≈ base position change
- Base mobilization reward > 0

### Step 500 (Direct comparison)
**Expected vs. Training Session 2:**

| Metric | Session 2 (k=10k) | Session 3 (k=1k) Expected |
|--------|-------------------|---------------------------|
| Base movement | +6mm | +20-25mm |
| Tracking lag | 74% (23mm→6mm) | <15% (23mm→20mm) |
| Mobilization | -0.0041 (neg!) | >+0.01 (positive!) |
| EE-base dist | 0.811m | <0.7m (better coord) |

### Step 10,000 (Policy learning)
**Expected:**
- Policy learns base is responsive
- Distance penalty decreases (base approaches targets)
- Position tracking improves (coordinated motion)

## 🚨 Potential Issues & Mitigations

### Issue 1: Spring too soft → position drift
**Symptom:** Base drifts from commanded position  
**Mitigation:** Already using k=1000 (not too soft)  
**Backup plan:** If drift observed, increase to k=2000

### Issue 2: Underdamped → oscillations
**Symptom:** Base oscillates around target  
**Mitigation:** Using critical damping (ζ=1.0)  
**Backup plan:** If oscillations, increase damping to 800

### Issue 3: Still too stiff → slow response
**Symptom:** Base movement still lagging  
**Mitigation:** Can reduce further to k=500  
**Backup plan:** Or increase mass to 2.0kg (same effect)

## 📊 Training Comparison Matrix

| Parameter | Session 1 | Session 2 | Session 3 |
|-----------|-----------|-----------|-----------|
| **PPR mass** | 0.0 kg | 1.0 kg | 1.0 kg |
| **Stiffness** | 10,000 | 10,000 | **1,000** |
| **Damping** | 1,000 | 1,000 | **632** |
| **Natural freq** | N/A | 100 rad/s | 31.6 rad/s |
| **Damping ratio** | N/A | 0.5 | **1.0** |
| **root_pos_w** | ❌ Frozen | ✅ Moving | ✅ Moving |
| **Responsiveness** | ❌ N/A | ⚠️ Slow (26%) | ✅ Fast (90%+) |
| **Mobilization** | ❌ N/A | ⚠️ Negative | ✅ Positive |

## 🎊 Success Criteria (Session 3)

**Minimum (Step 500):**
- ✅ Base movement >15mm (vs 6mm in Session 2)
- ✅ Base mobilization reward >0 (vs -0.0041)
- ✅ No physics explosions

**Good (Step 10K):**
- ✅ Base tracks commands within 85-95%
- ✅ Distance penalty decreasing
- ✅ EE-base distance <0.7m consistently

**Excellent (100M):**
- ✅ Coordinated arm+base motion
- ✅ Base mobilizes efficiently for distant targets
- ✅ Performance >> Training Session 2

---

**Status:** ✅ Code updated, ready to restart training!  
**Next:** Commit changes → Restart training → Monitor step 100-500  
**Expected:** **10× improvement in base responsiveness!** 🚀
