# Systematic Investigation Plan

## Issue Observed

During evaluation visualization, the robot arm moves but **chassis remains completely frozen** - no movement at all.

## Investigation Questions

We need to systematically answer these questions WITHOUT jumping to conclusions:

### 1. Trajectory Loading

**Question:** Did the training actually see the 1,038 recorded trajectories?

**How to verify:**
```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\verify_trajectory_loading.py
```

**What to check:**
- Is trajectory_manager of type `MultiRecordedTrajectoryManager`?
- How many trajectories are loaded? (Should be 1,038 or 519)
- Do target positions vary significantly between resets?
- Do targets require >1m reach (implying chassis movement needed)?

**Possible findings:**
- ❌ Only 1 trajectory loaded → Training used simple circle/line
- ❌ Trajectories don't move much → Wrong trajectory type used
- ✅ 1,038 trajectories loaded → System working

---

### 2. Policy Chassis Actions

**Question:** Is the trained policy actually outputting non-zero chassis commands?

**How to verify:**
```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\investigate_chassis.py `
    --checkpoint C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --steps 100
```

**What to check:**
- Are chassis actions (vx, wz) non-zero?
- Do they vary over time or stay at zero?
- What's the mean and std of chassis actions?

**Possible findings:**
- ❌ Actions always near zero → Policy learned it doesn't need to move base
- ⚠️ Actions vary but small → Policy uncertain about base usage
- ✅ Actions significant → Policy tries to use base

---

### 3. Chassis Joint Response

**Question:** Are chassis joints actually responding to commands?

**How to verify:** Same script as above, but check:

**What to check:**
- Do chassis joint positions change when actions are non-zero?
- Do chassis velocities reflect commanded velocities?
- Is there a correlation between commands and movement?

**Possible findings:**
- ❌ Commands sent but no movement → Actuator problem
- ❌ Joint limits prevent movement → Configuration issue
- ✅ Commands cause movement → Actuators working

---

### 4. Chassis Observations

**Question:** Did the policy receive chassis state as input during training?

**How to verify:** Check observation space

**Where to look:**
- `src/rl_platform/tasks/mobile_mm/env.py` - `_get_observations()` function
- Training logs - observation dimension should be 70 (includes base state)

**What should be included:**
- Base position (x, y, theta)
- Base velocity (vx, vy, wz)
- Base state normalized

**Possible findings:**
- ❌ Chassis state missing from observations → Policy blind to base
- ✅ Chassis state included → Policy aware of base

---

### 5. Training Trajectory Distribution

**Question:** Did training trajectories actually require chassis movement?

**How to verify:**
```powershell
# Check the analysis we already did
Get-Content "C:\Users\yanbo\wSpace\cinebotRL\chassis_required_trajectories.txt" | Measure-Object -Line
```

**What to check:**
- How many of 1,038 trajectories flagged as requiring chassis?
- Were these actually used in training?
- What's the distribution?

**Known facts:**
- 519 trajectories identified as requiring chassis movement
- Analysis exists in `trajectory_analysis_results.csv`

---

### 6. Self-Collision Impact

**Question:** How does broken self-collision detection affect chassis use?

**Theory:** If self-collision isn't detected, policy might:
- Move arm through chassis instead of moving chassis away
- Learned unsafe shortcuts

**How this relates:**
- If trajectories require base movement BUT policy doesn't use it
- → Policy found alternative (possibly unsafe) arm-only solutions

---

## Investigation Scripts Created

1. **verify_trajectory_loading.py**
   - Checks what trajectories are actually loaded
   - Verifies multi-trajectory system is working
   - Samples trajectory targets to see movement requirements

2. **investigate_chassis.py**
   - Monitors chassis actions from policy
   - Tracks actual chassis movement
   - Correlates commands with movement
   - Provides diagnostic conclusion

3. **diagnose_rewards.py** (already exists)
   - Shows reward component breakdown
   - Identifies which penalties are exploding

---

## Investigation Steps (In Order)

### Step 1: Verify Trajectory Loading (5 min)
```powershell
& "I:\isaaclab\isaaclab.bat" -p scripts\verify_trajectory_loading.py
```

**Critical question:** Are 1,038 trajectories actually being loaded?

---

### Step 2: Check Policy Chassis Actions (5 min)
```powershell
& "I:\isaaclab\isaaclab.bat" -p scripts\investigate_chassis.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip `
    --steps 100
```

**Critical question:** Is policy outputting non-zero chassis commands?

---

### Step 3: Verify Observations Include Chassis State (2 min)
```powershell
# Check observation function
grep -n "base_pos\|base_vel\|base_ang_vel" src\rl_platform\tasks\mobile_mm\env.py
```

**Critical question:** Does policy see chassis state?

---

### Step 4: Check Training Logs for Trajectory Info (2 min)
```powershell
# Check if training logs mention trajectory loading
Select-String "trajectory" logs\sb3\mobilemmtrackee_v0\20251018_001233\PPO_1\*.log -SimpleMatch
```

---

## Decision Tree

```
START: Chassis doesn't move during evaluation

Q1: Are 1,038 trajectories loaded?
├─ NO → Root cause: Training used wrong trajectories
│  └─ Solution: Fix trajectory loading, retrain
│
└─ YES → Continue to Q2

Q2: Does policy output non-zero chassis actions?
├─ NO → Root cause: Policy learned it doesn't need base
│  ├─ Check Q3: Were trajectories reachable without base?
│  │  └─ YES → Training trajectories too easy
│  └─ Check Q4: Was chassis state in observations?
│     └─ NO → Policy was blind to base
│
└─ YES → Continue to Q3

Q3: Does chassis actually move when commanded?
├─ NO → Root cause: Actuator/configuration problem
│  └─ Check: Joint limits, stiffness, damping
│
└─ YES → SUCCESS! System working as expected
```

---

## Running the Complete Investigation

```powershell
# 1. Verify trajectories
Write-Host "`n=== STEP 1: Trajectory Loading ===" -ForegroundColor Cyan
& "I:\isaaclab\isaaclab.bat" -p scripts\verify_trajectory_loading.py

# 2. Investigate chassis
Write-Host "`n=== STEP 2: Chassis Movement ===" -ForegroundColor Cyan
& "I:\isaaclab\isaaclab.bat" -p scripts\investigate_chassis.py `
    --checkpoint logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip

# 3. Check observations (manual inspection)
Write-Host "`n=== STEP 3: Check Observations ===" -ForegroundColor Cyan
code src\rl_platform\tasks\mobile_mm\env.py:600
```

---

## What We're NOT Assuming

- ❌ Not assuming contact forces are the root cause of chassis freeze
- ❌ Not assuming training was bad
- ❌ Not assuming trajectories weren't loaded
- ❌ Not assuming actuators don't work

## What We're Doing

- ✅ Systematically checking each component
- ✅ Collecting data before drawing conclusions
- ✅ Creating reproducible diagnostics
- ✅ Building evidence-based understanding

---

**Next:** Run the investigation scripts and document findings.

**Status:** Investigation scripts created, ready to run.
