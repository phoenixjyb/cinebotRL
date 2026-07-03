# Training Sessions Master Log

**Project:** CinebotRL - Mobile Manipulator End-Effector Tracking  
**Repository:** phoenixjyb/cinebotRL  
**Branch:** train-windows  
**Last Updated:** November 1, 2025 17:00 +0800

---

## Current Codex Loop: Proto2 Stage1 Recovery Aux Policy - 2026-07-03

**Status:** Not deployable yet. Best mean unreachable improved to `25.1410%`, but target is `<15%` and weighted candidates worsened tail risk.

**Loop log:** `docs/03_training/STAGE1_RECOVERY_AUX_POLICY_LOOP_LOG_20260703.md`

**Current promotion guidance:**
- Conservative best/risk-balanced: `stage1_recovery_auxloss_vxvy_noassist_hookstart_40k_seed20260703_20260703_1606`
- Best mean/workspace but worse tail: `stage1_recovery_auxloss_round4_weighted_noassist_40k_seed20260703_20260703_1621`
- Do not promote any policy as deployable until raw eval reaches `<15%` unreachable and obstacle unsafe/collision remain `0`.

**Key lesson:** do not continue callback-only aux tuning blindly. Next stage should use integrated PPO minibatch auxiliary loss, reward/environment tail shaping, or stratified failure analysis before more training.

---

## 🎉 **LATEST: Session 8f - BEST RESULTS!**

**Status:** ✅ **COMPLETE - BEST PERFORMING SESSION**  
**Goal:** Apply playbook fixes (atomic state write, distance gating, heading cue)  
**Duration:** Oct 31 13:35 → Nov 1 01:44 (~12.2 hours)  
**Timesteps:** 100,663,296 (100M)  
**Evaluation:** Nov 1 15:15 (200 episodes, 64 parallel envs)

### Session 8f Results Summary

**🏆 Performance Records:**
- ✅ **Best position accuracy:** 307.8cm mean, 298.7cm median
- ✅ **Best orientation:** 46.5° mean, 40.7° median  
- ✅ **Highest reward:** -126k (vs -177k in 8d, -259k in 8e)
- ✅ **Mobilization working:** 0.32 (vs 0.15 in 8e)
- ⚠️ **Reachability bonus:** 0.64 (still low, but tracking works!)

**Implemented Fixes (from playbook):**
1. ✅ Atomic root state write (13-element tensor)
2. ✅ Distance-gated penalties (>0.55m: OFF, <0.55m: ON)
3. ✅ Heading cue observations (+2 dims: sin/cos yaw error)
4. ✅ Two-zone linear reachability (0.35-0.55m plateau)

**Remaining Issue:**
- ⚠️ Workspace distance still drifts (0.42m → 0.60m over training)

**See:** `docs/training_sessions/SESSION_8F_EVALUATION.md` for full analysis

### Quick Start (Session 8g - Next)
```powershell
# Stronger reachability gravity or workspace distance reward
.\scripts\launch_session_8g.ps1  # TBD: implement recommendations
```

---

## 🚀 **NEXT: Session 8g (Planned)**

**Goal:** Fix persistent workspace distance drift  
**Options:**
1. Increase reachability_maintenance_reward (40 → 80)
2. Add explicit workspace_distance_reward component
3. Progressive reachability weight schedule (40 → 100)
4. Success-based curriculum learning

---

## 📊 **Session 8f - Detailed Results**

**Date:** October 31 - November 1, 2025  
**Duration:** Oct 31 13:35 → Nov 1 01:44 (~12.2 hours)  
**Timesteps:** 100,663,296 (100M)  
**Environments:** 16,384 parallel (RTX 3090, Isaac Lab 2.2.0)  
**Evaluation:** Nov 1 15:15 (200 episodes, 64 parallel envs, deterministic)

### Session 8f Results - THE BEST SO FAR! 🏆

**Performance Metrics:**
```
Position Error:    307.8cm mean, 298.7cm median  ⭐ BEST
Orientation Error: 46.5° mean, 40.7° median     ⭐ BEST
Mean Reward:       -126,482                      ⭐ BEST
Episode Length:    Mean 399.2 steps

Environment Health Distribution:
  Excellent (<100cm):      8.0%
  Good (100-250cm):       33.5%
  Acceptable (250-400cm): 41.0%
  Poor (400-600cm):       14.0%
  Critical (>600cm):       3.5%
```

**Reward Component Breakdown:**
```
position_tracking:               21.83 ± 46.90
orientation_tracking:           144.72 ± 24.27
reachability_bonus:               0.64 ± 2.18   ⚠️ Low but OK
reachability_distance_penalty:  231.77 ± 217.31  (Much better than 8e!)
inner_margin_penalty:             0.13 ± 0.43
base_mobilization:                0.32 ± 0.52   ✅ Working!
base_target_alignment:            0.36 ± 0.33   ✅ Much improved!
```

### Session 8f vs 8d vs 8e Comparison

| Metric | Session 8d | Session 8e | Session 8f | vs 8d | vs 8e |
|--------|------------|------------|------------|-------|-------|
| **Position (cm)** | 311.0 | 349.4 | **307.8** | -1.0% ✅ | -11.9% ✅ |
| **Orientation (°)** | 47.4 | 48.5 | **46.5** | -1.9% ✅ | -4.1% ✅ |
| **Reward** | -177k | -259k | **-126k** | +28.8% ✅ | +51.4% ✅ |
| **Reachability Bonus** | 7.06 | 0.79 | 0.64 | -91% ❌ | -19% ⚠️ |
| **Distance Penalty** | 360 | 529 | **232** | -35.6% ✅ | -56.2% ✅ |
| **Mobilization** | N/A | 0.15 | **0.32** | N/A | +113% ✅ |
| **Alignment** | N/A | 0.056 | **0.36** | N/A | +543% ✅ |

### What Worked (Session 8f Implementation)

**1. ✅ Atomic Root State Write** (`env.py` lines ~1160-1185)
- **Problem:** Sequential velocity→pose writes were conflicting
- **Fix:** Single `write_root_state_to_sim()` with 13-element tensor
- **Impact:** Base responds more smoothly, no control fighting

**2. ✅ Distance-Gated Penalties** (`rewards.py` lines ~960-975)
- **Problem:** Stability penalties fought mobilization even when far
- **Fix:** `sigmoid((0.55 - distance) * 10.0)` gates penalties OFF when >0.55m
- **Impact:** Mobilization +113%, alignment +543% vs Session 8e

**3. ✅ Heading Cue Observations** (`observations.py` lines ~78-90)
- **Problem:** Policy didn't know "which way to turn"
- **Fix:** Added sin/cos(base→target yaw error) = +2 obs dims
- **Impact:** Orientation median improved 13.8% (47.2° → 40.7°)

**4. ⚠️ Two-Zone Linear Reachability** (`rewards.py` lines ~95-145)
- **Problem:** Bell-shaped (8e) was too brittle, collapsed
- **Fix:** Linear zones: approach (0.35-0.45m), plateau (0.45-0.55m), decay (0.55-0.9m)
- **Impact:** Better than 8e, but reachability bonus still low (0.64)

### What Still Needs Fixing

**Workspace Distance Drift:**
```
Training Progress:
  10M: 0.416m  ⚠️  Too close
  20M: 0.450m  ✅  Good!
  30M: 0.481m  ✅  Good!
  40M: 0.489m  ✅  Good!
  50M: 0.556m  ⚠️  Starting to drift
  60M: 0.554m  ⚠️  Drifting
  70M: 0.625m  🚨  Too far!
  80M: 0.568m  ⚠️  Fluctuating
  90M: 0.600m  ⚠️  Still too far
```

**Analysis:**
- Distance gating helped early (20-40M stable)
- But drift occurred after 50M
- Policy learned: "Stay farther = avoid penalties"
- Two-zone linear wasn't strong enough

**Reachability Bonus Still Low:**
- Target: >5.0 (like Session 8d's 7.06)
- Actual: 0.64 (10% of target)
- Root cause: Workspace distance at 0.60m during eval
- Two-zone gives ~zero bonus beyond 0.55m

### Training Configuration (Session 8f)

**Architecture:**
```python
task: MobileMMTrackEE-v0
num_envs: 16384
device: cuda:0 (NVIDIA GeForce RTX 3090)
isaac_lab: 2.2.0
observation_dims: 76 total
  - Base: 51 dims (13 root + 12 joints + 13 EE + 7 error + 6 base-to-target)
  - Optional: 9 lookahead + 16 action history
action_dims: 12 (arm: 9, base: 3)
```

**PPO Hyperparameters:**
```python
learning_rate: 3e-4
n_steps: 16 (per env)
batch_size: 32768 (2048 minibatches)
n_epochs: 4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: entropy decay schedule
vf_coef: 0.5
max_grad_norm: 1.0

# Advanced features
adaptive_kl_enabled: True
kl_schedule_enabled: True
target_kl: 0.01
tf32_enabled: True
cudnn_benchmark: True
```

**Reward Weights (Session 8f):**
```python
position_tracking: 50.0
orientation_tracking: 30.0
reachability_maintenance: 40.0  ⚠️ Should increase for 8g
reachability_distance_penalty: 10.0
position_distance_penalty: 5.0
inner_margin_penalty: 100.0
base_mobilization: 0.5
base_overshoot_penalty: 5.0
base_target_alignment: 2.0

# Distance-gated (active only <0.55m):
stability_penalty: 1.0
velocity_limit_penalty: 0.1
acceleration_limit_penalty: 0.1
jerk_penalty: 0.1
```

**Observation Features (Session 8f):**
- ✅ Base state (13 dims): pos, quat, lin_vel, ang_vel
- ✅ Joint states (12 dims): positions + velocities
- ✅ End-effector (13 dims): pos, quat, lin_vel, ang_vel
- ✅ Tracking error (7 dims): position + orientation
- ✅ **NEW:** Base-to-target (6 dims): distance, heading sin/cos, relative pos
- ✅ Lookahead (9 dims): future target trajectory
- ✅ Action history (16 dims): smoothness signal

### Playbook Compliance Checklist

From `mobile_mm_training_playbook.md`:

- ✅ **Atomic Root State Write:** Implemented (env.py)
- ✅ **Distance-Gated Penalties:** Implemented (rewards.py)
- ✅ **Heading Cue Observations:** Implemented (observations.py)
- ✅ **Two-Zone Linear Reachability:** Implemented (rewards.py)
- ⚠️ **Reachability Weight:** 40 (playbook suggests higher)
- ⚠️ **Workspace Distance Control:** Still drifts (need 8g fix)

### Files Generated (Session 8f)

**Training Artifacts:**
- `logs/sb3/mobilemmtrackee_v0/20251101_013539/final_model.zip` (policy checkpoint)
- `logs/sb3/mobilemmtrackee_v0/20251101_013539/events.out.tfevents.*` (TensorBoard)

**Evaluation Results:**
- `evaluation_plots/session_8f_100M/20251101_013539/eval_summary_20251101_151551.json`
- `evaluation_plots/session_8f_100M/20251101_013539/episodes_20251101_151551.csv`
- `evaluation_plots/session_8f_100M/20251101_013539/steps_20251101_151551.csv`
- `evaluation_plots/session_8f_100M/20251101_013539/arrays_20251101_151551.npz`

**Documentation:**
- `docs/training_sessions/SESSION_8F_IMPLEMENTATION.md` (setup guide)
- `docs/training_sessions/SESSION_8F_EVALUATION.md` (this analysis)
- `scripts/launch_session_8f.ps1` (launcher)
- `scripts/check_workspace_8f.py` (monitoring tool)

### Recommendations for Session 8g

**Based on Session 8f analysis, try ONE of these:**

**Option 1: Stronger Reachability Gravity** (Recommended)
```python
reachability_maintenance_reward: 40 → 80  # Double the pull
optimal_plateau_width: ±0.05m → ±0.10m   # Wider safety margin
```

**Option 2: Explicit Workspace Distance Reward**
```python
# New component in rewards.py
workspace_distance_reward = lambda d: {
    +50 if 0.45 ≤ d ≤ 0.55,  # Large bonus in optimal zone
    +25 if 0.40 ≤ d < 0.45 or 0.55 < d ≤ 0.60,  # Approach bonus
    -100 * (d - 0.5)**2 otherwise  # Quadratic penalty
}
```

**Option 3: Progressive Weight Schedule**
```python
# Increase reachability importance over time
0-20M:   reachability_weight = 40
20M-50M: reachability_weight = 60
50M-100M: reachability_weight = 100
```

**Option 4: Success-Based Curriculum**
```python
# Only enforce strict workspace after tracking is good
if position_error < 250cm AND orientation_error < 35°:
    enforce_strict_workspace_distance()
else:
    relax_workspace_requirements()
```

### Session 8f Verdict

**Overall Grade: A-** (Best session so far!)

**Strengths:**
- ✅ Best tracking accuracy (position & orientation)
- ✅ Highest reward (+51% vs 8e, +29% vs 8d)
- ✅ Mobilization working (base moves purposefully)
- ✅ All playbook fixes validated
- ✅ Stable training (no crashes, no collapse)

**Weaknesses:**
- ⚠️ Workspace distance still drifts after 50M
- ⚠️ Reachability bonus remains low (0.64 vs target 5.0+)
- ⚠️ Policy learns to stay farther to minimize risk

**Recommendation:**
- **Use Session 8f as baseline** for future work
- **Launch Session 8g** with Option 1 (stronger reachability gravity)
- Consider progressive weight schedule if Option 1 insufficient

---

## ✅ **COMPLETED: Session 7c**

**Status:** Base movement enabled with Z-clamp fix  
**Duration:** Oct 27 18:02 → Oct 28 06:32 (~11.5 hours)  
**Timesteps:** 100,073,472 (100M)  
**Evaluation:** Oct 28 09:23 (100 episodes, 16 parallel envs)

### Session 7c Results Summary

**Major Achievements:**
- ✅ **Base MOVES!** 0.1-1.8m per episode (90x improvement vs Session 6)
- ✅ **Reward fixed:** +12,330 (vs -11.7M in Session 6)
- ✅ **Best tracking:** 0.15m (vs 0.49m in Session 6)
- ✅ **Training stable:** 100M steps, no crashes

**Critical Issues:**
- ⚠️ **Not goal-directed:** 93% targets unreachable (6% reachability)
- ⚠️ **Mean error high:** 1.01m (target: <0.30m)
- ⚠️ **Low mobilization:** 0.0-2.3 pts reward (too weak)
- ⚠️ **High variance:** CV=0.77 (inconsistent performance)

**Detailed Results:**
```
Episodes: 100 completed
Mean reward: 12,330 ± 9,483
Min/Max: -34,196 to +26,999
Episode length: 399 steps (all completed)

Environment Health Distribution:
  Excellent (<0.1m):     0%
  Good (0.1-0.3m):      19%
  Poor (0.3-2.0m):      75%
  Broken (>2.0m):        6%

Base Movement Examples:
  Env 2: 1.78m moved
  Env 4: 1.61m moved
  Env 5: 1.57m moved
  Mean: 0.78m

Base Velocity Commands:
  base_vx: -0.47 m/s (vs 0.002 in Session 6)
  base_wz: 1.08 rad/s (vs 0.01 in Session 6)
```

### Session 7c vs Session 6 Comparison

| Metric | Session 6 | Session 7c | Change |
|--------|-----------|------------|--------|
| Mean Reward | -11,715,724 | +12,330 | +11.7M ✅ |
| Base Movement | 0.02-0.20m | 0.1-1.8m | 90x ✅ |
| Best Error | 0.49m | 0.15m | 70% better ✅ |
| Mean Error | ~1.5m | 1.01m | 33% better ⚠️ |
| Reachability | N/A | 6% | Need 50%+ ❌ |
| Broken Envs | 50% | 6% | 88% fewer ✅ |

**See:** `docs/SESSION_7C_VS_SESSION_6_COMPARISON.md` for full analysis

### What Worked (Session 7c)

1. **Z-Clamp Fix (env.py lines 680-689):**
   - Prevents base "jumping"
   - Keeps Z ~0.0 stable
   - Allows smooth X, Y, theta movement
   - **This was the breakthrough!** 🎉

2. **Collision Penalty Reduction (1000→5):**
   - Restored learning signal
   - Episode rewards positive
   - Balanced reward distribution

3. **Reachability Integration:**
   - Loaded 12,646 reachable voxels
   - Provides workspace guidance
   - Penalizes unreachable targets

### What Needs Improvement (Session 7c → 7d)

1. **Base Mobilization Reward Too Low:**
   - Current: 0.0-2.3 pts (2% of total reward)
   - Issue: Not enough incentive for strategic movement
   - Solution: Increase from 150 → 250 (67% boost)

2. **No Directional Guidance:**
   - Base moves but not toward targets
   - 93% targets remain unreachable
   - Solution: Add alignment reward (10.0 weight)

3. **Distance Penalty Too Harsh:**
   - Current: 13+ pts penalty for far targets
   - Discourages exploration
   - Solution: Reduce from 5.0 → 3.0 (40% gentler)

### Training Configuration (Session 7c)
```
Environment: 4096 parallel instances
Episode length: 20s (400 steps @ 20Hz)
Trajectories: 1,038 recorded (all categories)
Network: 235K params (Actor: 118K, Critic: 117K)
Algorithm: PPO with adaptive KL
Checkpoints: 1,018 saved (every 100K steps)
Log dir: H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251027_180246
Final model: 5.84 MB
```

### Files Created (Session 7c Analysis)
- `docs/SESSION_7C_VS_SESSION_6_COMPARISON.md` - Detailed comparison
- `docs/SESSION_7D_REWARD_TUNING_PROPOSAL.md` - Fix proposal
- `docs/SESSION_7C_VISUALIZATION_GUIDE.md` - GUI observation guide
- `scripts/summarize_training.py` - Training summary tool

### Next Steps
1. ✅ Evaluation complete (100 episodes analyzed)
2. ✅ Comparison with Session 6 documented
3. ⏳ Optional: Visualize behavior (GUI, 10-15 min)
4. ⏳ Implement Session 7d reward changes (5 min)
5. ⏳ Launch Session 7d training (200M steps, ~11 hours)

---

## Session 7c Details

**Status:** Reachability-guided base planning with Z-clamp fix  
**Session:** 7c - Base Movement Enabled (Z-clamp applied)  
**Commits:** 3 total (reachability + Z-clamp + validation)

### Quick Start (Session 7c - COMPLETED)
```powershell
# Full training: 4096 environments (100M steps)
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 4096 --n_steps 128 --batch_size 1024 --n_epochs 10 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --clip_range 0.2 --gamma 0.99 --gae_lambda 0.95 --trajectory_type multi_recorded --use_all_trajectories --save_freq 100000 --headless
```

### What's New (Session 7c)
✅ **NEW:** Reachability map (12,646 voxels, 5cm resolution, ARM base frame)  
✅ **NEW:** Z-clamp fix (lines 680-689) - prevents base jumping  
✅ **NEW:** Base movement actually working (0.1-1.8m)  
✅ Coordinate transformation: World → Arm base frame  
✅ All Session 6 fixes still active (jerk penalty, contact sensor, shape fix)

### Why Session 7c Was Needed
🎯 **Fix Base Z-axis Instability:**
- Session 7a/7b: Base Z drifted/jumped (not physical)
- Z-clamp: Locks base Z ~0.0 while allowing X, Y, theta
- Result: Stable base movement on ground plane

🎯 **Validate Reachability Integration:**
- Reachability map loaded successfully
- KD-tree queries working
- Workspace-aware rewards active

---

## Session 6 (Completed)

**Status:** Critical frozen base fixes - jerk penalty + contact sensor  
**Session:** 6 - Frozen Base Fix (3 CRITICAL fixes)  
**Commits:** 10 total (jerk penalty + contact sensor + shape fix + validation)  
**Latest Commit:** `c59c72b` - Fix KeyError in debug script scene access

### Quick Start
```powershell
# 8192 Environments (~9 hours to 100M steps)
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 8192 --n_steps 64 --batch_size 512 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless
```

### What's Fixed (Session 6)
✅ **CRITICAL FIX #1:** Jerk penalty 5.0 → 50.0 m/s³ (THE KEY FIX)  
   - Old: -451 penalty + 150 bonus = -301 NET (base FROZEN)
   - New: -125 penalty + 150 bonus = +25 NET (can move!)
✅ **CRITICAL FIX #2:** ContactSensor added (952.51 N validated!)  
   - Monitors arm-chassis collisions (self-collision detection)
   - Filter: `abstract_chassis_link` with `left_arm.*` bodies
✅ **CRITICAL FIX #3:** prev_joint_vel shape 6 → 9 columns  
   - Fixed observation space mismatch
✅ USD limits verified: ±360° (±6.28 rad) - base CAN rotate
✅ Debug validation complete: Contact forces working

### Why Session 6 is Needed
❌ Session 5b: Base still FROZEN despite observation space fix!
- Root cause: Jerk penalty (5 m/s³) too harsh
- Normal 0→0.25 m/s movement = -301 NET penalty
- Base learning disabled by catastrophic jerk penalties
- No self-collision detection (contact forces reading 0.0 N)

### Quick Start
```powershell
# Option A: 4096 Environments (~12 hours)
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 4096 --n_steps 64 --batch_size 256 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless

# Option B: 8192 Environments (~9 hours, may be tight on VRAM)
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 8192 --n_steps 64 --batch_size 512 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless
```

### What's Fixed (Session 5b)
✅ **CRITICAL:** Capped base_mobilization progress to max 0.2m per step  
✅ **NEW:** Added excessive_base_movement_penalty (10.0 × excess beyond 0.1m)  
✅ **REBALANCED:** Increased target_distance_penalty from 3.0 to 5.0  
✅ Session 5 observation space fix (still active)  
✅ All unbounded reward exploits eliminated

### Why Session 5b is Needed
❌ Session 5 CATASTROPHIC FAILURE at 20M steps:
- Unbounded reward → policy learned "move base as far as possible = infinite rewards"
- 51.5%→63.5% environments broken (>2m error)
- Base moving wildly: 5-10 meters!
- Mean error DOUBLED: 0.877m → 2.242m
- Training death spiral, had to abort

### Expected Outcome (Session 5b)
🎯 Base movements: 5-15cm per step (NOT 5-10 meters!)  
🎯 base_mobilization_reward: -5 to +30 range (BOUNDED!)  
🎯 Mean tracking error < 0.5m after 100M steps  
🎯 <5% broken environments throughout training

---

This document maintains a chronological record of all training sessions, including git commits, configurations, commands, and outcomes.

---

## 📋 Quick Navigation

- [Session 1](#session-1-initial-training) - Initial baseline
- [Session 2](#session-2) - [TBD]
- [Session 3](#session-3) - [TBD]
- [Session 4a](#session-4a) - [TBD]
- [Session 4b](#session-4b-25m-steps-no-base-movement) - 25.7M steps, no base movement (stopped)
- [Session 5](#session-5-base-mobilization-fix) - Observation space fix, ran to 20M steps, CATASTROPHIC FAILURE
- [Session 5b](#session-5b-unbounded-reward-fix) - Reward capping fixes
- [Session 6](#session-6-fix-frozen-base-3-critical-fixes) - **COMPLETED** - Jerk penalty + contact sensor (mixed results)
- [Session 7](#session-7-reachability-guided-base-planning) - **CURRENT** - Intelligent base navigation with FK workspace map

---

## Session 7: Reachability-Guided Base Planning

**Status:** 🚀 READY TO LAUNCH  
**Date:** October 26, 2025  
**Objective:** Guide base movement intelligently using pre-computed reachability workspace

### Git Commits

**Core Implementation:**
1. `37d23d5` - Add reachability map loader with HDF5 support
   - Load MATLAB v7.3 (.mat) files with h5py
   - Fast KD-tree queries for reachability checks (O(log 12646))
   - Return best arm joint configs for reachable positions
   - Compute distance to workspace boundary
   - World-to-arm frame coordinate transformation
   - Tested and verified with 12,646 reachable voxels

2. `d0f44cc` - Add reachability-guided reward integration
   - Integrated reachability map queries into `_get_rewards()`
   - Two-stage reward strategy: reachable vs unreachable
   - Transform body velocity to world frame
   - Compute alignment: `dot(vel_world, direction_to_target)`
   - Reward positive alignment + speed bonus
   - Log reachability stats every 100 steps
   - Track reward components in extras

**Additional Files:**
- `src/rl_platform/utils/reachability_map.py` - ReachabilityMap class (346 lines)
- `scripts/inspect_mat_file.py` - HDF5 inspection utility
- `matlab/reach_map_mobile_mm_arm_only.mat` - Pre-computed workspace (12,646 voxels)

### Changes Applied

**NEW: Reachability Map System**
- **Data:** 12,646 reachable voxels from 35,840 total (35.3% coverage)
- **Resolution:** 5cm voxel size
- **Frame:** ARM BASE FRAME (shoulder frame, not ground)
- **Coverage:** [-0.8→0.8, -1.0→1.0, -0.6→0.8] meters around shoulder
- **Storage:** Each voxel stores best 6-joint arm configuration (qExample)

**NEW: Two-Stage Reward Strategy**

**Case 1: Target IS Reachable (from current base position)**
```python
# Bonus for being in a good base position
reachability_bonus = 0.5

# Policy focuses on arm tracking accuracy
# No need to move base, save energy
```

**Case 2: Target NOT Reachable (need to move base)**
```python
# Compute direction to target in world X-Y plane
direction = target_xy - base_xy
direction_normalized = direction / (distance + 1e-6)

# Transform base velocity from body frame to world frame
base_vx_world = base_vx_body * cos(base_theta)
base_vy_world = base_vx_body * sin(base_theta)

# Reward alignment with desired direction
alignment = dot([vx_world, vy_world], direction_normalized)
base_direction_reward = 1.0 * clamp(alignment, min=0.0)

# Speed bonus: move faster in right direction
speed_bonus = 0.3 * speed * clamp(alignment, min=0.0)
```

**How Reachability Check Works:**
1. Get target EE position in world frame
2. Transform to arm base frame: `world_to_arm_frame(target, base_pose)`
3. KD-tree query: Find nearest of 12,646 reachable voxels
4. Check distance: `is_reachable = (distance < 0.1m)`
5. Split rewards based on boolean mask

### Training Configuration

**Environment:**
- Task: MobileMMTrackEE-v0
- Trajectory Type: multi_recorded (all 1,038 trajectories)
- Rendering: Headless
- Number of Environments: 4096 (recommended), or 64 for testing
- Robot: theta_before_x USD (joint order: theta, x, y, arm1-6)

**Hyperparameters:**
- Total Timesteps: 100,000,000
- n_steps: 128
- Batch Size: 1024
- Learning Rate: 3e-4
- Entropy Coefficient: 0.001 (decay to 1e-4)
- KL Schedule: warmup 0.25, main 0.15, finetune 0.07

**All Session 6 Fixes Still Active:**
- ✅ Jerk penalty: 50.0 m/s³ (allows base movement)
- ✅ ContactSensor: Monitors self-collisions
- ✅ Shape fix: prev_joint_vel 9 columns

### Launch Command

**Quick Test (64 envs, ~5 min to verify reachability system works):**
```powershell
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 64 `
    --n_steps 128 `
    --batch_size 64 `
    --total_timesteps 1000000 `
    --learning_rate 3e-4 `
    --ent_coef 0.001 `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --headless
```

**Full Training (4096 envs, ~12 hours to 100M):**
```powershell
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --n_steps 128 `
    --batch_size 1024 `
    --n_epochs 10 `
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
    --clip_range 0.2 `
    --gamma 0.99 `
    --gae_lambda 0.95 `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --save_freq 100000 `
    --headless
```

**Parameters Explained:**
- `--n_steps 128`: Rollout length (128 × 4096 = 524K timesteps/iteration)
- `--batch_size 1024`: Minibatch size for updates
- `--n_epochs 10`: PPO update epochs per rollout
- `--ent_coef 0.001`: Initial entropy (decay to 0.0001 after 50M steps)
- `--enable_kl_schedule`: Three-phase KL schedule for stability
  - Warmup (0-10M): 0.25 (exploration)
  - Main (10M-80M): 0.15 (balanced)
  - Finetune (80M-100M): 0.07 (refinement)
- `--save_freq 100000`: Checkpoint every 100K steps (every ~2.4M timesteps)

### Expected Results

**Early Phase (0-10M steps):**
- Reachability stats should show varying reachable/unreachable ratios
- Base direction alignment should increase from ~0 to >0.5
- Base should start moving toward targets when unreachable

**Mid Phase (10M-50M steps):**
- Policy learns: "Move base FIRST when target unreachable"
- Fewer wasted arm movements when base is in wrong position
- Mean tracking error should drop faster than Session 6

**Late Phase (50M-100M steps):**
- Smooth coordination: base navigates, then arm tracks
- Reachability bonus guides base to good positions
- Expected final error: <0.3m (vs 0.5m in Session 6)

**Key Metrics to Monitor:**
- `reachability_bonus`: Average value (higher = more good positions)
- `base_direction_reward`: Average alignment (should increase over time)
- Reachability stats: % reachable vs unreachable per step
- Base→target distance when unreachable (should decrease)

### Validation Before Launch

✅ **Environment Test Passed:**
```bash
Test: 4 envs × 10 steps
Result: Successful
Reachability: Detected 4/4 envs with unreachable targets
Alignment: 0.000 (no base movement with zero actions)
Distance: 0.500m average
Reward: 21.62 → 24.25 (increasing over steps)
```

✅ **Code Commits Clean:**
- 2 commits: loader + integration
- 13 files changed, 453 insertions
- USD files organized correctly
- All tests passing

---

## Session 6: Fix Frozen Base (COMPLETED)

**Status:** 🚀 READY TO LAUNCH  
**Date:** October 21, 2025  
**Objective:** Fix critical observation space bug preventing base mobilization learning

### Git Commits

#### Commit 1: URDF Physics Fixes
```
Commit: 00060bc492efe10a028fba0fa5a01570fa80afa6
Date: October 21, 2025 00:51:41 +0800
Message: Fix critical URDF physics issues for base mobility
```

**Changes:**
- Set PPR helper links (base_link_x/y) mass to 0.0 kg (was 0.001kg)
  - Eliminates numerical stiffness from micro-bodies + 10kN/m springs
  - PhysX computes composite mass from articulation tree (more stable)
- Fix joint_theta limits: -inf/+inf → ±6.283185 rad (±2π)
  - USD converter collapses infinite limits to zero-width (locked joint)
  - Finite limits allow base rotation within ±360°+ range
- Regenerated USD from corrected URDF with Isaac Sim 5.0
  - All PPR joints set to Position control (matches env.py)
  - Mesh scale 0.001 (mm→m conversion)
  - Moveable base enabled

**URDF Physics Summary (5 fixes total):**
1. Base mass: 0.0 → 20.0 kg (movable, not static)
2. Chassis mass: 50.96 → 30.96 kg (fixed duplication, 51kg total)
3. Base inertia: (1,1,1) → (0.833, 0.833, 1.2) (realistic)
4. PPR helpers: 0.001 → 0.0 kg (stable simulation)
5. joint_theta: -inf/+inf → ±6.28 rad (rotatable)

**Files Modified:**
- URDF files (PPR helper masses, joint limits)
- USD asset regeneration
- Documentation: `docs/urdf_physics_analysis.md`, `docs/URDF_PHYSICS_ISSUES_REMAINING.md`, `docs/PPR_CONTROL_ARCHITECTURE.md`, `URDF_FIXES_APPLIED.md`

---

#### Commit 2: PPR Helper Mass Fix
```
Commit: c59cda8b45306321d637c1ba352fb94097821f36
Date: October 21, 2025 09:07:55 +0800
Message: Fix PPR helper link masses: 0.0 → 1.0 kg for real base mobility
```

**Changes:**
- Changed PPR helper links mass from 0.0 kg → 1.0 kg
- Ensures PhysX propagates position updates through articulation tree
- Required for `root_pos_w` to update when PPR joints move

**Files Modified:**
- URDF files (PPR helper link masses)

---

#### Commit 3: USD Regeneration with 1.0kg Masses
```
Commit: 29d66686315649cdad5547c1bf79326bc845e219
Date: October 21, 2025 09:23:03 +0800
Message: Regenerate USD with 1.0kg PPR helper masses for real base mobility
```

**Changes:**
- Imported corrected URDF in Isaac Sim 5.0
- PPR helper links now have 1.0 kg mass (was 0.0 kg)
- All joints set to Position control
- Mesh scale 0.001 (mm to m)
- Moveable base enabled

**Files Modified:**
- `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`
- `assets_own/usd/configuration/` (if present)

---

#### Commit 4: Repository Reorganization
```
Commit: eca70435ba2970d66176f452f8379833da96eaed
Date: October 21, 2025 18:51:12 +0800
Message: Reorganize repository structure for better maintainability
```

**Changes:**
- Created organized directory structure: `docs/{legacy/, training_sessions/, urdf_fixes/}`, `logs/evaluation/`
- Moved 18 scattered files into appropriate directories
- Created `DIRECTORY_STRUCTURE.md` navigation guide
- Reduced root clutter by 45% (44 items → 24 items)

**Files Modified:**
- 28 files changed (moved to organized directories)
- Created `DIRECTORY_STRUCTURE.md`

---

#### Commit 5: Critical Observation Space Fix
```
Commit: 7ff0e8de8282bdd7457299cd999d3269f3169dc1
Date: October 21, 2025 19:12:01 +0800
Message: Fix CRITICAL observation space bug preventing base mobilization learning
```

**ROOT CAUSE IDENTIFIED:**
- 83.8% of trajectory waypoints OUT OF REACH (>0.6m from base)
- Mean distance: 2.3m, Max distance: 4.0m (trajectory REQUIRES base movement!)
- Policy had 0.958 explained variance but zero base movement after 25.7M steps
- Policy learned "I will get low rewards" but couldn't learn "HOW to fix it"

**CRITICAL FIXES:**

**Fix #1: Observation Space Enhancement (+4 dims)**
- `src/rl_platform/tasks/mobile_mm/observations.py`:
  - Added `base_to_target_xy` [2 dims]: XY offset from base to target
  - Added `base_to_target_dist` [1 dim]: Euclidean distance to target
  - Added `out_of_reach_flag` [1 dim]: Binary signal when target >0.6m from base
  - Total: 59 dims (was 55 dims with lookahead)

**Fix #2: Reward Rebalancing**
- `src/rl_platform/tasks/mobile_mm/config.py`:
  - `base_progress_reward`: 50.0 → 150.0 (3x stronger!)
  - `target_distance_penalty`: 10.0 → 3.0 (less harsh during movement)
  - `action_magnitude`: 0.01 → 0.005 (encourage base action exploration)
  - `jerk_limit_penalty`: 0.1 → 0.05 (reduce movement suppression)

**Fix #3: Smart Distance Penalty**
- `src/rl_platform/tasks/mobile_mm/rewards.py`:
  - 90% penalty reduction when base is moving (>1cm threshold)
  - Full penalty when stationary (maintains gradient)
  - Encourages exploration: "Try moving base → penalty drops dramatically!"

**Files Modified:**
- `src/rl_platform/tasks/mobile_mm/observations.py`
- `src/rl_platform/tasks/mobile_mm/config.py`
- `src/rl_platform/tasks/mobile_mm/rewards.py`
- `scripts/analyze_trajectory_reach.py` (NEW - trajectory analysis tool)
- `docs/BASE_MOVEMENT_BUG_ANALYSIS.md` (NEW - 10-page investigation report)

---

#### Commit 6: Training Session Logging Infrastructure
```
Commit: 78216e1ec98b86b84cd2e8deb46cf0cc85e8aa43
Date: October 21, 2025 19:30:00 +0800 (approx)
Message: Add comprehensive training session logging infrastructure
```

**Changes:**
- Created `TRAINING_SESSIONS_MASTER_LOG.md` (this file) - single source of truth
- Created `docs/training_sessions/SESSION_5_LAUNCH_LOG.md` - detailed tracking template
- Documented all 5 previous commits with full git hashes and timestamps
- Both launch command options (4096 and 8192 envs)
- Complete success criteria, monitoring plan, and failure indicators
- Session comparison matrix and template for future sessions

**Files Modified:**
- `TRAINING_SESSIONS_MASTER_LOG.md` (NEW)
- `docs/training_sessions/SESSION_5_LAUNCH_LOG.md` (NEW)
- `docs/training_sessions/SESSION_5_BASE_FIX.md` (added in this commit)

---

**ROOT CAUSE IDENTIFIED:**
- 83.8% of trajectory waypoints OUT OF REACH (>0.6m from base)
- Mean distance: 2.3m, Max distance: 4.0m (trajectory REQUIRES base movement!)
- Policy had 0.958 explained variance but zero base movement after 25.7M steps
- Policy learned "I will get low rewards" but couldn't learn "HOW to fix it"

**CRITICAL FIXES:**

**Fix #1: Observation Space Enhancement (+4 dims)**
- `src/rl_platform/tasks/mobile_mm/observations.py`:
  - Added `base_to_target_xy` [2 dims]: XY offset from base to target
  - Added `base_to_target_dist` [1 dim]: Euclidean distance to target
  - Added `out_of_reach_flag` [1 dim]: Binary signal when target >0.6m from base
  - Total: 59 dims (was 55 dims with lookahead)

**Fix #2: Reward Rebalancing**
- `src/rl_platform/tasks/mobile_mm/config.py`:
  - `base_progress_reward`: 50.0 → 150.0 (3x stronger!)
  - `target_distance_penalty`: 10.0 → 3.0 (less harsh during movement)
  - `action_magnitude`: 0.01 → 0.005 (encourage base action exploration)
  - `jerk_limit_penalty`: 0.1 → 0.05 (reduce movement suppression)

**Fix #3: Smart Distance Penalty**
- `src/rl_platform/tasks/mobile_mm/rewards.py`:
  - 90% penalty reduction when base is moving (>1cm threshold)
  - Full penalty when stationary (maintains gradient)
  - Encourages exploration: "Try moving base → penalty drops dramatically!"

**Files Modified:**
- `src/rl_platform/tasks/mobile_mm/observations.py`
- `src/rl_platform/tasks/mobile_mm/config.py`
- `src/rl_platform/tasks/mobile_mm/rewards.py`
- `scripts/analyze_trajectory_reach.py` (NEW - trajectory analysis tool)
- `docs/BASE_MOVEMENT_BUG_ANALYSIS.md` (NEW - 10-page investigation report)

---

### Training Configuration

**Environment:**
- Task: `MobileMMTrackEE-v0`
- Trajectory Type: `multi_recorded`
- All Trajectories: Enabled
- Rendering: Headless

**Hyperparameters:**
- Total Timesteps: 100,000,000 (100M)
- Number of Environments: **[4096 or 8192 - TBD]**
- Steps per Rollout: 64
- Batch Size: **[256 or 512 - TBD based on num_envs]**
- Learning Rate: 3e-4

**PPO Settings:**
- Initial Entropy: 0.001
- Entropy Decay: Enabled (final: 1e-4, start: 50M, duration: 50M)
- Target KL: 1.0
- KL Schedule: Enabled (warmup: 0.25, main: 0.15, finetune: 0.07)

### Launch Command

**Option A: 4096 Environments**
```powershell
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 4096 `
  --n_steps 64 `
  --batch_size 256 `
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

**Option B: 8192 Environments**
```powershell
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 8192 `
  --n_steps 64 `
  --batch_size 512 `
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

**Command Used:** [TO BE FILLED AT LAUNCH]

### Expected Performance

**Reward Improvement Per Step (7.5cm base movement):**
- Base Mobilization Reward: +11.0
- Distance Penalty Improvement: +4.6 (90% reduction when moving)
- Position Tracking Improvement: +0.1
- Action Penalties: -0.025
- **NET IMPROVEMENT: +15.7 points** (vs -5.1 for staying still)

**Success Criteria:**
- **Early (1M steps, 5 min):** base_vx_action_std > 0.3, base_mobilization_reward > 1.0
- **Intermediate (10M steps, 1 hr):** base_movement > 0.005m/step, explained_variance > 0.80
- **Final (100M steps):** mean_tracking_error < 0.3m, total_reward > 35, explained_variance > 0.92

### Training Timeline

- **Launch Time:** [TO BE FILLED]
- **Estimated Duration:** 9-12 hours (depending on num_envs)
- **Checkpoint 1 (5 min):** [TO BE FILLED]
- **Checkpoint 2 (1 hour):** [TO BE FILLED]
- **Checkpoint 3 (3 hours):** [TO BE FILLED]
- **Checkpoint 4 (6 hours):** [TO BE FILLED]
- **Completion:** [TO BE FILLED]

### Results

**Status:** ❌ CATASTROPHIC FAILURE (Stopped at ~20M steps)

**Training Progression:**
- **10M steps (1 hour):** ✅ SUCCESS! Base actions ACTIVE
  - Base movements: 0.006-0.112m (reasonable)
  - base_mobilization_reward: -1.34 to +2.43
  - Mean tracking error: 0.877m (62% better than Session 4b!)
  - Broken envs: 0.3% (25 out of 8192)
  - **Observation space fix WORKED!**

- **20M steps (2 hours):** ❌ CATASTROPHIC FAILURE
  - Base moving WILDLY: 5-10 meters!
  - base_mobilization_reward: -21.31 to +10.26 (EXPLODED!)
  - Mean tracking error: 2.242m (156% WORSE!)
  - Broken envs: 51.5%→63.5% (4222→5205 out of 8192)
  - Constant reset spam (100+ resets per tracking window)
  - Wild arm motion: 8.67 rad/s (limit is 2.0 rad/s)
  - **Training death spiral - ABORTED**

**Root Cause Identified:**
- **Unbounded base_mobilization_reward** causing reward hacking
- Policy discovered: "move base as far as possible = huge rewards!"
- If base moves 5m: reward = 150 × 5.0 = 750 points! (vs 50 max for position tracking)
- Value function lags reality: explained_variance 0.927 while 63.5% broken!

**Second Bug Discovered:**
- Many environments spawning at identical trajectory start positions (e.g., [1.050, 0.080, 0.860])
- Lack of initial condition diversity
- Some envs (like 8093) consistently stuck in catastrophic failures

**Resolution:** Session 5b implements 3 critical fixes (reward capping, excessive penalty, rebalancing)

**Detailed Log:** [docs/tracking/SESSION_5_LAUNCH_LOG.md](docs/tracking/SESSION_5_LAUNCH_LOG.md)

---

## Session 5b: Unbounded Reward Fix

**Status:** 🚀 READY TO LAUNCH  
**Date:** October 21, 2025  
**Objective:** Fix unbounded reward bug that caused Session 5 catastrophic failure

### Git Commits

#### Commit 7: Session 5b Reward Fixes
```
Commit: 00fdbf5
Date: October 21, 2025 [Time TBD]
Message: Session 5b: Fix unbounded reward & add excessive movement penalty
```

**CRITICAL FIXES (3 changes):**

1. **Cap base_mobilization progress** (Priority 1 - CRITICAL):
   - Added `torch.clamp(progress, min=0.0, max=0.2)` in `base_mobilization_reward()`
   - Max reward = 150 × 0.2 = 30 points (reasonable vs 50 position tracking)
   - Prevents: 5m movement → 750 point explosion
   - Typical 7.5cm movement → 11.25 points

2. **Add excessive_base_movement_penalty** (Priority 2 - IMPORTANT):
   - New function: `excessive_base_movement_penalty()`
   - Heavily penalizes movements >10cm per step
   - Example: 1m movement → 10.0 × 0.9 = 9.0 point penalty
   - Prevents policy from exploiting uncapped rewards

3. **Increase target_distance_penalty** (Priority 3 - BALANCING):
   - Increased from 3.0 to 5.0
   - Compensates for capped mobilization reward
   - Maintains pressure to move base when needed

**Files Modified:**
- `src/rl_platform/tasks/mobile_mm/rewards.py`:
  * `base_mobilization_reward()`: Added progress capping (lines ~91-94)
  * `excessive_base_movement_penalty()`: New function (lines ~175-200)
  * `compute_combined_reward()`: Added excessive penalty to total
  * `components` dict: Added excessive_base_movement_penalty logging

- `src/rl_platform/tasks/mobile_mm/config.py`:
  * `RewardWeights.target_distance_penalty`: 3.0 → 5.0
  * `RewardWeights.excessive_base_movement_penalty`: 10.0 (NEW)

**Documentation:**
- `docs/tracking/SESSION_5B_FIX_SUMMARY.md`: Comprehensive fix documentation

---

### Training Configuration

**Environment:**
- Task: MobileMMTrackEE-v0
- Trajectory Type: multi_recorded (all 23 trajectories)
- Rendering: Headless
- Observations: 43 dims (+4 from Session 5 fix)

**Hyperparameters:**
- Total Timesteps: 100,000,000
- Number of Environments: 8192
- n_steps: 64
- Batch Size: 512
- Learning Rate: 3e-4
- Entropy Coefficient: 0.001 (decay to 1e-4)
- Enable KL Schedule: Yes (warmup 0.25, main 0.15, finetune 0.07)
- Target KL: 1.0

**Key Changes from Session 5:**
- ✅ base_mobilization progress CAPPED at 0.2m per step
- ✅ excessive_base_movement_penalty ADDED (10.0 × excess beyond 0.1m)
- ✅ target_distance_penalty INCREASED (3.0 → 5.0)
- ✅ All other hyperparameters UNCHANGED (isolate reward fixes)

### Launch Command

**Full Command:**
```powershell
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 8192 --n_steps 64 --batch_size 512 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless
```

**Command Used:** [TO BE FILLED AT LAUNCH]

### Expected Performance

**Reward Structure (Session 5b):**
- Position tracking: 0 to +50 points (exponential decay)
- Base mobilization: -5 to +30 points (CAPPED at 0.2m progress!)
- Distance penalty: 0 to -10 points (out of reach penalty, increased scale)
- Excessive movement: 0 to -X points (wild movements >10cm penalized)
- Action penalties: -0.5 to -2 points (typical)

**Success Criteria:**

**CRITICAL First 5 Minutes (1M steps):**
- [ ] Base movements: <0.15m (NOT >1m!)
- [ ] base_mobilization_reward: -5 to +30 range (NOT -21 to +10!)
- [ ] Excessive penalty: Near 0 (movements within bounds)
- [ ] Broken envs: <1% (NOT 51.5%!)
- **IF ANY FAIL → STOP IMMEDIATELY!**

**Early (5M steps, 30 min):**
- Mean tracking error < 1.0m (improving)
- Base mobilization mean > 0.0 (net positive strategy)
- Broken envs < 2%
- No reset spam

**Intermediate (10M steps, 1 hour):**
- Mean tracking error < 0.7m
- Base movements coordinated (5-15cm typical, rarely >20cm)
- "Good" tracking category > 5%
- Explained variance 0.80-0.85

**Final (100M steps, ~9 hours):**
- Mean tracking error < 0.25m (TARGET!)
- Base actively mobilizes for >70% OOR targets
- Total reward > 35 (consistently positive!)
- Explained variance > 0.92
- Policy respects 10cm movement limit naturally

### Training Timeline

- **Launch Time:** [TO BE FILLED]
- **Estimated Duration:** ~9 hours (8192 envs)
- **Checkpoint 1 (5 min, 1M steps):** [CRITICAL - ABORT IF FAILS]
- **Checkpoint 2 (30 min, 5M steps):** [TO BE FILLED]
- **Checkpoint 3 (1 hour, 10M steps):** [TO BE FILLED]
- **Checkpoint 4 (3 hours, 30M steps):** [TO BE FILLED]
- **Checkpoint 5 (6 hours, 60M steps):** [TO BE FILLED]
- **Completion (9 hours, 100M steps):** [TO BE FILLED]

### Results

**Status:** Not Started

**Final Metrics:** [TO BE FILLED]

**Outcome:** [TO BE FILLED]

**Detailed Log:** [docs/tracking/SESSION_5B_LAUNCH_LOG.md](docs/tracking/SESSION_5B_LAUNCH_LOG.md) (to be created at launch)

**Comparison with Session 5:**
- Session 5 @ 20M: 51.5% broken, mean error 2.242m, wild movements (5-10m)
- Session 5b @ 20M: [TO BE FILLED]

---

## Session 5: Base Mobilization Fix (FAILED)

**Status:** ❌ CATASTROPHIC FAILURE (Aborted at ~20M steps)  
**Date:** October 21, 2025  
**Objective:** Fix critical observation space bug preventing base mobilization learning

### Why This Session is Important
Session 5 PROVED the observation space fix works (base actions became active at 10M steps!), but discovered a NEW critical bug: unbounded rewards causing policy to learn catastrophic behavior. Session 5b addresses this with reward capping.

### Git Commits (Sessions 1-6 from Session 5)

[Previous commit documentation remains unchanged...]

**Status:** ⏸️ STOPPED (Issue Identified)  
**Date:** [Date TBD]  
**Objective:** Continue training to 100M steps

### Issue Summary

**Symptoms:**
- Excellent metrics: 0.958 explained variance, 0.00443 value loss
- **ZERO base movement** despite training to 25.7M steps
- Policy correctly predicted low rewards but didn't learn how to fix it

**Root Cause:**
- Observation space missing explicit base-to-target distance signal
- Reward structure made base movement net negative even when necessary
- 83.8% of trajectory waypoints require base movement (>0.6m from base)

**Resolution:** Session 5 implements critical fixes

### Training Configuration

[TO BE FILLED FROM SESSION 4b LOGS]

### Results

**Final Metrics:**
- Steps Completed: 25,700,000 / 100,000,000
- Explained Variance: 0.958
- Value Loss: 0.00443
- Base Movement: ~0 mm (ZERO!)
- Decision: Stopped for investigation

**Detailed Log:** [TO BE CREATED IF NEEDED]

---

## Session 4a

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** [TBD]

[TO BE FILLED]

---

## Session 3

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** [TBD]

[TO BE FILLED]

---

## Session 2

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** [TBD]

[TO BE FILLED]

---

## Session 1: Initial Training

**Status:** [TBD]  
**Date:** [TBD]  
**Objective:** Establish baseline performance

[TO BE FILLED]

---

## 📊 Session Comparison Matrix

| Session | Steps | Envs | Obs Dims | Base Movement | Tracking Error | Explained Var | Notes |
|---------|-------|------|----------|---------------|----------------|---------------|-------|
| 1 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | Baseline |
| 2 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 3 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4a | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4b | 25.7M | [TBD] | 55 | ~0 mm | ~2.3m | 0.958 | Stopped - no base movement |
| 5 | 0M | 4096/8192 | 59 | **TBD** | **TBD** | **TBD** | **Observation space fix** |

---

## 🔧 Hardware & Environment Info

**System:**
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- OS: Windows
- Isaac Lab: I:\isaaclab
- Isaac Sim: 5.0
- Workspace: C:\Users\yanbo\wSpace\cinebotRL

**Performance Benchmarks:**
- 4096 envs: ~8.3M steps/hour (~12 hours for 100M)
- 8192 envs: ~11M steps/hour (~9 hours for 100M)
- Memory usage: 18-20 GB (4096 envs), 22-23 GB (8192 envs)

---

## 📚 Key Documentation

### Session-Specific
- [Session 5 Launch Log](docs/training_sessions/SESSION_5_LAUNCH_LOG.md) - Detailed tracking
- [Session 5 Fix Documentation](docs/training_sessions/SESSION_5_BASE_FIX.md) - Fixes explained
- [Base Movement Bug Analysis](docs/BASE_MOVEMENT_BUG_ANALYSIS.md) - 10-page investigation

### General
- [Training Architecture](docs/architecture/training_architecture.md)
- [Reward System](docs/reference/reward_system.md)
- [Reward Cheatsheet](docs/reference/reward_cheatsheet.md)
- [Robot Constraints](docs/reference/robot_constraints_updated.md)
- [Training Diary](docs/training_sessions/TRAINING_DIARY.md)

### Setup Guides
- [Train on Windows](docs/setup/TRAIN_ON_WINDOWS.md)
- [Isaac Lab Windows Setup](docs/setup/isaaclab_windows.md)
- [Daily Workflow](docs/workflows/daily_workflow.md)

---

## 🎯 Success Metrics Definitions

### Base Action Activity
- **base_vx_action_std > 0.3**: Policy actively using forward velocity actions
- **base_wz_action_std > 0.2**: Policy actively using rotation actions
- **base_vel_x_mean > 0.01 m/s**: Real movement, not numerical noise

### Training Health
- **explained_variance > 0.80**: Value function learning properly
- **value_loss < 1.0**: Value predictions accurate
- **approx_kl < target_kl**: Policy updates stable

### Task Performance
- **mean_tracking_error < 0.3m**: Good end-effector tracking
- **total_reward > 35**: Net positive rewards (successful behavior)
- **base_mobilization_active > 70%**: Using base when needed

---

## 🚨 Common Issues & Resolutions

### Issue 1: No Base Movement (Session 4b)
**Symptoms:** Zero base action despite good explained variance  
**Root Cause:** Missing base-to-target observations  
**Resolution:** Added 4 observation dims in Session 5

### Issue 2: OOM (Out of Memory)
**Symptoms:** Training crashes, CUDA out of memory  
**Resolution:** Reduce num_envs from 8192 to 4096

### Issue 3: NaN Values
**Symptoms:** NaN in loss or rewards  
**Resolution:** Check for gradient explosion, reduce learning rate

---

## Session 6: Fix Frozen Base (3 Critical Fixes)

**Status:** ⚠️ **MIXED RESULTS** - Base moves, but catastrophic self-collision  
**Date:** 2025-10-22 to 2025-10-23  
**Objective:** Fix frozen base by adjusting jerk penalty, adding ContactSensor, and fixing shape bug

### Changes Applied

**Critical Fixes:**
1. **Jerk penalty: 5.0 → 50.0 m/s³** - Allow base movement without excessive penalties
2. **Added ContactSensor** - Detect contact forces (validated at 952.51 N)
3. **Fixed prev_joint_vel shape: 6 → 9 columns** - Match observation buffer

### Training Configuration

**Environment:**
- Task: MobileMMTrackEE-v0
- Trajectory Type: multi_recorded (all 1,038 trajectories)
- Rendering: Headless
- Number of Environments: 4096 (reduced from 8192)

**Hyperparameters:**
- Total Timesteps: 100,000,000
- n_steps: 128 (increased from 64)
- Batch Size: 1024 (increased from 512)
- Learning Rate: 3e-4
- Entropy Coefficient: 0.001 (decay to 1e-4)
- KL Schedule: warmup 0.25, main 0.15, finetune 0.07

### Launch Command

```powershell
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
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

**Started:** 2025-10-22 22:30  
**Completed:** 2025-10-23 morning (~overnight run)

### Results

**Training Completion:**
- ✅ Completed 100,073,472 steps (100M target)
- Final checkpoint: `logs/sb3/mobilemmtrackee_v0/20251022_230622/final_model.zip`
- No crashes, no NaN values
- All 3 critical fixes deployed successfully

**Evaluation Results (2025-10-23 09:17):**
```
Command: evaluate.py --checkpoint final_model.zip --num_envs 16 --num_episodes 5 --headless
Episodes: 5 (16 parallel envs × 399 steps)
Mean Reward: -11,715,724 ± 39,599
Min Reward: -11,781,028
Max Reward: -11,661,779
Episode Length: 399 steps
```

### Critical Analysis

✅ **SUCCESSES - The 3 Fixes WORKED:**

1. **✅ Base IS Moving!**
   - Base mobilization rewards: 0.02-0.20 (non-zero confirms movement)
   - Base position changes: [1.135, 0.103] → [1.153, 0.136] → [1.163, 0.145]
   - PPR offsets varying: -0.058 to -0.190 m (base adjusting)
   - **Jerk penalty fix successful!**

2. **✅ ContactSensor WORKING!**
   - Detecting forces: 635.66 N at step 302
   - Contact warnings: "Collision on body: base"
   - Sensor properly integrated and reporting

3. **✅ Shape Fix Successful!**
   - No runtime errors
   - Training completed full 100M steps
   - Observation buffer stable (43 dims)

🎯 **Tracking Performance (ignoring collision penalties):**
- **Best:** 0.49m error (Env 13) - decent tracking!
- **Typical:** 0.5-2.5m range
- **Position tracking rewards:** Up to 39.26 points
- **Base movements:** Coordinated 5-15cm adjustments

❌ **CATASTROPHIC PROBLEM: Self-Collision Explosion**

**The Issue:**
```
[DEBUG Step 300] Total reward: -30,308.07
  self_collision_penalty: +30,264.8711  ← DOMINATES EVERYTHING
  position_tracking: +0.5647
  base_mobilization: +0.0770

⚠️ [COLLISION DETECTED] Step 302
   Max contact force: 635.66 N (threshold: 1.00 N)
   Collision on body: base
```

**Impact:**
- Self-collision penalty: ~30,000 per step
- Over 399 steps: 30,000 × 399 = 11.97M negative reward
- **This is 1000x larger than all other rewards combined!**
- Learning signal completely destroyed

**Root Causes Identified:**
1. **Penalty magnitude excessive:** 30K per collision vs 50 max for position tracking
2. **Constant base collisions:** Robot geometry causing perpetual self-collision
3. **Possible sign error:** Penalty shown as "+30264" (should be negative?)
4. **Collision filtering:** Base may be colliding with arms/legs inappropriately

### Key Insights

**What We Learned:**
- ✅ Base CAN move when jerk penalty is reasonable (50.0 works)
- ✅ ContactSensor works perfectly (detecting real forces)
- ✅ Policy learned decent tracking (~0.5-2.5m) despite reward corruption
- ❌ Self-collision penalty destroys learning (needs 100x reduction OR geometry fix)

**Why Reward is Catastrophic:**
- Position tracking best: +39.26 points
- Base mobilization: +0.2 points
- Self-collision: -30,000 points ← **Overwhelms everything!**
- **Net:** Policy has no incentive to track well, only avoid collisions

### Next Steps → Session 7

**Fix Options (choose 1 or combine):**

1. **Option A: Reduce Penalty Weight**
   - `self_collision_penalty_weight: 1000.0 → 10.0` (100x reduction)
   - Keep detecting collisions, make penalty proportional to other rewards

2. **Option B: Fix Collision Geometry**
   - Investigate robot URDF collision meshes
   - Check if base is colliding with legs/arms inappropriately
   - Add collision filtering (base should NOT collide with own parts)

3. **Option C: Verify Penalty Math**
   - Check if penalty is subtracting or adding
   - Why is self_collision shown as "+30264" instead of "-30264"?
   - Ensure reward = tracking + base_mob - penalties

**Recommended Approach for Session 7:**
```python
# In config.py RewardWeights:
self_collision_penalty: 1000.0 → 5.0  # Make it 200x smaller
# Then test 1M steps to verify:
# - Base still moves
# - Reward becomes positive
# - Tracking improves
```

**Detailed Analysis:** [To be created: `docs/SESSION_6_COLLISION_ANALYSIS.md`]

---

## 📝 Template for New Session

```markdown
## Session X: [Session Name]

**Status:** [🚀 Active / ✅ Complete / ⏸️ Paused / ❌ Failed]  
**Date:** [Date]  
**Objective:** [Primary goal]

### Git Commits

#### Commit 1: [Commit Title]
```
Commit: [Git hash]
Date: [Date]
Message: [Commit message]
```

**Changes:**
- [Change 1]
- [Change 2]

**Files Modified:**
- [File 1]
- [File 2]

---

### Training Configuration

**Environment:**
- Task: [Task name]
- Trajectory Type: [Type]
- Rendering: [Headless/GUI]

**Hyperparameters:**
- Total Timesteps: [Number]
- Number of Environments: [Number]
- Learning Rate: [Value]
- [Other params]

### Launch Command

```powershell
[Full command]
```

**Command Used:** [Timestamp and confirmation]

### Results

**Status:** [In Progress / Complete / Failed]

**Final Metrics:**
- [Metric 1]: [Value]
- [Metric 2]: [Value]

**Outcome:** [Success / Failure / Partial]

**Detailed Log:** [Link to session-specific log]

---
```

---

**Last Updated:** October 23, 2025 09:30 +0800  
**Current Session:** 6 (EVALUATED - Self-collision issue found)  
**Next Session:** 7 (Fix self-collision penalty)  
**Repository Status:** Session 6 evaluated, Session 7 plan ready

**Quick Action for Session 7:**
```python
# In src/rl_platform/tasks/mobile_mm/config.py:
self_collision_penalty: float = 5.0  # Change from 1000.0
# Then launch training with same config as Session 6
```
