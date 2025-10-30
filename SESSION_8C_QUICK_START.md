# Session 8c Quick Start Guide

**Goal**: Train a policy that keeps the mobile base within optimal arm workspace (0.4-0.6m from targets) while maintaining excellent tracking accuracy.

**Status**: Implementation complete ✅  
**Next Step**: Run smoke test 🚀

---

## TL;DR - Run This Now

```powershell
# 1. Navigate to project
cd C:\Users\yanbo\wSpace\cinebotRL

# 2. Run 15-30 minute smoke test (MANDATORY)
.\scripts\launch_session_8c.ps1 -Phase smoke

# 3. Watch TensorBoard for reachability_maintenance_reward
#    Should trend toward POSITIVE (not -135 like Session 8b)

# 4. If smoke test looks good, run Phase 1:
.\scripts\launch_session_8c.ps1 -Phase easy
```

---

## What Changed from Session 8b?

### The Problem
Session 8b had **reachability reward = -135.21** (should be positive!). Analysis revealed:
- Linear penalty too forgiving: at 2.0m, penalty only -280
- Policy learned to accept penalties rather than move base closer
- Base drifted to ~1.95m from targets (way beyond optimal 0.6m)

### The Fix (Session 8c)
1. **Quadratic penalty**: `-2(d-0.6)²` instead of `-2(d-0.6)`
   - At 2.0m: penalty now -3,920 (14× harsher!)
   - Makes large distances prohibitively expensive
   
2. **Doubled reachability weight**: 50 → 100
   - Now 50% as important as position tracking
   
3. **Increased tracking weights**: position 150→200, orientation 75→100
   - Ensures tracking remains dominant objective
   
4. **Softened motion penalties**: Allow agile base movements
   - excessive_base_movement: 15→10
   - velocity_limit: 1.5→1.0
   - jerk_limit: 0.01→0.005

5. **Curriculum learning**: 40M easy + 60M medium + 100M full
   - Gradual difficulty increase vs monolithic 200M
   
6. **Tighter PPO control**: KL bounds 60% lower, value clipping at 0.3
   - Prevents instability from reward spikes

---

## Smoke Test (15-30 Minutes)

### Purpose
Verify the new config doesn't cause:
- ❌ NaN values (training explosion)
- ❌ Reward oscillations (instability)
- ❌ Reachability still hugely negative (tuning inadequate)

### Command
```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\launch_session_8c.ps1 -Phase smoke
```

### What to Watch

#### 1. Console Output
```
Iteration 1/38:  approx_kl=0.xxx, clip_fraction=0.xxx, entropy=-x.xx
  ✓ reachability_maintenance_reward: -50 to +10 (expected initially negative)
  ✓ position_tracking: +50 to +100 (should be positive)
  ✓ No NaN values

Iteration 10/38:
  ✓ reachability_maintenance_reward: -10 to +30 (trending up!)
  
Iteration 38/38:
  ✓ reachability_maintenance_reward: +20 to +50 (POSITIVE!)
```

#### 2. TensorBoard (Open in Browser)
```powershell
# In separate terminal
cd C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\<timestamp>
tensorboard --logdir .
# Open: http://localhost:6006
```

**Key plots to check**:
- `rollout/reachability_maintenance_reward` → should trend **upward** toward positive
- `rollout/ep_rew_mean` → should be +10k to +50k (not -405k catastrophic)
- `train/entropy` → should stay ~0.001 (no premature decay)
- `train/approx_kl` → should be < 0.15 (controlled updates)

### Success Criteria
- ✅ No crashes or NaN values
- ✅ Reachability reward trends positive (even if starts negative)
- ✅ Episode rewards reasonable (+10k to +50k, not -405k)
- ✅ Training stable (KL < 0.2, entropy steady)

### If Smoke Test Fails

**Problem**: Reachability still hugely negative (-100+) at end
```python
# Fix: Increase weight further
reachability_maintenance_reward: 100 → 150
```

**Problem**: NaN values or training explosion
```python
# Fix: Lower learning rate or increase value clipping
learning_rate: 3e-4 → 1e-4
clip_range_vf: 0.3 → 0.5
```

**Problem**: Tracking degrades as reachability improves
```python
# Fix: Increase tracking weights
position_tracking: 200 → 250
orientation_tracking: 100 → 125
```

---

## Full Training (30-40 Hours Total)

### Phase 1: Easy (6-8 Hours)

**Command**:
```powershell
.\scripts\launch_session_8c.ps1 -Phase easy
```

**Configuration**:
- Timesteps: 40M
- Environments: 128
- Trajectories: All 1,038 (TODO: filter for easy ones)

**Expected Results** (at 40M):
- Reachability reward: 0 to +30 (learning to stay in workspace)
- Position error: 200-250 cm (still learning)
- Base-target distance: 0.8-1.2m (improving from 1.95m)

**Evaluation**:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate_quantitative.py `
  --checkpoint <logs/.../final_model.zip> `
  --task MobileMMTrackEE-v0 `
  --num_episodes 200 `
  --num_envs 64 `
  --headless `
  --trajectory_type multi_recorded `
  --trajectory_dir C:\Users\yanbo\wSpace\cinebotRL\trajectoryToLearn\world_json `
  --use_all_trajectories
```

**Decision Point**: If position P95 error < 300 cm and reachability reward positive, proceed to Phase 2.

---

### Phase 2: Medium (9-12 Hours)

**Command**:
```powershell
.\scripts\launch_session_8c.ps1 -Phase medium -Checkpoint <logs/.../final_model.zip>
```

**Configuration**:
- Timesteps: 60M (cumulative 100M)
- Environments: 160
- Start from: Phase 1 checkpoint

**Expected Results** (at 100M):
- Reachability reward: +30 to +60 (mastered workspace positioning)
- Position error: 150-200 cm (significant improvement)
- Base-target distance: 0.5-0.8m (approaching optimal)

**Evaluation**: Same as Phase 1

**Decision Point**: If metrics improving, proceed to Phase 3.

---

### Phase 3: Full (15-20 Hours)

**Command**:
```powershell
.\scripts\launch_session_8c.ps1 -Phase full -Checkpoint <logs/.../final_model.zip>
```

**Configuration**:
- Timesteps: 100M (cumulative 200M)
- Environments: 192
- Start from: Phase 2 checkpoint
- Entropy decay: Starts at 120M total (20M into this phase)

**Expected Results** (at 200M):
- Reachability reward: +50 to +100 (optimal)
- Position error: 100-150 cm (target quality)
- Orientation error: 30-35° (acceptable)
- Base-target distance: 0.4-0.6m (optimal workspace)
- Reward variance: ±50k (consistent, not ±155k)

**Final Evaluation**: Generate comprehensive analysis report

---

## Alternative: Complete Run (Skip Curriculum)

**When to use**: Only if smoke test perfectly validates config and you're confident.

**Command**:
```powershell
.\scripts\launch_session_8c.ps1 -Phase complete
```

**Configuration**:
- Timesteps: 200M (single run)
- Environments: 192
- Duration: 30-40 hours continuous

**Trade-offs**:
- ✅ No checkpoint management
- ✅ Simpler workflow
- ❌ Can't evaluate/abort at intermediate points
- ❌ If config wrong, waste 30-40 hours

---

## Monitoring During Training

### Console Output (Every 5 Iterations)
```
=== Training Monitor (Iteration 5) ===
Reward Components:
  position_tracking: +70.2 ± 30.5 [+10.1, +120.4]
  orientation_tracking: +52.8 ± 25.3 [+5.2, +98.1]
  reachability_maintenance_reward: +35.6 ± 15.2 [-5.1, +75.3]  ← WATCH THIS
  jerk_penalty: -8.5 ± 3.2 [-15.2, -2.1]
  
Tracking Errors:
  Position: 1.85 ± 0.95m [0.15, 4.25]
  Orientation: 38.2 ± 18.5° [5.1, 85.3]
  
Base Movement:
  Linear speed: 0.35 ± 0.18 m/s [0.05, 0.85]
  Angular speed: 0.05 ± 0.08 rad/s [0.0, 0.35]
  
Reachability:
  Percentage: 68.2% (131/192 envs)
  Mean distance: 0.72 ± 0.35m [0.25, 1.85]
  Alignment: 0.85 ± 0.15 [0.35, 1.0]
```

### TensorBoard Metrics

**Primary metrics**:
- `rollout/reachability_maintenance_reward` → Target: +50 to +100
- `rollout/ep_rew_mean` → Target: +50k to +100k
- `rollout/ep_len_mean` → Should stay stable (not early termination)

**Diagnostic metrics**:
- `train/approx_kl` → Should be < target_kl (0.5)
- `train/clip_fraction` → Should be 0.1-0.3 (healthy range)
- `train/entropy` → Should decay after 120M timesteps
- `train/value_loss` → Should stabilize (not explode)

---

## Expected Timeline

| Phase | Duration | Cumulative | Timesteps | Key Milestone |
|-------|----------|------------|-----------|---------------|
| Smoke test | 15-30 min | 30 min | 10M | Validate config |
| Phase 1 (Easy) | 6-8 hours | 8 hours | 40M | Learn workspace positioning |
| Evaluation | 30 min | 8.5 hours | - | Check P95 error |
| Phase 2 (Medium) | 9-12 hours | 20 hours | 100M | Improve tracking + reachability |
| Evaluation | 30 min | 20.5 hours | - | Verify metrics improving |
| Phase 3 (Full) | 15-20 hours | 40 hours | 200M | Master all trajectories |
| Final Evaluation | 30 min | 40.5 hours | - | Comprehensive analysis |

**Total**: ~40-41 hours from start to final results

---

## Success Indicators (What to Celebrate)

### At 40M (Phase 1 Complete)
- ✅ Reachability reward positive (0 to +30)
- ✅ Base staying within 1.0m of targets
- ✅ No catastrophic failures (-50k min reward)

### At 100M (Phase 2 Complete)
- ✅ Reachability reward healthy (+30 to +60)
- ✅ Base within 0.8m most of the time
- ✅ Position error trending down (200→150 cm)

### At 200M (Phase 3 Complete)
- ✅ Reachability reward excellent (+50 to +100)
- ✅ Base within optimal 0.4-0.6m workspace
- ✅ Position error 100-150 cm (target quality)
- ✅ Orientation error 30-35° (acceptable)
- ✅ Consistent performance (reward std < ±50k)
- ✅ Ready for deployment!

---

## Files Reference

### Modified Files
```
src/rl_platform/tasks/mobile_mm/
├── rewards.py          - Quadratic reachability penalty
├── config.py           - Updated weights
└── env.py              - Reachability stats storage

scripts/reinforcement_learning/sb3/
├── train.py            - Value clipping support
└── launch_session_8c.ps1  - Curriculum launcher (NEW)
```

### Documentation
```
SESSION_8C_IMPLEMENTATION.md     - Detailed implementation notes
SESSION_8B_VS_8C_COMPARISON.md   - Side-by-side comparison
SESSION_8C_QUICK_START.md        - This file
evaluation_results/session_8b_200M/ANALYSIS_REPORT.md  - Session 8b results
```

---

## Quick Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Reachability stays negative | Weight too low | Increase weight 100→150 |
| Tracking degrades | Tracking weights too low | Increase position 200→250 |
| Base barely moves | Penalties too strong | Reduce excessive_base 10→5 |
| NaN values | Learning rate too high | Lower LR 3e-4→1e-4 |
| Policy oscillates | KL bounds too loose | Lower target_kl 0.5→0.3 |
| Crashes with OOM | Batch size too large | Reduce batch 2048→1024 |

---

## What Makes Session 8c Different?

**Session 8b**: "Move fast, learn later"
- 20,480 envs, 12 hours, one shot
- Result: Base drifts to 1.95m, reachability -135 ❌

**Session 8c**: "Learn right, then scale"
- 128-192 envs, 40 hours, three phases
- Expected: Base stays at 0.5m, reachability +75 ✅

**The insight**: Quadratic penalties + curriculum learning = policies that respect hard constraints.

---

## Ready to Launch?

```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\launch_session_8c.ps1 -Phase smoke
```

**Time commitment**: 15-30 minutes to know if config is good.

**If smoke test succeeds**: Plan for 40 hours of training over next 2-3 days.

**If smoke test fails**: Debug, adjust weights, retry (much better than wasting 40 hours!).

---

**Status**: Implementation complete ✅  
**Next Step**: Smoke test 🚀  
**Key Command**: `.\scripts\launch_session_8c.ps1 -Phase smoke`

Good luck! 🎯
