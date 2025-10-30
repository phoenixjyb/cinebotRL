# Session 8c-v2 Monitoring Guide

**Training interrupted at**: Iteration 7/96 (14.7M/200M, 7.3% complete)  
**Status**: All monitoring tools implemented, ready to resume training

---

## Quick Resume Checklist

1. **Resume training** (will continue from iteration 7):
   ```powershell
   .\scripts\launch_session_8c.ps1 -Phase complete
   ```

2. **Start monitoring in separate terminal** (non-intrusive):
   ```powershell
   # Option A: One-time check
   python scripts/monitoring/check_training_progress.py --log_dir logs/sb3/mobilemmtrackee_v0/<timestamp>
   
   # Option B: Continuous monitoring (30s refresh)
   python scripts/monitoring/check_training_progress.py --log_dir logs/sb3/mobilemmtrackee_v0/<timestamp> --watch
   ```

3. **View TensorBoard** (real-time metrics):
   ```powershell
   tensorboard --logdir logs/sb3
   ```
   - Navigate to: http://localhost:6006
   - Watch: `monitoring/base_target_dist_mean`, `reward_components/*`

---

## Monitoring Infrastructure (NEW)

### 1. Enhanced TrainingMonitorCallback ✅

**Location**: `scripts/reinforcement_learning/sb3/train.py` (lines 513-620)

**Features**:
- **Base-Target Distance Tracking** (CRITICAL for quadratic penalty):
  - Logs mean/std/max distance to TensorBoard (`monitoring/base_target_dist_*`)
  - Zone distribution percentages:
    - `monitoring/optimal_zone_pct` (target <0.4m)
    - `monitoring/acceptable_zone_pct` (0.4-0.6m)
    - `monitoring/unreachable_zone_pct` (>0.6m) ⚠️ WATCH THIS
  - Prints every 5 iterations in training logs
  
- **Reward Component Logging**:
  - All reward components → TensorBoard (`reward_components/*`)
  - Watch for: `reachability_maintenance_reward` (should climb positive)
  - Monitor penalties: `target_distance_penalty`, `base_overshoot_penalty`

**Expected Output** (every 5 iterations):
```
============================================================
[Training Monitor] Iteration 10 @ 20.1M steps
============================================================

[Reward Components] Mean per episode:
  position_tracking                   :  150.234
  orientation_tracking                :   85.123
  reachability_maintenance_reward     :   12.456  ← Watch this!
  base_overshoot_penalty              :   -5.234
  ...

[Base-Target Distance] ⚠️ CRITICAL for quadratic penalty
  Distance (m):        mean=0.5234, std=0.1234
                       min=0.2134, max=0.8765
  Zone distribution:   Optimal (<0.4m): 45.2%
                       Acceptable (0.4-0.6m): 42.3%
                       Unreachable (>0.6m): 12.5% ⚠️  ← TARGET: <15%
```

### 2. check_training_progress.py ✅

**Location**: `scripts/monitoring/check_training_progress.py`

**Purpose**: Non-intrusive monitoring (reads CSV, no process interaction)

**Usage**:
```powershell
# One-time check
python scripts/monitoring/check_training_progress.py --log_dir <path>

# Continuous watch (30s refresh)
python scripts/monitoring/check_training_progress.py --log_dir <path> --watch
```

**Health Checks**:
- **Explained Variance** (EV):
  - ✅ GOOD: EV ≥ 0.75
  - ⚠️ WARNING: 0.65 ≤ EV < 0.75
  - 🔴 CRITICAL: EV < 0.65 (consider curriculum)
  
- **KL Divergence**:
  - ✅ GOOD: approx_kl < 0.03
  - ⚠️ WARNING: approx_kl ≥ 0.03
  
- **Clip Fraction**:
  - ✅ GOOD: clip_fraction < 0.2
  - ⚠️ WARNING: clip_fraction ≥ 0.2

**Trend Analysis**:
- EV trend: IMPROVING / STABLE / DEGRADING
- FPS trend: STABLE / DECLINING / IMPROVING

**Example Output**:
```
================================================================================
  SESSION 8C-V2 TRAINING PROGRESS - 2024-01-15 14:30:00
================================================================================

📊 PROGRESS:
  Iteration:  10/96 (10.4%)
  Timesteps:  20,971,520 / 200,000,000
  FPS:        15,533 (STABLE)

⏱️  TIME:
  Elapsed:    0.35 hours
  Remaining:  3.15 hours (est.)
  Total:      3.50 hours (est.)

🏥 TRAINING HEALTH:
  ✅ Explained Variance: 0.752 [GOOD]
     Trend: IMPROVING
  ✅ Approx KL:          0.0089 (target: <0.03)
  Clip Fraction:      0.115 (target: <0.2)
  Entropy Loss:       -3.2
  Std:                0.361
  Value Loss:         0.249

💡 RECOMMENDATIONS:
  ✅ EV healthy (>0.75). Continue training.

  📌 Checkpoint available for evaluation:
     checkpoints/ppo_mobile_mm_20971520_steps.zip
     Run: python scripts/reinforcement_learning/sb3/evaluate_quantitative.py \
          --checkpoint <path> --num_episodes 10 --headless
```

### 3. quick_checkpoint_eval.py ✅

**Location**: `scripts/monitoring/quick_checkpoint_eval.py`

**Purpose**: Fast checkpoint evaluation (10-20 episodes) for validation

**Usage**:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p ..\cinebotRL\scripts\monitoring\quick_checkpoint_eval.py `
  --checkpoint ..\cinebotRL\logs\sb3\MobileMMTrackEE-v0\<timestamp>\checkpoints\ppo_mobile_mm_20000000_steps.zip `
  --num_episodes 10 `
  --headless
```

**What It Checks**:
- Base-target distance distribution (CRITICAL)
- Position error mean (target <1.0m)
- Orientation error mean (target <35°)
- Reachability reward (should be positive)

**When to Run**:
- At 20M checkpoint (iteration 10): First validation
- At 60M checkpoint (iteration 29): Mid-training check
- At 120M checkpoint (iteration 58): Entropy decay validation
- At 200M (final): Full evaluation

---

## Critical Metrics to Monitor

### 1. Base-Target Distance (MOST IMPORTANT)

**Why**: Quadratic reachability penalty is `-2.0 × (dist-0.6)²` at scale=100
- At 1.0m: -320 reward (devastating)
- At 1.5m: -1,620 reward (catastrophic)
- At 2.0m: -3,920 reward (apocalyptic)

**Target Distribution**:
- ✅ **GOAL**: 85%+ timesteps in 0.4-0.7m range
- ⚠️ **WARNING**: 70-85% in range, monitor closely
- 🔴 **CRITICAL**: <70% in range, consider penalty adjustment

**Where to Monitor**:
- TensorBoard: `monitoring/base_target_dist_mean`, `monitoring/*_zone_pct`
- Training logs: `[Base-Target Distance]` section every 5 iterations
- Checkpoint eval: `base_target_distance_mean` in results

### 2. Explained Variance (EV)

**Current**: 0.75 at iteration 7 (recovered from 0.653 at iter 4)

**Decision Tree**:
- **EV ≥ 0.75**: ✅ Healthy, continue
- **0.70 ≤ EV < 0.75**: ⚠️ Monitor, may recover
- **0.65 ≤ EV < 0.70**: 🟡 Watch closely, prepare curriculum
- **EV < 0.65**: 🔴 ABORT, switch to curriculum training

**Where to Monitor**:
- TensorBoard: `train/explained_variance`
- progress.csv: `train/explained_variance` column
- check_training_progress.py: Health status with trend

### 3. Reachability Maintenance Reward

**Expected Behavior**:
- **Early (0-20M)**: Negative to small positive (learning phase)
- **Mid (20-120M)**: Climbing toward +50-100 (optimization)
- **Late (120-200M)**: Stable +80-100 (converged)

**Red Flags**:
- Stuck at negative: Policy not learning base positioning
- Large negative (<-50): Trajectories incompatible with reach constraints
- Oscillating wildly: Reward scale too high, destabilizing training

**Where to Monitor**:
- TensorBoard: `reward_components/reachability_maintenance_reward`
- Training logs: `[Reward Components]` section every 5 iterations

### 4. Entropy Decay Validation (at 120M steps)

**Milestone**: Iteration 58 (120M steps, ~2 hours from resume)

**Expected**:
- `[EntropyDecay] Step 120.0M: ent_coef = 0.000XXX` messages
- `train/std` drops from 0.361 → 0.30-0.35
- `train/entropy_loss` decreases

**Red Flags**:
- No `[EntropyDecay]` messages at 120M → callback broken
- Std stays at 0.361 → schedule not updating
- Entropy_loss unchanged → exploration not decreasing

**Where to Monitor**:
- Training logs: Search for `[EntropyDecay]`
- TensorBoard: `train/std`, `train/entropy_loss`

---

## Evaluation Checkpoints

### Iteration 10 (20M steps, ~30 min from resume)

**Quick Eval**:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p ..\cinebotRL\scripts\monitoring\quick_checkpoint_eval.py `
  --checkpoint <path>/ppo_mobile_mm_20000000_steps.zip `
  --num_episodes 10 --headless
```

**Decision Point**:
- **Scenario A** (base-target 0.4-0.7m, position <1.0m):
  - ✅ Continue to 60M
  - No changes needed
  
- **Scenario B** (base-target 0.8-1.2m, position 1.0-2.0m):
  - ⚠️ Monitor closely
  - Continue to 60M for more data
  
- **Scenario C** (position >2.0m, base-target >1.5m):
  - 🔴 PAUSE training
  - Adjust tracking reward scale: 1.0 → 0.5
  - Reduce reachability penalty: scale=100 → 50
  
- **Scenario D** (EV <0.65):
  - 🔴 ABORT
  - Switch to curriculum: easy → medium → full

### Iteration 29 (60M steps, ~1.5 hours from resume)

**Check**:
- Base-target distance tightening?
- Position error improving?
- Reachability reward climbing?

**Decision**:
- If all improving → Continue to 120M
- If stagnant → Consider reducing reachability scale 100→75

### Iteration 58 (120M steps, ~2 hours from resume)

**Validate**:
- Entropy decay working (`[EntropyDecay]` messages)
- Std dropping from 0.361 → 0.30-0.35
- Policy exploiting learned behavior

### Iteration 96 (200M steps, ~3.5 hours from resume)

**Full Evaluation**:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p ..\cinebotRL\scripts\reinforcement_learning\sb3\evaluate_quantitative.py `
  --checkpoint logs/.../ppo_mobile_mm_200000000_steps.zip `
  --task MobileMMTrackEE-v0 `
  --num_episodes 200 `
  --num_envs 64 `
  --headless `
  --trajectory_type multi_recorded `
  --trajectory_dir ..\trajectoryToLearn\world_json `
  --use_all_trajectories
```

**Comparison vs Session 8b**:
- Session 8b: -135 reachability, 238cm position, 47.8° orientation
- Session 8c-v2 target: +50-100 reachability, <100cm position, <35° orientation

---

## TensorBoard Graphs to Watch

### Critical Graphs:
1. **monitoring/base_target_dist_mean** (MOST IMPORTANT)
   - Should stabilize in 0.5-0.6m range
   - Watch for upward drift (bad) or tightening (good)

2. **monitoring/unreachable_zone_pct**
   - Target: <15%
   - Warning: 15-25%
   - Critical: >25%

3. **reward_components/reachability_maintenance_reward**
   - Should climb from negative/zero → +50-100
   - Watch for stagnation or oscillation

4. **train/explained_variance**
   - Should stay >0.75
   - Trend: flat or slightly improving

5. **train/std**
   - Should drop at 120M (entropy decay)
   - 0.361 → 0.30-0.35 by 200M

### Supporting Graphs:
- `train/approx_kl`: Should stay <0.03
- `train/clip_fraction`: Should stay <0.2
- `reward_components/position_tracking`: Should improve
- `reward_components/orientation_tracking`: Should improve

---

## Troubleshooting

### Training stuck (EV <0.65, no improvement after 20M)

**Action**: Switch to curriculum training
```powershell
# Easy phase (128 envs, 20M steps)
.\scripts\launch_session_8c.ps1 -Phase easy

# Medium phase (512 envs, 40M steps)
.\scripts\launch_session_8c.ps1 -Phase medium

# Full phase (16384 envs, 140M steps)
.\scripts\launch_session_8c.ps1 -Phase complete
```

### Base-target distance too high (>1.0m mean)

**Action 1**: Reduce reachability penalty harshness
- Edit `src/rl_platform/tasks/mobile_mm/rewards.py` line 280:
  ```python
  quadratic_penalty = -1.0 * (excess_dist ** 2)  # Was -2.0
  ```

**Action 2**: Increase tracking reward scale
- Edit `scripts/launch_session_8c.ps1`:
  ```powershell
  $trackingRewardScale = 1.5  # Was 1.0
  ```

### Entropy decay not working (std stuck at 0.361 after 120M)

**Check**: Look for `[EntropyDecay]` messages in logs
**Fix**: Ensure `--enable_entropy_decay` flag in launch script
**Validate**: Check `train/std` in TensorBoard after 120M

---

## Files Modified (Commit: 2f58e59)

1. **scripts/reinforcement_learning/sb3/train.py**:
   - Enhanced `TrainingMonitorCallback` (lines 565-597)
   - Added base-target distance logging with zone analysis
   - Added reward component logging to TensorBoard

2. **scripts/monitoring/check_training_progress.py** (NEW):
   - Non-intrusive progress monitoring via CSV
   - Health checks, trend analysis, recommendations
   - Watch mode for continuous monitoring

3. **scripts/monitoring/quick_checkpoint_eval.py** (NEW):
   - Fast checkpoint evaluation wrapper
   - Focuses on critical metrics for Session 8c-v2
   - Provides expected range guidance

---

## Next Steps (After Resume)

1. ✅ Resume training: `.\scripts\launch_session_8c.ps1 -Phase complete`
2. ✅ Start monitoring: `python scripts/monitoring/check_training_progress.py --log_dir <path> --watch`
3. ✅ Open TensorBoard: `tensorboard --logdir logs/sb3`
4. ⏰ Wait for iteration 10 (~20 min): Run quick eval
5. ⏰ Monitor to iteration 58 (~2 hours): Validate entropy decay
6. ⏰ Complete training (~3.5 hours total): Full evaluation

**Good luck! 🚀**
