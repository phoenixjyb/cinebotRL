# Session 8H Implementation Plan

**Date**: November 3, 2025  
**Status**: 📋 **PLANNING** - Ready for implementation  
**Goal**: Fix Session 8g curriculum failures while preserving workspace improvements

---

## Executive Summary

Session 8g proved the evidence-based workspace expansion approach (position improved +2% vs 8f baseline), but **catastrophically failed due to curriculum design flaws**:
- Orientation under-trained in Stage 1 (weight 15.0 insufficient, only 3x position 5.0)
- Abrupt transition @ 50M shocked value function (variance dropped to -0.241)
- Final collapse @ 100M (std exploded 19,000x)

**Session 8h fixes** these issues with:
1. **Proportional curriculum weights**: (4.0, 12.0) → (10.0, 30.0) maintains 1:3 ratio
2. **Gradual transition**: 45M-55M linear interpolation (not instant @ 50M)
3. **Trajectory curriculum**: Staged difficulty (easy → recovery → moderate → full)
4. **Proactive monitoring**: Auto-pause on KL>0.1 or variance<0
5. **Incremental validation**: 20M checkpoint gates full 100M run

---

## Current State (Session 8g Baseline)

### Configuration (config.py)
```python
# BASE COORDINATION - KEEP THESE (Evidence-based, PROVEN)
reachability_distance_weight: float = 30.0  # Gentler penalties
reachability_hard_margin: float = 0.7       # Expanded workspace (65% FK coverage)
reachability_optimal_distance: float = 0.6  # FK median alignment

# CURRICULUM - FIX THESE (Orientation under-trained)
use_curriculum: bool = True
curriculum_stage_1_steps: int = 50_000_000  # Stage 1: 0-50M
curriculum_stage_1_position_weight: float = 5.0    # ❌ TOO LOW (50% of final)
curriculum_stage_1_orientation_weight: float = 15.0  # ❌ TOO LOW (50% of final)
curriculum_stage_2_position_weight: float = 10.0   # Stage 2: Full weight
curriculum_stage_2_orientation_weight: float = 30.0
```

### Environment Logic (env.py)
```python
# Lines 377-395: Curriculum initialization
self.use_curriculum = self.task_cfg.rewards.use_curriculum
self.curriculum_stage_1_steps = self.task_cfg.rewards.curriculum_stage_1_steps
self.current_training_step = 0
self.curriculum_stage = 1 if self.use_curriculum else 2

# Apply Stage 1 weights if curriculum enabled
if self.use_curriculum:
    self.reward_weights["position_tracking"] = stage_1_position
    self.reward_weights["orientation_tracking"] = stage_1_orientation
```

**Problems**:
- ❌ No gradual transition logic (instant switch @ 50M)
- ❌ No trajectory stage switching
- ❌ No stability monitoring (KL, variance thresholds)

### Observations (observations.py)
```python
# 78 dimensions total (KEEP)
# - Base observations: 70 dims
# - Heading cue: +2 dims (sin/cos yaw error)
# - Workspace comfort: +2 dims (comfort signal, normalized distance)
```

**Status**: ✅ Working well, keep as-is

---

## Session 8H Changes

### 1. Config.py Updates

#### Curriculum Weights (PRIMARY FIX)
```python
# ========================================
# CURRICULUM LEARNING (Session 8h - Balanced Weights)
# ========================================
use_curriculum: bool = True
curriculum_stage_1_steps: int = 45_000_000  # Stage 1: 0-45M (transition starts earlier)
curriculum_transition_steps: int = 10_000_000  # 10M gradual ramp (45M-55M)

# Stage 1: Balanced ratio from start (40% of final, maintains 1:3 ratio)
curriculum_stage_1_position_weight: float = 4.0    # Was 5.0, now 40% of 10.0
curriculum_stage_1_orientation_weight: float = 12.0  # Was 15.0, now 40% of 30.0

# Stage 2: Full precision tracking (100% weights)
curriculum_stage_2_position_weight: float = 10.0   # Unchanged
curriculum_stage_2_orientation_weight: float = 30.0  # Unchanged

# Optional Stage 2 boost if orientation still struggles @ 20M
# curriculum_stage_2_orientation_weight: float = 40.0  # Increase to 4:1 ratio if needed
```

**Rationale**:
- (4.0, 12.0) maintains 1:3 ratio → orientation gets adequate signal from start
- 40% scaling → gentler Stage 1, more room for Stage 2 improvement
- Transition starts @ 45M → longer ramp period before full weights

#### Monitoring Thresholds (NEW)
```python
# ========================================
# STABILITY MONITORING (Session 8h - Auto-pause)
# ========================================
enable_auto_pause: bool = True  # Pause training on instability
kl_threshold: float = 0.1  # Pause if KL divergence exceeds this
variance_threshold: float = 0.0  # Pause if explained variance drops below this
checkpoint_frequency: int = 2_000_000  # Save every 2M steps (finer granularity)
```

### 2. Environment Logic Updates (env.py)

#### Gradual Weight Interpolation (NEW)
```python
# Lines ~380-420: Replace instant switch with gradual transition
def _update_curriculum_stage(self):
    """Update curriculum stage with gradual weight interpolation."""
    if not self.use_curriculum or self.curriculum_stage == 2:
        return
    
    # Estimate current training step from episode buffer
    if hasattr(self, 'episode_length_buf'):
        completed_steps = int(self.episode_length_buf.sum().item())
        self.current_training_step += completed_steps
    
    stage_1_end = self.curriculum_stage_1_steps  # 45M
    transition_end = stage_1_end + self.task_cfg.rewards.curriculum_transition_steps  # 55M
    
    if self.current_training_step < stage_1_end:
        # Stage 1: Keep reduced weights
        return
    
    elif self.current_training_step < transition_end:
        # Transition: Linear interpolation 45M-55M
        if self.curriculum_stage == 1:
            print(f"[Session 8h] Starting gradual curriculum transition @ {self.current_training_step:,} steps")
            self.curriculum_stage = 1.5  # Mark as transitioning
        
        progress = (self.current_training_step - stage_1_end) / self.task_cfg.rewards.curriculum_transition_steps
        progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
        
        # Linear interpolation
        stage_1_pos = self.task_cfg.rewards.curriculum_stage_1_position_weight
        stage_1_ori = self.task_cfg.rewards.curriculum_stage_1_orientation_weight
        stage_2_pos = self.task_cfg.rewards.curriculum_stage_2_position_weight
        stage_2_ori = self.task_cfg.rewards.curriculum_stage_2_orientation_weight
        
        self.reward_weights["position_tracking"] = stage_1_pos + progress * (stage_2_pos - stage_1_pos)
        self.reward_weights["orientation_tracking"] = stage_1_ori + progress * (stage_2_ori - stage_1_ori)
        
        # Log every 1M steps during transition
        if self.current_training_step % 1_000_000 < 100_000:
            print(f"[Session 8h] Transition progress: {progress*100:.1f}% | "
                  f"pos={self.reward_weights['position_tracking']:.1f}, "
                  f"ori={self.reward_weights['orientation_tracking']:.1f}")
    
    else:
        # Stage 2: Full weights reached
        if self.curriculum_stage < 2:
            self.reward_weights["position_tracking"] = self.task_cfg.rewards.curriculum_stage_2_position_weight
            self.reward_weights["orientation_tracking"] = self.task_cfg.rewards.curriculum_stage_2_orientation_weight
            self.curriculum_stage = 2
            print(f"[Session 8h] Curriculum Stage 2 reached @ {self.current_training_step:,} steps | "
                  f"position_weight={self.reward_weights['position_tracking']}, "
                  f"orientation_weight={self.reward_weights['orientation_tracking']}")
```

#### Stability Monitoring (NEW)
```python
# Lines ~1250+: Add after _update_curriculum_stage() call
def _check_training_stability(self, kl_divergence, explained_variance):
    """Monitor training stability and pause if thresholds exceeded."""
    if not self.task_cfg.rewards.enable_auto_pause:
        return True  # Continue training
    
    if kl_divergence > self.task_cfg.rewards.kl_threshold:
        print(f"⚠️  [Session 8h] KL DIVERGENCE TOO HIGH: {kl_divergence:.4f} > {self.task_cfg.rewards.kl_threshold}")
        print(f"    Training step: {self.current_training_step:,}")
        print(f"    ACTION REQUIRED: Check TensorBoard, consider rollback to previous checkpoint")
        return False  # Signal to pause (implementation depends on training loop)
    
    if explained_variance < self.task_cfg.rewards.variance_threshold:
        print(f"⚠️  [Session 8h] EXPLAINED VARIANCE NEGATIVE: {explained_variance:.4f} < {self.task_cfg.rewards.variance_threshold}")
        print(f"    Training step: {self.current_training_step:,}")
        print(f"    ACTION REQUIRED: Value function failing, consider rollback")
        return False
    
    return True
```

### 3. Trajectory Curriculum Structure (NEW)

#### Directory Structure
```
trajectoryToLearn/
├── stage0_easy/              # 0-20M: Short, static, within arm reach
│   ├── README.md            # Stage description and selection criteria
│   ├── static_front_*.npy   # 30-60 waypoints, 0.4-0.6m reach, front sector
│   └── short_side_*.npy     # 30-60 waypoints, 0.4-0.6m reach, side sectors
│
├── stage1_recovery/          # 20-40M: Learn to approach from far
│   ├── README.md
│   ├── recovery_1.5m_*.npy  # Start 1.5m out, drive to 0.55-0.65m, static target
│   └── recovery_2.0m_*.npy  # Start 2.0m out, more challenging approach
│
├── stage2_moderate/          # 40-70M: Medium length, some behind-base
│   ├── README.md
│   ├── medium_arc_*.npy     # 60-120 waypoints, 0.5-0.8m reach, arcing trajectories
│   └── moderate_back_*.npy  # 60-120 waypoints, includes behind-base positions
│
└── stage3_full/              # 70M+: All trajectories including hard ones
    ├── README.md
    ├── long_fast_*.npy       # 120+ waypoints, 0.4-1.0m reach, fast movement
    └── full_rotation_*.npy   # Full 360° coverage, challenging orientations
```

#### Trajectory Selection Criteria

**Stage 0 (Easy) - 0-20M**:
- Length: 30-60 waypoints
- Reach: 0.4-0.6m (comfortable zone)
- Speed: <0.1 m/s (quasi-static)
- Sectors: Front and sides only (no behind-base)
- Goal: Learn basic arm tracking + workspace positioning

**Stage 1 (Recovery) - 20-40M**:
- Length: 60-90 waypoints
- Initial distance: 1.5-2.0m (far start)
- Target: Static (no movement)
- Task: Drive from far → comfortable zone (0.55-0.65m)
- Goal: Learn approach strategy, prevent "unreachable from start" failures

**Stage 2 (Moderate) - 40-70M**:
- Length: 60-120 waypoints
- Reach: 0.5-0.8m (expanded range)
- Speed: <0.15 m/s
- Sectors: 270° coverage (includes some behind-base)
- Goal: Generalize to longer trajectories, practice moderate challenges

**Stage 3 (Full) - 70M+**:
- Length: All (30-300+ waypoints)
- Reach: Full workspace (0.35-1.0m)
- Speed: Up to 0.2 m/s
- Sectors: Full 360° coverage
- Goal: Final policy refinement on hardest cases

### 4. Launcher Updates (launch_session_8h.ps1)

```powershell
# Phase definitions
$Phases = @{
    "smoke" = @{
        NumEnvs = 64
        TotalSteps = 500000  # 500K steps
        TrajectoryStage = "stage0_easy"
        Description = "Quick validation (78 dims, Stage 1 weights 4.0/12.0)"
    }
    "stage1" = @{
        NumEnvs = 16384
        TotalSteps = 20000000  # 20M steps
        TrajectoryStage = "stage0_easy"
        Description = "Stage 1 validation (check orientation improvement)"
    }
    "full" = @{
        NumEnvs = 16384
        TotalSteps = 100000000  # 100M steps
        TrajectoryStage = "auto"  # Auto-switch based on training step
        Description = "Full curriculum run with gradual transition"
    }
}

# Trajectory stage switching logic
if ($TrajectoryStage -eq "auto") {
    # Estimate current step from checkpoint (if resuming)
    # Or use stage timing from curriculum config
    # 0-20M: stage0_easy
    # 20-40M: stage1_recovery
    # 40-70M: stage2_moderate
    # 70M+: stage3_full
}

# Success criteria @ 20M checkpoint
Write-Host "20M Evaluation Thresholds:" -ForegroundColor Yellow
Write-Host "  Position error: <350cm mean" -ForegroundColor Gray
Write-Host "  Orientation error: <80° mean" -ForegroundColor Gray
Write-Host "  Workspace distance: 0.50-0.65m" -ForegroundColor Gray
Write-Host "  Unreachable %: <10%" -ForegroundColor Gray
Write-Host "  Explained variance: >0.3" -ForegroundColor Gray
Write-Host "  KL divergence: 0.01-0.05" -ForegroundColor Gray
```

---

## Implementation Sequence

### Phase 1: Preparation (2-3 hours)

#### Step 1.1: Create Trajectory Stages
```powershell
# Create directory structure
New-Item -ItemType Directory -Path "trajectoryToLearn\stage0_easy" -Force
New-Item -ItemType Directory -Path "trajectoryToLearn\stage1_recovery" -Force
New-Item -ItemType Directory -Path "trajectoryToLearn\stage2_moderate" -Force
New-Item -ItemType Directory -Path "trajectoryToLearn\stage3_full" -Force

# TODO: Populate with actual trajectories (requires MATLAB export or manual selection)
# For now, can use symbolic links or copies from world_json/
```

#### Step 1.2: Update config.py
- Change curriculum weights: (5.0, 15.0) → (4.0, 12.0)
- Add transition parameters: stage_1_steps=45M, transition_steps=10M
- Add monitoring thresholds: KL=0.1, variance=0.0

#### Step 1.3: Update env.py
- Replace instant switch with gradual interpolation (45-55M ramp)
- Add _check_training_stability() method
- Update curriculum logging

#### Step 1.4: Create launch_session_8h.ps1
- Copy from launch_session_8g.ps1
- Update phase definitions (smoke, stage1, full)
- Add trajectory stage auto-switching
- Add 20M success criteria display

#### Step 1.5: Update monitoring documentation
- MONITORING_GUIDE.md: Add Session 8h thresholds
- SESSION_8H_IMPLEMENTATION_PLAN.md: This document

### Phase 2: Smoke Test (30 mins)
```powershell
.\scripts\launch_session_8h.ps1 -Phase smoke

# Verify:
# ✓ 78 dims observation space
# ✓ Stage 1 weights active: position=4.0, orientation=12.0
# ✓ stage0_easy trajectories loading
# ✓ No crashes or errors
```

### Phase 3: 20M Validation (5 hours)
```powershell
.\scripts\launch_session_8h.ps1 -Phase stage1

# Monitor metrics:
# - Position tracking error trend
# - Orientation tracking error trend
# - Workspace distance convergence
# - KL divergence stability
# - Explained variance health

# Checkpoint every 2M steps
# Fast evaluation @ 20M: 100 episodes
```

**20M Decision Point**:

| Metric | Target | Action if Failed |
|--------|--------|------------------|
| Position error | <350cm | Check workspace distance, verify reachability |
| Orientation error | <80° | Increase Stage 2 orientation weight to 40.0 |
| Workspace distance | 0.50-0.65m | Adjust reachability penalties |
| Unreachable % | <10% | Check trajectory stage0_easy difficulty |
| Explained variance | >0.3 | Reduce learning rate or increase entropy |
| KL divergence | 0.01-0.05 | Adjust KL schedule or clip range |

### Phase 4: Full Run (14 hours) - ONLY IF 20M PASSED
```powershell
.\scripts\launch_session_8h.ps1 -Phase full

# Automatic trajectory stage switching:
# 0-20M: stage0_easy (continue from stage1 run if resuming)
# 20-40M: stage1_recovery (learn far → near approach)
# 40-70M: stage2_moderate (includes weight transition 45-55M)
# 70-100M: stage3_full (all trajectories, full weights)

# Monitor for auto-pause conditions:
# - KL > 0.1 → pause, wait for manual intervention
# - Variance < 0 → pause, suggest rollback to last 2M checkpoint
```

---

## Expected Outcomes

### Conservative Targets (90% confidence)
| Metric | Session 8g @ 40M | Session 8h @ 20M Target | Session 8h @ 100M Target |
|--------|------------------|-------------------------|--------------------------|
| Position error | 301cm (163cm median) | <350cm mean | 280-300cm mean |
| Orientation error | 130° (catastrophic) | <80° mean | 60-80° mean |
| Workspace distance | 0.554m (perfect!) | 0.50-0.65m | 0.55-0.60m |
| Unreachable % | 78% @ step 100! | <10% | <5% |
| Training stability | Collapsed @ 100M | Stable @ 20M | Stable throughout |

### Optimistic Targets (50% confidence)
| Metric | Session 8f (BEST) | Session 8h @ 100M Target |
|--------|-------------------|--------------------------|
| Position error | 308cm mean | 250-280cm (**better**!) |
| Orientation error | 46.5° mean | 45-60° (**matching**!) |
| Reward | -126k | -100k to -120k |

---

## Risk Mitigation

### Risk 1: Orientation Still Struggles @ 20M
**Symptoms**: Orientation error >100° at 20M checkpoint

**Root Cause Options**:
1. Weight 12.0 still insufficient (need more signal)
2. Observation space lacks orientation rate cues
3. Trajectory stage0_easy has poor orientation variety

**Fixes**:
```python
# Option A: Increase Stage 2 orientation weight
curriculum_stage_2_orientation_weight: float = 40.0  # Was 30.0 (4:1 ratio)

# Option B: Add orientation rate observations (+2 dims → 80 total)
# In observations.py:
yaw_rate = self.robot.data.root_ang_vel_w[:, 2:3]  # Angular velocity around Z
ori_error_rate = # Calculate rate of change of orientation error
components.extend([yaw_rate, ori_error_rate])

# Option C: Increase base orientation tracking weight
orientation_tracking: float = 250.0  # Was 200.0
```

### Risk 2: Position Degrades @ 20M
**Symptoms**: Position error >400cm at 20M (worse than 8g @ 40M)

**Root Cause Options**:
1. stage0_easy trajectories too hard (unreachable)
2. Curriculum Stage 1 weight 4.0 too low
3. Workspace drift outside 0.5-0.7m

**Fixes**:
- Verify stage0_easy trajectories all within 0.4-0.6m reach
- Increase Stage 1 position weight to 5.0 (but keep orientation at 15.0 = 3:1 ratio)
- Check workspace_distance_mean in logs (should be 0.50-0.65m)
- Reduce reachability_distance_weight to 20 (from 30) if too aggressive

### Risk 3: Training Instability (KL spike or variance drop)
**Symptoms**: KL >0.1 or explained_variance <0 triggers auto-pause

**Root Cause Options**:
1. Transition ramp 45-55M still too abrupt
2. Learning rate too high during transition
3. Entropy decay too fast

**Fixes**:
```python
# Option A: Extend transition period
curriculum_transition_steps: int = 20_000_000  # 20M ramp (40-60M) instead of 10M

# Option B: Reduce learning rate during transition
# In train.py, add learning rate schedule:
if 45M < step < 55M:
    learning_rate = 2e-4  # Was 3e-4

# Option C: Pause entropy decay during transition
# In callbacks, freeze entropy coefficient during transition window
```

### Risk 4: Trajectory Stage Switching Causes Instability
**Symptoms**: Performance drops sharply when switching trajectory stages

**Root Cause**: Policy overfitted to stage0_easy, struggles with stage1_recovery

**Fixes**:
- Add "warm-up" period: Mix 20% new stage with 80% old stage for 2M steps
- Reduce weight increases during trajectory transitions
- Evaluate policy on new stage BEFORE switching fully

---

## Success Criteria

### 20M Checkpoint (Gate for Full Run)
**ALL must pass**:
- ✅ Position error: <350cm mean
- ✅ Orientation error: <80° mean (50% improvement vs 8g's 130°)
- ✅ Workspace distance: 0.50-0.65m mean
- ✅ Unreachable %: <10% (vs 8g's 78%!)
- ✅ Explained variance: >0.3
- ✅ KL divergence: 0.01-0.05
- ✅ No auto-pause triggers during training

### 100M Final (Success Definition)
**Minimum (Must Achieve)**:
- ✅ Position error: <300cm mean (match 8g @ 40M, better than 8f)
- ✅ Orientation error: <80° mean (50% improvement vs 8g @ 40M)
- ✅ Training stable: No collapse, std <1.0 throughout
- ✅ Workspace converged: 0.55-0.60m sustained

**Stretch (Ideal Outcome)**:
- 🎯 Position error: <280cm mean (**better than 8f baseline 308cm**)
- 🎯 Orientation error: <60° mean (**approaching 8f's 46.5°**)
- 🎯 Reward: >-120k (better than 8f's -126k)
- 🎯 Curriculum smooth: No instability during 45-55M transition

---

## Rollback Plan

### If 20M Fails
1. Analyze which metric(s) failed
2. Apply appropriate fix from Risk Mitigation section
3. Rollback to last stable 10M checkpoint if available
4. Restart Stage 1 with adjusted config
5. Re-evaluate @ 20M

### If Training Becomes Unstable Mid-Run
1. Auto-pause triggered (KL>0.1 or variance<0)
2. Review TensorBoard metrics around pause point
3. Rollback to checkpoint from 2M steps before pause
4. Apply stability fix (extend transition, reduce LR, increase entropy)
5. Resume training with adjusted parameters

### If 100M Collapses Like 8g
1. **DO NOT PANIC** - We have 2M checkpoints throughout
2. Evaluate checkpoints: 40M, 60M, 80M to find last stable
3. Use best pre-collapse checkpoint for deployment
4. Analyze collapse timing vs curriculum transition
5. Design Session 8i with even gentler transition or no curriculum

---

## Files to Create/Modify

### New Files
- [ ] `trajectoryToLearn/stage0_easy/README.md`
- [ ] `trajectoryToLearn/stage1_recovery/README.md`
- [ ] `trajectoryToLearn/stage2_moderate/README.md`
- [ ] `trajectoryToLearn/stage3_full/README.md`
- [ ] `scripts/launch_session_8h.ps1`
- [ ] `docs/training_sessions/SESSION_8H_IMPLEMENTATION_PLAN.md` (this file)
- [ ] `docs/training_sessions/SESSION_8H_MONITORING_GUIDE.md`

### Modified Files
- [ ] `src/rl_platform/tasks/mobile_mm/config.py` (curriculum weights, monitoring thresholds)
- [ ] `src/rl_platform/tasks/mobile_mm/env.py` (gradual interpolation, stability checks)
- [ ] `MONITORING_GUIDE.md` (add Session 8h thresholds)
- [ ] `TRAINING_SESSIONS_MASTER_LOG.md` (add Session 8h entry)

---

## Next Steps

1. ✅ Review this plan with user for approval
2. ⏳ Create trajectory stage directories (or identify suitable existing trajectories)
3. ⏳ Implement config.py changes
4. ⏳ Implement env.py gradual interpolation
5. ⏳ Create launch_session_8h.ps1
6. ⏳ Run smoke test
7. ⏳ Run 20M validation
8. ⏳ Evaluate @ 20M and decide on full run

**Estimated Timeline**:
- Preparation: 2-3 hours
- Smoke test: 30 minutes
- 20M validation: ~5 hours
- Full 100M run (if approved): ~14 hours
- **Total**: ~22 hours from start to finish

---

## References

- Session 8g evaluation: `docs/training_sessions/SESSION_8G_EVALUATION_RESULTS.md`
- Session 8f baseline: `evaluation_plots/session_8f_200M/`
- FK workspace analysis: `matlab/exports/reach_surface.mat` (1,677 points, 0.594m median)
- Current config: `src/rl_platform/tasks/mobile_mm/config.py` (Session 8g state)
