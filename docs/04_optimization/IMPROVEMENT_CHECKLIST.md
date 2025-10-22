# Trajectory Tracking Improvements - Quick Checklist

**Date:** October 21, 2025  
**Current Status:** Training Session 4b at 25.7M/100M steps (8192 envs)

---

## 🎯 Priority 1: CRITICAL FIXES (DO NOW)

### [ ] Implement Grace Period for Termination

**Problem:** ~1% envs stuck in infinite reset loops (terminate immediately from broken spawns)

**Solution:** Add grace period - only terminate after N consecutive high-error steps

**Files to modify:**
- [ ] `src/rl_platform/tasks/mobile_mm/config.py`
  ```python
  tracking_error_grace_period: int = 10  # consecutive steps @ 20Hz = 0.5s
  ```

- [ ] `src/rl_platform/tasks/mobile_mm/env.py`
  - [ ] Add buffer in `_initialize_buffers()`:
    ```python
    self.high_error_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    ```
  - [ ] Modify `_get_dones()` logic (lines ~1140-1185)
  - [ ] Reset counter in `_reset_idx()` (lines ~1220-1245)

**Testing:**
- [ ] Test with 256 envs, 10 minutes
- [ ] Verify no `[RESET]` spam in console
- [ ] Check "Broken (>2.0m)" percentage in display < 0.05%

**Expected outcome:** Infinite reset loops eliminated

---

## 🎯 Priority 2: STRATEGIC ENHANCEMENTS (AFTER PHASE 1)

### [ ] Add Velocity Tracking Rewards

**Problem:** Policy only cares about position, not trajectory timing

**Solution:** Add velocity and optionally acceleration tracking rewards

**Files to modify:**
- [ ] `src/rl_platform/tasks/mobile_mm/trajectories.py`
  - [ ] Add `get_target_velocity()` method
  - [ ] Compute from waypoint spacing: `(next_pos - curr_pos) / waypoint_dt`

- [ ] `src/rl_platform/tasks/mobile_mm/rewards.py`
  - [ ] Add `velocity_tracking_reward()` function
  - [ ] Optionally add `acceleration_tracking_reward()`

- [ ] `src/rl_platform/tasks/mobile_mm/env.py`
  - [ ] Call `trajectory_manager.get_target_velocity()` in `_get_rewards()`
  - [ ] Compute velocity reward term
  - [ ] Add to total reward

- [ ] `src/rl_platform/tasks/mobile_mm/config.py`
  ```python
  velocity_tracking: float = 0.3  # Start conservative
  ```

**Testing:**
- [ ] Test with 256 envs, compare metrics vs position-only
- [ ] Tune weight: try 0.1, 0.2, 0.3
- [ ] Full retrain 100M steps with 8192 envs

**Expected outcome:** Tighter trajectory timing adherence

⚠️ **Note:** Requires restart from 0 (architectural change)

---

## 💡 Priority 3: OPTIONAL ENHANCEMENTS (IF NEEDED)

### [ ] Increase Lookahead Horizon

**Current:** 3 waypoints (0.3s)  
**Proposed:** 5 waypoints (0.5s)

**File to modify:**
- [ ] `src/rl_platform/tasks/mobile_mm/config.py`
  ```python
  lookahead_steps: int = 5  # Up from 3
  ```

**Tradeoff:** +6 dims observation space (15 instead of 9)

---

### [ ] Add Trajectory Phase Information

**Idea:** Tell policy where in trajectory it is (start/middle/end)

**File to modify:**
- [ ] `src/rl_platform/tasks/mobile_mm/observations.py`
  ```python
  trajectory_phase = current_waypoint_idx / total_waypoints
  components.append(trajectory_phase.unsqueeze(-1))  # +1 dim
  ```

**Tradeoff:** Only works for fixed-length trajectories

---

### [ ] Spatial Progress Rewards

**Idea:** Reward advancing through waypoints, not just tracking error

**File to modify:**
- [ ] `src/rl_platform/tasks/mobile_mm/rewards.py`
  ```python
  def trajectory_progress_reward(prev_idx, curr_idx):
      return (curr_idx - prev_idx).float()
  ```

**Tradeoff:** May conflict with position tracking (rushing ahead)

---

## 📊 Validation Metrics

### After Phase 1 (Grace Period):
- [ ] "Broken (>2.0m)" envs < 0.05% (down from 0.1-1%)
- [ ] No `[RESET]` message spam (should see <5 resets per 50 steps)
- [ ] Explained variance maintained > 0.95
- [ ] FPS drop < 5% (extra logic overhead)

### After Phase 2 (Velocity Rewards):
- [ ] EE velocity error < 0.05 m/s average
- [ ] Trajectory timing lag < 1 timestep average
- [ ] Final position accuracy maintained (no tradeoff)
- [ ] Explained variance > 0.92 (may start lower, should converge)

---

## 📝 Documentation Updates

### After implementing changes:
- [ ] Update `README.md` with new config parameters
- [ ] Update `docs/reference/reward_cheatsheet.md` if adding rewards
- [ ] Update `docs/workflows/multi_trajectory_training.md` with best practices
- [ ] Document termination policy changes in architecture overview

---

## 🚀 Deployment Sequence

### Phase 1 (TODAY):
1. [x] Create improvement analysis doc (`TRAJECTORY_TRACKING_IMPROVEMENTS.md`)
2. [x] Create this checklist
3. [ ] Implement grace period (1-2 hours)
4. [ ] Test with 256 envs (10 minutes)
5. [ ] Restart training 8192 envs (continue from 25.7M OR restart)
6. [ ] Monitor for 1-2 hours, verify fixes

### Phase 2 (AFTER PHASE 1 SUCCESS):
1. [ ] Implement velocity tracking (1-2 hours)
2. [ ] Test with 256 envs, tune weights (30 minutes)
3. [ ] Retrain from scratch 100M steps (~9 hours)
4. [ ] Compare results with baseline (position-only)
5. [ ] Document findings

### Phase 3 (OPTIONAL):
1. [ ] Decide based on Phase 2 results
2. [ ] Implement selected enhancements
3. [ ] Retrain and compare

---

## 🔧 Quick Commands

### Test small run (256 envs):
```powershell
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 256 -Headless -TotalTimesteps 1000000
```

### Full training (8192 envs):
```powershell
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 8192 -Headless
```

### Check current training status:
```powershell
# Monitor log file
Get-Content logs\sb3\MobileMMTrackEE-v0\* -Tail 50 -Wait
```

### Git commit after changes:
```powershell
git add -A
git commit -m "Implement grace period for trajectory tracking termination"
git push origin train-windows
```

---

## ✅ Success Criteria

**Phase 1 Success = Can proceed to Phase 2:**
- ✅ Zero infinite reset loops
- ✅ <0.05% broken envs
- ✅ Training metrics maintained (explained variance >0.95)
- ✅ No console spam

**Phase 2 Success = Architecture validated:**
- ✅ Velocity error < 0.05 m/s
- ✅ Position accuracy maintained
- ✅ Training converges (explained variance >0.92)
- ✅ FPS acceptable (>2500 with 8192 envs)

**Overall Success = Ready for deployment:**
- ✅ Robust trajectory tracking
- ✅ Temporal timing adherence
- ✅ Stable training dynamics
- ✅ Generalizes to new trajectories

---

**Last Updated:** October 21, 2025  
**Next Review:** After Phase 1 implementation
