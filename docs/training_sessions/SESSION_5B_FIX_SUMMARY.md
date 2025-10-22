# Training Session 5b: Critical Reward Fixes

**Date**: 2025-01-XX  
**Previous Session**: Session 5 (CATASTROPHIC FAILURE at 20M steps)  
**Status**: Ready to launch with fixes

---

## 🚨 What Went Wrong in Session 5?

### Initial Success (10M steps):
- ✅ Base actions ACTIVE (PPR offsets 0.006-0.112m)
- ✅ base_mobilization_reward reaching +2.43
- ✅ Mean tracking error: 0.877m (62% better!)
- ✅ **OBSERVATION SPACE FIX WORKED!**

### Catastrophic Failure (20M steps):
- ❌ 51.5%→63.5% environments BROKEN (>2m error)
- ❌ Mean tracking error DOUBLED: 0.877m → 2.242m
- ❌ Base moving WILDLY: 5-10 meters (way beyond capability!)
- ❌ base_mobilization_reward: -21.31 to +10.26 (EXPLODED!)
- ❌ Constant reset spam (100+ resets per tracking window)
- ❌ Wild arm motion: 8.67 rad/s (limit is 2.0 rad/s)

### Root Cause:
**UNBOUNDED base_mobilization_reward** causing **REWARD HACKING**:

```python
# BROKEN (Session 5):
progress = dist_if_static - dist_current  # NO LIMIT!
reward = 150.0 * progress

# If base moves 5 meters:
# reward = 150 × 5.0 = 750 points!!!
# (vs position_tracking max = 50 points)

# Policy learned WRONG lesson:
# "More base movement = more rewards!" (CATASTROPHIC!)
```

---

## ✅ Session 5b Fixes (3 Critical Changes)

### Fix #1: Cap Base Mobilization Progress (Priority 1 - CRITICAL)

**File**: `src/rl_platform/tasks/mobile_mm/rewards.py` (line ~91-94)

```python
# Positive when the chassis actually moved closer to the target.
progress = dist_if_static - dist_current

# CRITICAL FIX (Session 5b): Cap progress to prevent reward explosion!
# Max 20cm progress per step → max reward = 150 × 0.2 = 30 points
# (reasonable compared to position_tracking max = 50 points)
progress = torch.clamp(progress, min=0.0, max=0.2)
```

**Why This Works**:
- Max reward: 150 × 0.2 = **30 points** (reasonable!)
- Typical 7.5cm movement: 150 × 0.075 = **11.25 points**
- Prevents: 5m movement → 750 point explosion
- Policy learns: "Move JUST ENOUGH, not as much as possible"

---

### Fix #2: Add Excessive Movement Penalty (Priority 2 - IMPORTANT)

**File**: `src/rl_platform/tasks/mobile_mm/rewards.py` (new function, lines ~175-200)

```python
def excessive_base_movement_penalty(
    base_pos: torch.Tensor,
    prev_base_pos: torch.Tensor,
    threshold: float = 0.1,  # 10cm per step
    scale: float = 10.0,
) -> torch.Tensor:
    """Heavily penalize excessive base movements to prevent wild behavior.
    
    Example: If base moves 1 meter in one step:
        excess = 1.0 - 0.1 = 0.9 meters
        penalty = 10.0 × 0.9 = 9.0 points
    """
    movement = torch.norm(base_pos[:, :2] - prev_base_pos[:, :2], dim=-1)
    excess = torch.clamp(movement - threshold, min=0.0)
    return scale * excess
```

**Why This Works**:
- Movements ≤10cm: **No penalty** (encourages reasonable exploration)
- Movements >10cm: **Heavily penalized** (discourages wild behavior)
- Example: 1m movement → -9.0 points (strong discouragement)
- Prevents policy from exploiting reward loopholes

---

### Fix #3: Increase Distance Penalty (Priority 3 - BALANCING)

**File**: `src/rl_platform/tasks/mobile_mm/config.py` (line ~80)

```python
target_distance_penalty: float = 5.0  # Up from 3.0
```

**Why This Works**:
- Compensates for capped mobilization reward (now max 30 instead of unbounded)
- Maintains pressure to move base when target is out of reach
- Balances: "Move when needed" vs "Don't move excessively"

---

## 📊 Expected Session 5b Behavior

### Base Movement Characteristics:
- **Typical movements**: 5-15cm per step (reasonable for differential drive)
- **Max allowed without penalty**: 10cm
- **base_mobilization_reward range**: -5 to +30 (bounded!)
- **Excessive penalty**: 0 for ≤10cm, scales linearly for >10cm

### Reward Structure (per step):
```
Position tracking:        0 to +50 points (exponential decay)
Base mobilization:       -5 to +30 points (CAPPED at 20cm progress!)
Distance penalty:         0 to -10 points (out of reach penalty)
Excessive movement:       0 to -X points (wild movements penalized)
Action penalties:        -0.5 to -2 points (typical)
```

### Expected Training Progression:

**1M steps** (5 minutes):
- [ ] Base actions active but CONTROLLED (<0.15m movements)
- [ ] base_mobilization_reward: -5 to +30 range (no explosions!)
- [ ] No wild movements (>0.5m)
- [ ] Broken envs < 1%
- **IF FAIL**: Rewards still too strong, need further reduction

**5M steps** (30 minutes):
- [ ] Mean tracking error < 1.0m (improving steadily)
- [ ] Base mobilization mean > 0.0 (net positive strategy!)
- [ ] Broken envs < 2%
- [ ] No reset spam
- **IF FAIL**: Stop and re-analyze reward balance

**10M steps** (1 hour):
- [ ] Mean tracking error < 0.7m
- [ ] Base movements coordinated (5-15cm typical, rarely >20cm)
- [ ] "Good" tracking category > 5%
- [ ] Explained variance 0.80-0.85

**30M steps** (3 hours):
- [ ] Mean tracking error < 0.4m
- [ ] Base mobilization consistent, effective strategy
- [ ] "Good" tracking category > 15%
- [ ] Few excessive movement penalties (policy learned bounds)

**100M steps** (9 hours - FINAL):
- [ ] Mean tracking error < 0.25m (TARGET!)
- [ ] Base actively mobilizes for >70% OOR targets
- [ ] Total reward > 35 (consistently positive!)
- [ ] Explained variance > 0.92
- [ ] Policy respects 10cm movement limit naturally

---

## 🐛 Second Bug Discovered: Base Spawning Diversity

### User Observation:
> "it seems to me in many environments base is spawn at the start point"

### Evidence from Session 5:
```
[RESET] Env 1: Base moved to trajectory start [1.050, 0.080, 0.860]
[RESET] Env 42: Base moved to trajectory start [1.050, 0.080, 0.860]
[RESET] Env 127: Base moved to trajectory start [1.050, 0.080, 0.860]
(repeated for many environments!)
```

### Root Cause:
In `src/rl_platform/tasks/mobile_mm/env.py` (_reset_idx, lines ~1202-1210):

```python
# Reset robot base position to match trajectory starting point
first_target_pos, _ = self.trajectory_manager.get_target_pose()

# Set base position to match trajectory XY
new_root_state[:, 0] = first_target_pos[env_ids, 0]  # X position
new_root_state[:, 1] = first_target_pos[env_ids, 1]  # Y position
```

**Problem**: Many trajectories start at the SAME position (common in recorded data). This leads to:
- Many envs spawning at identical positions
- Lack of initial condition diversity
- Some envs (like 8093) consistently stuck in catastrophic failures
- Harder for policy to generalize

### Solution (Future Session 5c?):
1. **Add positional randomization**:
   ```python
   # Add random offset to initial base position
   random_offset = torch.randn(len(env_ids), 2, device=self.device) * 0.5  # ±50cm
   new_root_state[:, 0] += random_offset[:, 0]
   new_root_state[:, 1] += random_offset[:, 1]
   ```

2. **Trajectory-specific initial positions**:
   - Sample random waypoint from trajectory (not always start)
   - Position base near that waypoint instead
   
3. **Increased spawn area**:
   - Define spawn region (e.g., 5m × 5m)
   - Randomly place base within region
   - Trajectory waypoints relative to base position

**Note**: Not fixing in Session 5b to isolate reward fixes. Will address if Session 5b succeeds but shows diversity issues.

---

## 🚀 Launch Command (Session 5b)

```powershell
cd I:\isaaclab
I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 8192 --n_steps 64 --batch_size 512 --total_timesteps 100000000 --learning_rate 3e-4 --ent_coef 0.001 --enable_entropy_decay --final_ent_coef 1e-4 --decay_start_timestep 50000000 --decay_duration_timesteps 50000000 --enable_kl_schedule --kl_warmup 0.25 --kl_main 0.15 --kl_finetune 0.07 --target_kl 1.0 --trajectory_type multi_recorded --use_all_trajectories --headless
```

**Configuration**:
- **Environments**: 8192 (same as Session 5)
- **Total timesteps**: 100M (same as Session 5)
- **Hyperparameters**: UNCHANGED (isolate reward fixes)
- **Only change**: Fixed reward functions (3 fixes above)

---

## 📝 Monitoring Strategy

### First 5 Minutes (1M steps) - CRITICAL CHECKPOINT:
```
Watch for:
- Base movements: Should be <0.15m (NOT >1m like Session 5!)
- base_mobilization_reward: Should be -5 to +30 range (NOT -21 to +10!)
- Excessive penalty: Should be near 0 (movements within bounds)
- Broken envs: Should be <1% (NOT 51.5%!)

If ANY of these fail → STOP IMMEDIATELY and reduce rewards further
```

### Every 5M Steps:
```
Log to SESSION_5B_LAUNCH_LOG.md:
- Environment health (poor/good/excellent/broken percentages)
- Mean tracking error (should improve steadily)
- base_mobilization_reward range (should stay bounded)
- Base movement statistics (mean, max, std)
- Excessive movement penalty (should decrease over time)
- Any warning signs (reset spam, wild movements, etc.)
```

### Key Metrics to Track:
1. **Mean tracking error**: Should decrease monotonically
2. **base_mobilization_reward**: Should stay in [-5, +30] range
3. **Base movement magnitudes**: Should stay <0.20m (mostly <0.15m)
4. **Excessive penalty**: Should be near 0 (policy respects bounds)
5. **Broken envs**: Should stay <5% throughout training
6. **Reset frequency**: Should be low (no spam like Session 5)

---

## 🎯 Success Criteria (Session 5b)

### Minimum Viable Success:
- Training completes 100M steps without catastrophic failure
- Mean tracking error < 0.5m at 100M steps
- <10% broken environments at any point
- Base movements stay <0.30m (mostly <0.15m)
- Total reward > 20 (positive!)

### Target Success:
- Mean tracking error < 0.25m at 100M steps
- <5% broken environments throughout
- Base movements 5-15cm typical, rarely >20cm
- Total reward > 35 (consistently positive!)
- Policy naturally respects 10cm movement limit

### Stretch Goal:
- Mean tracking error < 0.15m at 100M steps
- <2% broken environments
- "Good" tracking (error <0.2m) > 20%
- Base mobilization efficient, coordinated
- Policy demonstrates emergent strategies (e.g., moving to optimal positions)

---

## 📚 Lessons Learned from Session 5

### Critical Insights:
1. **Always bound rewards**: Unbounded rewards = reward hacking opportunities
2. **Early success ≠ long-term success**: 10M steps looked great, 20M catastrophic
3. **Value function lags reality**: High explained_variance (0.927) while 63.5% broken!
4. **User observations are gold**: "base is spawn at the start point" spotted second bug
5. **Multi-stage monitoring essential**: Can't assume continued improvement

### Design Principles (for future sessions):
- **Reward capping**: ALWAYS cap progress-based rewards to reasonable maximums
- **Penalty for excess**: Add penalties for behaviors beyond physical limits
- **Balancing**: Ensure reward magnitudes are comparable across all terms
- **Diversity**: Randomize initial conditions to prevent overfitting
- **Monitoring**: Check EVERY major training milestone, not just endpoints

### What We Proved:
- ✅ Observation space fix WORKS (base actions became active!)
- ✅ Trajectory analysis correct (83.8% out of reach, base needed!)
- ✅ Expected reward math correct (+15.7 for 7.5cm movement)
- ❌ Reward formula needs bounds (unbounded = catastrophic!)

---

## 🔄 If Session 5b Also Fails...

### Fallback Options (ordered by severity):

**Option 1: Further reduce base_progress_reward**
- Reduce scale from 150.0 to 100.0 or 75.0
- Keep capping at 0.2m progress
- Keep excessive penalty at 10.0

**Option 2: Tighten excessive penalty threshold**
- Reduce threshold from 0.1m to 0.05m (5cm)
- Increase penalty scale from 10.0 to 15.0
- More aggressive discouragement

**Option 3: Curriculum learning**
- Start with targets always within reach (base doesn't need to move)
- Gradually increase out-of-reach percentage over time
- Policy learns coordination first, mobilization second

**Option 4: Shaped reward (direction-based)**
- Only reward base movement in CORRECT direction (toward target)
- Heavily penalize movement away from target
- More explicit gradient for policy

**Option 5: Hard-code base movement limits**
- Add environment-level constraint: max 0.15m per step
- Clip base actions before applying to robot
- Treat as physical limit, not reward-based

---

## 🎓 Summary: From Session 5 Disaster to Session 5b Hope

**What We Fixed**:
1. ✅ Capped base_mobilization progress to max 0.2m per step
2. ✅ Added excessive_base_movement_penalty (10.0 × excess beyond 0.1m)
3. ✅ Increased target_distance_penalty from 3.0 to 5.0

**Why Session 5b Will Succeed**:
- Policy CAN'T exploit unbounded rewards anymore (capped at 30 points!)
- Excessive movements WILL BE penalized (9 points per excessive meter!)
- Balance maintained (distance penalty compensates for capped reward)

**Expected Behavior**:
- Base moves 5-15cm per step (reasonable, controlled)
- Mean error improves steadily without catastrophic failures
- Training completes 100M steps successfully
- Policy learns "move JUST ENOUGH to reach target"

**Confidence Level**: HIGH (fixes address root cause directly)

**Next Steps**: Launch Session 5b, monitor first 5 minutes CRITICALLY!

---

**Status**: Ready for Session 5b launch! 🚀
