# Session 8g Implementation Proposal (REVISED)

**Date:** November 1, 2025  
**Goal:** Build on Session 8f's structural fixes with better workspace geometry and curriculum  
**Strategy:** Expand comfort zone + Improve observations + Curriculum staging  
**Anti-Pattern:** Do NOT increase penalty weights (that caused 8d stall!)

---

## 🎯 Target Performance (Conservative, Evidence-Based)

| Metric | Current Best | Session 8g Target | Rationale |
|--------|--------------|-------------------|-----------|
| **Position Error** | 307.8cm (8f) | **≤280cm** | Incremental via curriculum |
| **Orientation Error** | 46.5° (8f) | **≤42°** | Improve via better base positioning |
| **Mean Reward** | -126k (8f) | **≥-100k** | Natural outcome of better tracking |
| **Reachability Bonus** | 0.64 (8f) | **≥2.0** | Via expanded comfort zone |
| **Workspace Distance** | 0.42-0.60m (8f) | **0.50-0.70m stable** | Match FK workspace reality |

---

## 🔬 Analysis Summary: What ACTUALLY Works

### ❌ What DOESN'T Work (Stop Doing!)

**1. Heavy Reachability Penalties (Sessions 8c/8d)**
```
Problem: reachability_maintenance_reward = 80
Result: Policy stalls at 3m, huge negative slope overwhelms learning
Evidence: 8d stuck at 311cm with no improvement
```

**2. Bell-Shaped Rewards (Session 8e)**
```
Problem: Narrow Gaussian peak (sigma=0.15m at 0.5m)
Result: Policy gives up, catastrophic degradation 50M→100M
Evidence: 349cm→408cm position, reachability collapsed 0.79→0.58
```

**3. Tight Workspace Constraints (8f's 0.45-0.55m)**
```
Problem: Optimal zone too narrow for FK workspace reality
Evidence: PLY export shows significant reachable volume at 0.6-0.8m
Result: Unnecessary penalty pressure, workspace drift 0.42→0.60m
```

### ✅ What DOES Work (Build On This!)

**1. Session 8f Structural Fixes**
```
✅ Atomic root state write: Eliminates velocity/pose conflicts
✅ Heading cue observations: +2 dims for turn direction
✅ Distance-gated penalties: Far mode (mobilize) vs near mode (polish)
✅ Two-zone linear: Simpler than bell, more stable learning signal
```

**2. Lower Penalty Regime (8f's approach)**
```
✅ reachability_maintenance_reward = 40 (NOT 80!)
Evidence: 8f achieved 307cm (best position) with gentler penalties
Result: Base actually moved inward from starting cramped position
```

**3. Structural Changes Over Weight Tweaks**
```
✅ Heading cue helped orientation more than weight increases
✅ Distance gating enabled mobilization without fighting penalties
Evidence: 8f mobilization 0.32 vs 8e's 0.13 (+146%)
```

### 🧩 The Real Orientation Mystery

**Session 8e achieved 34.0° NOT because of reward weights but because:**
1. Policy gave up on position (408cm failure)
2. All learning capacity redirected to achievable sub-task (orientation)
3. This is a **failure mode**, not a success pattern to replicate!

**The Right Way to Improve Orientation:**
- Get base consistently in reach (0.5-0.7m)
- Improve heading cue observations (maybe add angular velocity?)
- Curriculum: Easy orientations → Hard orientations
- NOT: Crank orientation weight while stuck at 3m away

---

## 🛠️ Session 8g Implementation Plan

### Core Strategy: **Expand Workspace Geometry + Curriculum Learning**

**Principle:** Match FK workspace reality (0.5-0.8m reachable) and stage difficulty

---

## 📝 Changes from Session 8f

### 1. ✅ Keep All Session 8f Fixes (PROVEN)

**No changes needed - these work!**
- ✅ Atomic root state write (13-element tensor)
- ✅ Distance-gated penalties (sigmoid gating)
- ✅ Heading cue observations (+2 dims)
- ✅ Two-zone linear reachability structure
- ✅ **reachability_maintenance_reward = 40** (DO NOT INCREASE!)

---

### 2. 🔧 Expand Workspace Comfort Zone (Match FK Reality)

**File:** `src/rl_platform/tasks/mobile_mm/rewards.py`

**Current (Session 8f) - TOO TIGHT:**
```python
# Two-zone linear reachability
approach_zone: 0.35-0.45m  (ramp up)
optimal_zone:  0.45-0.55m  (plateau, max bonus)
decay_zone:    0.55-0.90m  (ramp down)
```

**Change to - MATCH FK WORKSPACE:**
```python
# Two-zone linear reachability (EXPANDED)
approach_zone: 0.35-0.50m  (ramp up)
optimal_zone:  0.50-0.75m  (plateau, max bonus) ← Expanded from 0.10m to 0.25m!
decay_zone:    0.75-0.95m  (ramp down)
```

**Implementation:**
```python
# In rewards.py, compute_reachability_maintenance_reward()

# Approach zone: 0.35-0.50m (wider ramp)
if base_target_dist < 0.35:
    bonus = 0.0
elif base_target_dist < 0.50:  # Was: 0.45
    bonus = (base_target_dist - 0.35) / 0.15  # Was: 0.10, now 0.15

# OPTIMAL ZONE: 0.50-0.75m (much wider! matches FK reality)
elif base_target_dist < 0.75:  # Was: 0.55
    bonus = 1.0  # Full bonus across 25cm range vs previous 10cm

# Decay zone: 0.75-0.95m (gentle slope)
elif base_target_dist < 0.95:  # Was: 0.90
    bonus = 1.0 - (base_target_dist - 0.75) / 0.20  # Was: 0.35/0.30
else:
    bonus = 0.0
```

**Rationale:**
- **Evidence from PLY export:** Significant reachable volume exists at 0.6-0.8m
- **8f drift pattern:** Policy naturally moved 0.42→0.60m seeking better reach
- **Penalty pressure:** Tight 0.45-0.55m zone fights natural FK geometry
- **New 0.50-0.75m zone:** Gives policy 25cm comfort (vs 10cm), reduces conflict

**Expected Impact:**
- Policy can stay in reach without fighting narrow margins
- Workspace distance stabilizes naturally within 0.50-0.70m
- Reduces penalty noise, cleaner position tracking signal
- Reachability bonus should increase naturally (0.64 → 2.0+)

---

### 3. � Adjust Distance Gate to Match New Comfort Zone

**File:** `src/rl_platform/tasks/mobile_mm/rewards.py`

**Current (Session 8f):**
```python
# Distance gating at 0.55m
gate = torch.sigmoid((0.55 - base_target_distance) * 10.0)
```

**Change to:**
```python
# Distance gating at 0.65m (center of new 0.50-0.75m optimal zone)
gate = torch.sigmoid((0.65 - base_target_distance) * 10.0)
```

**Rationale:**
- Align gate with NEW optimal zone center: (0.50 + 0.75) / 2 = 0.625m ≈ 0.65m
- Far mode (>0.65m): Allow mobilization, penalties OFF
- Near mode (<0.65m): Polish tracking, penalties ON
- Prevents premature penalty activation during approach

**Expected Impact:**
- Smoother approach phase without fighting penalties
- Polish phase activates when actually in good reach
- Better separation between mobilization and refinement

---

### 4. 🆕 Add Workspace Distance Observations

**File:** `src/rl_platform/tasks/mobile_mm/observations.py`

**Add new observation features (+2 dims):**
```python
# After existing heading cue (around line 94)

# Workspace comfort: How well-positioned is base? (0=bad, 1=optimal)
optimal_center = 0.625  # Center of 0.50-0.75m zone
optimal_radius = 0.125  # Half-width of zone
workspace_comfort = torch.clamp(
    1.0 - torch.abs(base_target_distance - optimal_center) / optimal_radius,
    0.0, 1.0
)
obs.append(workspace_comfort.unsqueeze(-1))  # +1 dim

# Direction to optimal zone (signed distance)
dist_to_optimal = torch.where(
    base_target_distance < 0.50,
    0.50 - base_target_distance,  # Negative: too close, back up
    torch.where(
        base_target_distance > 0.75,
        base_target_distance - 0.75,  # Positive: too far, approach
        torch.zeros_like(base_target_distance)  # Zero: in zone!
    )
)
obs.append(dist_to_optimal.unsqueeze(-1))  # +1 dim
```

**Update dimension count:**
```python
# Line ~209
dim += 8  # Was 6 (heading cue), now 8 (heading + workspace feedback)
# Total obs: 76 → 78 dims
```

**Rationale:**
- **Explicit workspace signal:** Policy knows "am I in good reach?"
- **workspace_comfort:** 1.0 in optimal zone, fades to 0.0 outside
- **dist_to_optimal:** Direction/magnitude to move (negative=retreat, positive=approach)
- Reduces trial-and-error in workspace positioning

**Expected Impact:**
- Faster learning of optimal workspace (Stage 1 curriculum benefit)
- Policy learns "get in range FIRST, then track" strategy
- Reduces workspace drift (explicit feedback loop)

---

### 5. 🎯 Curriculum Learning: Two-Stage Training

**File:** `src/rl_platform/tasks/mobile_mm/config.py`

**Add curriculum configuration:**
```python
# Curriculum: Stage difficulty progression
use_curriculum: bool = True
curriculum_stage_1_steps: int = 50_000_000  # 50M steps

# Stage 1: Learn workspace positioning
# Reduce position/orientation demands, focus on "get in range"
curriculum_stage_1_weights: dict = {
    "position_tracking": 5.0,        # 50% of final (was 10.0)
    "orientation_tracking": 15.0,    # 50% of final (was 30.0)
    "reachability_maintenance": 40.0, # Same (DO NOT CHANGE!)
}

# Stage 2: Full precision tracking (after 50M)
curriculum_stage_2_weights: dict = {
    "position_tracking": 10.0,       # Full weight
    "orientation_tracking": 30.0,    # Full weight
    "reachability_maintenance": 40.0, # Same
}
```

**Implementation in rewards.py:**
```python
def compute_rewards(self, current_step):
    # Curriculum stage selection
    if self.cfg.use_curriculum and current_step < self.cfg.curriculum_stage_1_steps:
        weights = self.cfg.curriculum_stage_1_weights
        stage = 1
    else:
        weights = self.cfg.curriculum_stage_2_weights
        stage = 2
    
    # Apply stage-specific weights
    position_reward = position_tracking_reward * weights["position_tracking"]
    orientation_reward = orientation_tracking_reward * weights["orientation_tracking"]
    reachability_reward = reachability_bonus * weights["reachability_maintenance"]
    
    # ... rest of reward computation
```

**Rationale:**
- **Stage 1 (0-50M):** "Learn to position base in 0.50-0.75m zone"
  - Lower position/orientation weights reduce noise
  - Policy focuses on workspace geometry
  - Builds foundation before precision demands
- **Stage 2 (50M-100M):** "Now do precision tracking"
  - Full weights restored
  - Policy refines on top of good workspace habits
- **Evidence:** 8e degraded 50M→100M trying to do everything at once

**Expected Impact:**
- Faster initial convergence (clear sub-goal)
- More stable training (incremental complexity)
- Better final performance (strong foundation)
- Natural transition at 50M checkpoint

---

### 6. 🆕 Workspace Distance Monitoring Callback

**File:** `scripts/reinforcement_learning/sb3/train.py`

**Add monitoring callback:**
```python
class WorkspaceDistanceMonitor(BaseCallback):
    """Monitor workspace distance, alert on drift."""
    def __init__(self, target_min=0.50, target_max=0.75, alert_steps=10000, verbose=0):
        super().__init__(verbose)
        self.target_min = target_min
        self.target_max = target_max
        self.alert_steps = alert_steps
        self.outside_target_count = 0
        
    def _on_step(self) -> bool:
        if hasattr(self.training_env, 'get_attr'):
            envs = self.training_env.get_attr('unwrapped')
            if envs and hasattr(envs[0], 'base_target_distance'):
                distances = [env.base_target_distance.mean().item() for env in envs]
                mean_dist = np.mean(distances)
                
                # Check drift
                if mean_dist < self.target_min or mean_dist > self.target_max:
                    self.outside_target_count += 1
                    if self.outside_target_count >= self.alert_steps:
                        self.logger.record("monitor/workspace_drift_alert", 1.0)
                        print(f"⚠️  Workspace drift: {mean_dist:.3f}m (target: {self.target_min}-{self.target_max}m)")
                else:
                    self.outside_target_count = 0
                
                self.logger.record("monitor/workspace_distance", mean_dist)
        return True

# Add to callbacks
callbacks = [
    # ... existing ...
    WorkspaceDistanceMonitor(target_min=0.50, target_max=0.75, alert_steps=10000),
]
```

---

## 📊 Expected Session 8g Results (REVISED - Conservative)

### Realistic Scenario (Evidence-Based)
```
Position Error:    290cm  (-6% vs 8f's 308cm) ← Incremental improvement
Orientation Error:  43°   (-8% vs 8f's 46.5°) ← Better base positioning helps
Mean Reward:      -110k   (+13% vs 8f's -126k)
Reachability:      2.5    (+291% vs 8f's 0.64) ← Wider zone increases naturally
Workspace:         0.55-0.70m stable ← Natural settling in expanded zone
```

### Optimistic Scenario (If curriculum works well)
```
Position Error:    275cm  (-11% vs 8f)
Orientation Error:  40°   (-14% vs 8f)
Mean Reward:       -95k   (+25% vs 8f)
Reachability:      3.5    (+447% vs 8f)
Workspace:         0.50-0.65m stable
```

### Minimum Viable (Conservative Fallback)
```
Position Error:    305cm  (-1% vs 8f, acceptable)
Orientation Error:  45°   (-3% vs 8f, slight improvement)
Mean Reward:      -120k   (+5% vs 8f)
Reachability:      1.5    (+134% vs 8f, still better)
Workspace:         0.50-0.75m (stays in zone)
```

**Why conservative?** No magic bullets—improvements come from better structure, not weight fiddling.

---

## 🚀 Implementation Checklist (REVISED)

### Phase 1: Code Changes
- [ ] Update `rewards.py`: Expand optimal zone 0.45-0.55m → 0.50-0.75m
- [ ] Update `rewards.py`: Adjust gate threshold 0.55m → 0.65m
- [ ] Update `observations.py`: Add workspace feedback (+2 dims, total 78)
- [ ] Update `config.py`: Add curriculum configuration
- [ ] Update `rewards.py`: Implement curriculum weight switching
- [ ] Add `WorkspaceDistanceMonitor` callback in `train.py`
- [ ] **KEEP reachability_maintenance_reward = 40** (do NOT increase!)

### Phase 2: Launcher Script
- [ ] Create `scripts/launch_session_8g.ps1`
- [ ] Task: MobileMMTrackEE-v0
- [ ] Num_envs: 16384
- [ ] Total_timesteps: 100M
- [ ] Headless: true
- [ ] Enable curriculum flag

### Phase 3: Smoke Test
- [ ] Run smoke test: `.\scripts\launch_session_8g.ps1 -Phase smoke -Test`
- [ ] Verify observation space: 78 dims (76 + 2 workspace feedback)
- [ ] Verify curriculum stage 1 weights active
- [ ] Check reward components
- [ ] Verify no crashes

### Phase 4: Stage 1 Training (0-50M)
- [ ] Launch training with curriculum
- [ ] Monitor workspace distance converging to 0.50-0.75m
- [ ] Check TensorBoard: reachability bonus should increase
- [ ] Alert if drift outside target zone >10k steps
- [ ] **Key metric:** Workspace distance should stabilize by 30M

### Phase 5: Stage 2 Training (50M-100M)
- [ ] Curriculum automatically transitions at 50M
- [ ] Monitor position/orientation errors decreasing
- [ ] Reachability should maintain (not collapse like 8e!)
- [ ] Workspace should stay stable (learned in Stage 1)

### Phase 6: Evaluation & Decision
- [ ] Evaluate at 50M (end of Stage 1)
  - Check workspace: Should be 0.50-0.75m ✅
  - Check reachability: Should be >1.5 ✅
  - If good: proceed to Stage 2
  - If drift: diagnose before continuing
- [ ] Evaluate at 100M (final)
  - Compare all metrics with 8f
  - Check for degradation (like 8e!)
  - Document what worked / what didn't

---

## 📈 Success Criteria (REVISED - Evidence-Based)

### Must Achieve (Minimum Viable Product)
✅ Position error ≤ 310cm (match 8f, don't regress)  
✅ Orientation error ≤ 46° (match 8f, don't regress)  
✅ Reachability bonus ≥ 1.5 (2x improvement vs 8f)  
✅ Workspace stable in 0.50-0.75m (no drift beyond zone)  
✅ No catastrophic collapse (avoid 8e failure pattern)

### Should Achieve (Realistic Target)
🎯 Position error ≤ 290cm (-6% vs 8f)  
🎯 Orientation error ≤ 43° (-8% vs 8f)  
🎯 Reachability bonus ≥ 2.5 (291% vs 8f)  
🎯 Workspace converges and stays 0.55-0.70m  
🎯 Curriculum Stage 1 shows clear workspace learning

### Stretch Goals (If everything works)
🏆 Position error ≤ 275cm (-11% vs 8f)  
🏆 Orientation error ≤ 40° (-14% vs 8f)  
🏆 Reachability bonus ≥ 3.5 (447% vs 8f)  
🏆 Workspace precision: 0.50-0.65m stable

---

## ⚠️ Risk Mitigation (REVISED)

### Risk 1: Wider Zone Reduces Precision
**Risk:** 0.50-0.75m optimal zone might allow sloppy positioning  
**Evidence:** PLY export shows this is FK reality, not "too loose"  
**Mitigation:** Curriculum Stage 2 will tighten with full position weight  
**Fallback:** If position degrades >5%, narrow zone to 0.50-0.70m

### Risk 2: Curriculum Transition Causes Instability
**Risk:** Weight jump at 50M might disrupt learned behaviors  
**Evidence:** Common in curriculum RL, but manageable  
**Mitigation:** PPO's adaptive KL + entropy decay should smooth transition  
**Fallback:** If instability detected, use gradual linear interpolation 45M-55M

### Risk 3: Workspace Drift Persists Despite Feedback
**Risk:** Even with explicit observations, policy might not use them  
**Evidence:** 8f drifted despite distance gating  
**Mitigation:** Curriculum Stage 1 focuses learning on workspace first  
**Fallback:** If drift >0.80m, add explicit workspace distance penalty (last resort!)

### Risk 4: No Improvement Over 8f
**Risk:** Changes might not move the needle  
**Evidence:** Possible, but structural fixes are sound  
**Mitigation:** Early evaluation at 50M, can adapt for Stage 2  
**Fallback:** If 50M shows no improvement, analyze and consider Recovery Drills (future 8h)

---

## 🔄 What We're NOT Doing (And Why)

### ❌ NOT Increasing Penalty Weights
**Rejected:** `reachability_maintenance_reward: 40 → 80`  
**Reason:** Caused 8c/8d stall, huge negative slope overwhelms learning  
**Alternative:** Expand comfort zone instead (structural fix, not weight fix)

### ❌ NOT Targeting 8e's 34° Orientation Directly
**Rejected:** Cranking orientation weight to replicate 8e  
**Reason:** 8e's orientation came from failure mode (gave up on position)  
**Alternative:** Improve orientation via better base positioning

### ❌ NOT Tightening Workspace Constraints
**Rejected:** Narrower optimal zone to "force" precision  
**Reason:** Fights FK geometry, creates penalty pressure, causes drift  
**Alternative:** Wider zone + curriculum precision in Stage 2

### ❌ NOT Bell-Shaped Rewards
**Rejected:** Gaussian peaks or narrow comfort zones  
**Reason:** 8e catastrophically failed with this approach  
**Alternative:** Linear two-zone with expanded plateau

---

## 🎯 Key Design Principles for Session 8g

1. **Match FK Workspace Reality:** 0.50-0.75m zone aligns with reachable volume
2. **Structural Over Parametric:** Change geometry, not just weights
3. **Curriculum Over Cramming:** Learn workspace first, then precision
4. **Evidence-Based Targets:** Conservative predictions based on 8f data
5. **Early Evaluation:** Decision point at 50M to adapt if needed
6. **Gentler, Not Heavier:** Lower penalty pressure enables learning

---

## 📊 Comparison Matrix: 8d/8e/8f/8g (Predicted)

| Metric | 8d | 8e @ 100M | 8f | 8g Target | Improvement |
|--------|----|-----------|----|-----------|-------------|
| **Position** | 311cm | 408cm | 308cm | **290cm** | -6% vs 8f |
| **Orientation** | 47.4° | 34.0° | 46.5° | **43°** | -8% vs 8f |
| **Reward** | -177k | -293k | -126k | **-110k** | +13% vs 8f |
| **Reachability** | 7.06 | 0.58 | 0.64 | **2.5** | +291% vs 8f |
| **Workspace** | 0.40m | 0.52→0.58m | 0.42→0.60m | **0.55-0.70m** | Stable in zone |
| **Approach** | Linear | Bell (failed) | Two-zone tight | **Two-zone expanded** | Better structure |

---

## 🚀 Next Steps

### Immediate (This Session)
1. **Review revised proposal** - Address any remaining concerns
2. **Begin Phase 1 implementation** - Code changes in rewards.py, observations.py, config.py
3. **Create launcher script** - `launch_session_8g.ps1` with curriculum flags
4. **Smoke test** - Verify 78-dim obs space and curriculum weights

### Stage 1 Training (0-50M)
1. **Monitor workspace convergence** - Should settle to 0.50-0.75m by 30M
2. **Check reachability growth** - Should increase from 0.64 toward 1.5+
3. **TensorBoard vigilance** - Watch for drift, instability, reward stagnation
4. **Decision at 50M** - Evaluate and decide: continue, adapt, or pivot

### Stage 2 Training (50M-100M)
1. **Curriculum transition** - Full weights engage automatically
2. **Position/orientation refinement** - Metrics should improve from Stage 1 baseline
3. **Maintain workspace stability** - Learned habits from Stage 1 should persist
4. **Final evaluation at 100M** - Compare with 8f, document lessons

### Future Sessions (If 8g Succeeds)
1. **Session 8h: Recovery Drills** - Handle edge cases, hard resets
2. **Session 8i: Tighter Precision** - Once workspace stable, reduce tolerances
3. **Session 8j: Full Deployment** - Transfer to real robot

---

## 💭 Why This Approach Should Work

**Session 8g builds on proven 8f fixes while addressing root causes:**

1. ✅ **Keep what works:** Atomic state, heading cue, distance gating, two-zone linear
2. 🔧 **Fix workspace geometry:** 0.50-0.75m matches FK reality (not arbitrary tight zone)
3. 🎯 **Curriculum structure:** Learn one thing at a time (workspace → precision)
4. 📊 **Explicit observations:** Policy gets direct feedback on workspace quality
5. ⚠️ **Early alerts:** Monitoring catches drift before it becomes catastrophic
6. 🚫 **Avoid past mistakes:** No heavy penalties (8d), no bell shapes (8e), no tight zones (8f)

**The key insight:** 8f was close! Just needed wider workspace tolerance and staged learning.

**Ready to implement?** 🚀

---

## 📊 Expected Session 8g Results (REVISED - Conservative)

### Realistic Scenario (Evidence-Based)
```
Position Error:    290cm  (-6% vs 8f's 308cm) ← Incremental improvement
Orientation Error:  43°   (-8% vs 8f's 46.5°) ← Better base positioning helps
Mean Reward:      -110k   (+13% vs 8f's -126k)
Reachability:      2.5    (+291% vs 8f's 0.64) ← Wider zone increases naturally
Workspace:         0.55-0.70m stable ← Natural settling in expanded zone
```

### Optimistic Scenario (If curriculum works well)
```
Position Error:    275cm  (-11% vs 8f)
Orientation Error:  40°   (-14% vs 8f)
Mean Reward:       -95k   (+25% vs 8f)
Reachability:      3.5    (+447% vs 8f)
Workspace:         0.50-0.65m stable
```

### Minimum Viable (Conservative Fallback)
```
Position Error:    305cm  (-1% vs 8f, acceptable)
Orientation Error:  45°   (-3% vs 8f, slight improvement)
Mean Reward:      -120k   (+5% vs 8f)
Reachability:      1.5    (+134% vs 8f, still better)
Workspace:         0.50-0.75m (stays in zone)
```

**Why conservative?** No magic bullets—improvements come from better structure, not weight fiddling.

---

## 🚀 Implementation Checklist (REVISED)

### Phase 1: Code Changes
- [ ] Update `rewards.py`: Expand optimal zone 0.45-0.55m → 0.50-0.75m
- [ ] Update `rewards.py`: Adjust gate threshold 0.55m → 0.65m
- [ ] Update `observations.py`: Add workspace feedback (+2 dims, total 78)
- [ ] Update `config.py`: Add curriculum configuration
- [ ] Update `rewards.py`: Implement curriculum weight switching
- [ ] Add `WorkspaceDistanceMonitor` callback in `train.py`
- [ ] **KEEP reachability_maintenance_reward = 40** (do NOT increase!)

### Phase 2: Launcher Script
- [ ] Create `scripts/launch_session_8g.ps1`
- [ ] Task: MobileMMTrackEE-v0
- [ ] Num_envs: 16384
- [ ] Total_timesteps: 100M
- [ ] Headless: true
- [ ] Enable curriculum flag

### Phase 3: Smoke Test
- [ ] Run smoke test: `.\scripts\launch_session_8g.ps1 -Phase smoke -Test`
- [ ] Verify observation space: 78 dims (76 + 2 workspace feedback)
- [ ] Verify curriculum stage 1 weights active
- [ ] Check reward components
- [ ] Verify no crashes

### Phase 4: Stage 1 Training (0-50M)
- [ ] Launch training with curriculum
- [ ] Monitor workspace distance converging to 0.50-0.75m
- [ ] Check TensorBoard: reachability bonus should increase
- [ ] Alert if drift outside target zone >10k steps
- [ ] **Key metric:** Workspace distance should stabilize by 30M

### Phase 5: Stage 2 Training (50M-100M)
- [ ] Curriculum automatically transitions at 50M
- [ ] Monitor position/orientation errors decreasing
- [ ] Reachability should maintain (not collapse like 8e!)
- [ ] Workspace should stay stable (learned in Stage 1)

### Phase 6: Evaluation & Decision
- [ ] Evaluate at 50M (end of Stage 1)
  - Check workspace: Should be 0.50-0.75m ✅
  - Check reachability: Should be >1.5 ✅
  - If good: proceed to Stage 2
  - If drift: diagnose before continuing
- [ ] Evaluate at 100M (final)
  - Compare all metrics with 8f
  - Check for degradation (like 8e!)
  - Document what worked / what didn't  
🏆 Orientation error ≤ 36° (approach 8e's 34°)  
🏆 Reachability bonus ≥ 4.5 (approach 8d's 7.06)  
🏆 Workspace distance stable 0.45-0.55m (tight)

---

## ⚠️ Risk Mitigation

### Risk 1: Orientation-Position Trade-off
**Risk:** Increasing orientation weight might hurt position tracking  
**Mitigation:** Increase reachability weight simultaneously to maintain position focus  
**Fallback:** If position degrades >5%, reduce orientation_tracking to 40

### Risk 2: Workspace Drift Persists
**Risk:** Even with 2x reachability weight, drift might continue  
**Mitigation:** Live monitoring with WorkspaceDistanceMonitor callback  
**Fallback:** If drift detected at 50M, stop and increase weight to 120 or add explicit workspace reward

### Risk 3: Wider Plateau Reduces Precision
**Risk:** 0.40-0.60m optimal zone might reduce tracking accuracy  
**Mitigation:** Higher reachability weight should compensate  
**Fallback:** If position degrades, narrow back to 0.42-0.58m

### Risk 4: Increased Weights Cause Instability
**Risk:** Higher reward weights might cause training instability  
**Mitigation:** PPO's adaptive KL should handle it, but monitor closely  
**Fallback:** If variance explodes, reduce both weights by 20%

---

## 📊 Comparison Matrix: 8d/8e/8f/8g (Predicted)

| Metric | 8d | 8e | 8f | 8g Target | Best |
|--------|----|----|----|-----------|----|
| **Position** | 311cm | 408cm | 308cm | **295cm** | 🏆 8g |
| **Orientation** | 47.4° | 34.0° | 46.5° | **38°** | 🏆 8e |
| **Reward** | -177k | -293k | -126k | **-105k** | 🏆 8g |
| **Reachability** | 7.06 | 0.58 | 0.64 | **3.0** | 🏆 8d |
| **Mobilization** | N/A | 0.13 | 0.32 | **0.35** | 🏆 8g |
| **Workspace** | 0.40m | 0.52→0.58m | 0.42→0.60m | **0.45-0.55m** | 🏆 8g |

---

## 🎯 Decision Points

### At 50M Checkpoint:
**Evaluate and decide:**
1. If position ≤310cm AND orientation ≤40° → **Continue to 100M** ✅
2. If workspace drifting (>0.60m) → **Stop and increase reachability weight** ⚠️
3. If orientation >45° → **Increase orientation_tracking to 60** ⚠️
4. If position >320cm → **Reduce orientation_tracking to 45** ⚠️

### At 100M Completion:
**Compare with Session 8f:**
1. If BOTH position AND orientation better → **SUCCESS!** 🎉
2. If ONE metric significantly better (>20%) → **Partial success** 👍
3. If BOTH metrics worse → **Analyze failure mode** 🔍

---

## 📝 Documentation to Create

1. **SESSION_8G_IMPLEMENTATION.md** - Detailed implementation guide
2. **scripts/launch_session_8g.ps1** - Training launcher
3. **SESSION_8G_CONFIG.md** - All config changes documented
4. **SESSION_8G_MONITORING.md** - What to watch during training

---

## 🚀 Next Steps

1. **Review this proposal** - Any concerns or adjustments?
2. **Implement Phase 1** - Code changes in config.py and rewards.py
3. **Create launcher** - Session 8g PowerShell script
4. **Smoke test** - Verify environment setup
5. **Launch training** - 100M steps with live monitoring

**Ready to proceed with implementation?** 🎯
