# Session 8g Implementation Plan (Evidence-Based)

**Date:** November 1, 2025  
**Objective:** Build on Session 8f structural fixes with workspace geometry aligned to FK reality  
**Strategy:** Expand comfort zone + Add workspace observations + Curriculum staging + Gentler penalties

---

## 📊 Baseline Data (Ground Truth)

### Evaluation Results
```
Session 8e @ 100M (Bell-Shaped FAILED):
  Position: 4.08m, Orientation: 34.0°, Reach penalty: 582
  
Session 8f @ 100M (Two-Zone Linear, BEST):
  Position: 3.08m, Orientation: 46.5°, Reach penalty: 232, Reward: -126k
  
Session 8cv2 @ 200M (Historical):
  Position: 3.28m, Orientation: 20.5° (EXCELLENT!)
```

### FK Workspace Reality (matlab/exports/reach_surface.mat)
```
Total points: 1,677
Median radius: 0.594m
P5/P25/P75/P95: [0.15m, 0.39m, 0.78m, 0.92m]

Distribution:
  <0.6m: 50.1%  ← Current hard margin (TOO TIGHT!)
  <0.7m: 64.6%  ← Better target
  <0.9m: 92.0%  ← Upper practical limit
```

**Critical Finding:** Current `reachability_hard_margin = 0.6m` excludes **50%** of reachable workspace!

---

## ✅ What We Keep from Session 8f (PROVEN)

All structural fixes stay:
- ✅ Atomic root state write (13-element tensor, eliminates race conditions)
- ✅ Distance-gated penalties (sigmoid gating for far/near mode)
- ✅ Heading cue observations (+2 dims: sin/cos of yaw error)
- ✅ Two-zone linear reachability (NOT bell-shaped!)
- ✅ Lighter penalty regime (`reachability_maintenance_reward = 40`, NOT 80)

**Evidence:** Session 8f achieved 3.08m (best position), -126k reward (best overall)

---

## 🔧 Session 8g Changes

### 1. Expand Workspace Comfort Zone (Match FK Reality)

**File:** `src/rl_platform/tasks/mobile_mm/config.py`

**Current (Session 8f):**
```python
reachability_hard_margin: float = 0.6  # Excludes 50% of FK workspace!
reachability_distance_weight: float = 60.0  # Quadratic penalty weight
```

**Change to:**
```python
reachability_hard_margin: float = 0.7  # Now includes 65% of FK workspace
reachability_distance_weight: float = 30.0  # Reduced from 60 (gentler)
```

**Rationale:**
- **FK data:** Median 0.59m, P75 = 0.78m → 0.7m is natural target
- **8f evidence:** Policy drifted 0.42→0.60m seeking better reach despite penalties
- **Gentler penalties:** Halving weight (60→30) reduces penalty slope while maintaining guidance

**Expected Impact:**
- Policy can explore 0.5-0.7m range without harsh quadratic penalties
- Natural workspace settling around median (0.59m)
- Reachability bonus should increase (8f: 0.64 → target: >2.0)

---

### 2. Add Workspace Comfort Observations (+2 dims)

**File:** `src/rl_platform/tasks/mobile_mm/observations.py`

**Add after heading cue (around line 94):**
```python
# Workspace comfort feedback
# Normalized distance to optimal zone for policy awareness
optimal_distance = 0.6  # Target working distance (FK median + margin)
comfort_width = 0.15  # ±15cm comfort band

# Comfort signal: 1.0 in optimal zone, fades to 0.0 outside
workspace_comfort = torch.clamp(
    1.0 - torch.abs(base_target_distance - optimal_distance) / comfort_width,
    0.0, 1.0
)
obs.append(workspace_comfort.unsqueeze(-1))  # +1 dim

# Normalized distance to optimal (signed)
# Negative: too close, Positive: too far
dist_to_optimal_normalized = (base_target_distance - optimal_distance) / comfort_width
dist_to_optimal_normalized = torch.clamp(dist_to_optimal_normalized, -2.0, 2.0)
obs.append(dist_to_optimal_normalized.unsqueeze(-1))  # +1 dim

# Total observation dims: 76 → 78
```

**Update dimension calculation (line ~209):**
```python
dim += 8  # Was 6 (heading cue only), now 8 (heading + workspace)
```

**Rationale:**
- **Observation-reward alignment:** Policy sees same "comfort zone" that reward enforces
- **workspace_comfort:** Explicit "am I in good reach?" signal
- **dist_to_optimal_normalized:** Direction/magnitude to move (approach vs retreat)
- **Evidence:** 8f lacked explicit workspace feedback, drifted seeking reach

**Expected Impact:**
- Faster learning of optimal workspace positioning
- Reduces trial-and-error in workspace geometry
- Policy learns "get in range, THEN track precisely"

---

### 3. Curriculum Staging (Two Phases)

**File:** `src/rl_platform/tasks/mobile_mm/config.py`

**Add curriculum configuration:**
```python
# Curriculum: Stage difficulty progression
use_curriculum: bool = True
curriculum_stage_1_steps: int = 50_000_000  # 50M steps

# Stage 1: Learn workspace positioning (easier trajectories)
# Reduce precision demands, focus on "get base in 0.5-0.7m zone"
curriculum_stage_1_params: dict = {
    "trajectory_filter": "short_simple",  # Use chassis_required_trajectories.txt
    "position_tracking_weight": 5.0,      # 50% of final
    "orientation_tracking_weight": 15.0,  # 50% of final
    "reachability_maintenance_weight": 40.0,  # SAME (not increased!)
}

# Stage 2: Full precision tracking (all trajectories)
curriculum_stage_2_params: dict = {
    "trajectory_filter": "all",           # Full cinematic paths
    "position_tracking_weight": 10.0,     # Restore full
    "orientation_tracking_weight": 30.0,  # Restore full
    "reachability_maintenance_weight": 40.0,  # SAME
}
```

**Implementation approach:**
1. **Stage 1 (0-50M):** Use existing `chassis_required_trajectories.txt` filtering
2. **Checkpoint at 50M:** Evaluate workspace positioning
3. **Stage 2 (50M-100M):** Switch to all trajectories + full weights

**Rationale:**
- **8e evidence:** Progressive degradation 50M→100M from trying everything at once
- **Curriculum principle:** Learn workspace geometry before precision demands
- **Existing infrastructure:** `chassis_required_trajectories.txt` already filters easier paths
- **Lower weights in Stage 1:** Reduce noise while policy learns "get in range"

**Expected Impact:**
- Faster Stage 1 convergence (clear sub-goal: workspace positioning)
- More stable Stage 2 (building on strong foundation)
- Natural evaluation point at 50M

---

### 4. Recovery Drills (Future Enhancement)

**Concept:** Add episodes where base starts 1.5-2.0m away from target

**Purpose:**
- Policy experiences successful approach sequences
- Learns mobilization without catastrophic penalties
- Builds confidence in large movements

**Implementation:** Post-8g if needed (Session 8h)

---

### 5. Exploration Schedule Adjustments

**File:** `scripts/reinforcement_learning/sb3/train.py`

**Current entropy decay:**
```python
EntropyDecayCallback(start_step=100_000_000, end_step=150_000_000)
```

**Change to:**
```python
EntropyDecayCallback(start_step=150_000_000, end_step=200_000_000)
# Start decay AFTER full 100M training, gives more exploration time
```

**Rationale:**
- **Longer exploration:** Policy needs time to discover 0.5-0.7m workspace zone
- **8f evidence:** Structural fixes (not just exploration) drove improvement
- **Conservative:** Delay decay to 150M gives full training budget for learning

**Expected Impact:**
- More diverse workspace positioning trials
- Better discovery of optimal working distances
- Avoid premature convergence to suboptimal workspace

---

## 📈 Expected Session 8g Results (Evidence-Based)

### Minimum Viable (Conservative)
```
Position Error:    300cm  (-3% vs 8f's 308cm)
Orientation Error:  44°   (-5% vs 8f's 46.5°)
Reachability:       1.5   (+134% vs 8f's 0.64)
Workspace:          0.55-0.70m (stable in expanded zone)
Mean Reward:       -120k  (+5% vs 8f's -126k)
```

### Realistic Target (Most Likely)
```
Position Error:    285cm  (-7% vs 8f)
Orientation Error:  42°   (-10% vs 8f)
Reachability:       2.5   (+291% vs 8f)
Workspace:          0.50-0.65m (centered on FK median)
Mean Reward:       -105k  (+17% vs 8f)
```

### Optimistic (If All Works)
```
Position Error:    270cm  (-12% vs 8f)
Orientation Error:  38°   (-18% vs 8f)
Reachability:       3.5   (+447% vs 8f)
Workspace:          0.55-0.65m (tight distribution)
Mean Reward:        -90k  (+29% vs 8f)
```

**Why conservative?** Structural changes take time to manifest. First 50M may show modest gains, Stage 2 brings precision.

---

## 🚀 Implementation Checklist

### Phase 1: Code Changes
- [ ] `config.py`: `reachability_hard_margin: 0.6 → 0.7`
- [ ] `config.py`: `reachability_distance_weight: 60.0 → 30.0`
- [ ] `observations.py`: Add workspace comfort observations (+2 dims, total 78)
- [ ] `config.py`: Add curriculum staging configuration
- [ ] `train.py`: Implement curriculum weight switching at 50M checkpoint
- [ ] `train.py`: Adjust entropy decay (100M→150M start)
- [ ] Add `WorkspaceDistanceMonitor` callback for live tracking

### Phase 2: Launcher Script
- [ ] Create `scripts/launch_session_8g.ps1`
- [ ] Stage 1: `--trajectory_dir` with chassis_required filter
- [ ] Stage 2: Switch to full trajectories at 50M
- [ ] Document curriculum transition procedure

### Phase 3: Smoke Test
- [ ] Run: `.\scripts\launch_session_8g.ps1 -Phase smoke -Test`
- [ ] Verify observation space: 78 dims (76 + 2 workspace)
- [ ] Verify curriculum Stage 1 active (reduced weights)
- [ ] Check trajectory filtering working
- [ ] No crashes, clean startup

### Phase 4: Stage 1 Training (0-50M)
- [ ] Launch with curriculum Stage 1
- [ ] Monitor `monitor/workspace_distance` in TensorBoard
- [ ] Target: Converge to 0.55-0.65m by 30M steps
- [ ] Alert if drift >0.8m (use WorkspaceDistanceMonitor)
- [ ] Checkpoint evaluation at 50M

### Phase 5: Stage 1 Evaluation (50M Checkpoint)
- [ ] Run quantitative evaluation
- [ ] **Go/No-Go Decision:**
   - ✅ GO if: workspace in 0.50-0.75m, reachability >1.5, position <320cm
   - ⚠️ ADAPT if: workspace drifting, reachability <1.0
   - ❌ STOP if: catastrophic collapse (like 8e @ 50M)
- [ ] Document Stage 1 results

### Phase 6: Stage 2 Training (50M-100M)
- [ ] Switch to full trajectories
- [ ] Restore full reward weights
- [ ] Monitor for instability during transition
- [ ] Target: Position error decreasing, orientation improving
- [ ] Workspace should stay stable (learned in Stage 1)

### Phase 7: Final Evaluation (100M)
- [ ] Quantitative eval with final_model.zip
- [ ] Compare all metrics with 8f baseline
- [ ] Check for 8e-style degradation
- [ ] Document lessons learned

---

## 📊 Monitoring Priorities (Update MONITORING_GUIDE.md)

### New Metrics to Track
1. **`monitor/workspace_distance`** - Live base-target distance
   - Target: 0.55-0.65m (FK median region)
   - Alert if: <0.4m or >0.8m for >10k steps

2. **`monitor/workspace_comfort_mean`** - Average comfort signal
   - Target: >0.7 (policy staying in zone)
   - Alert if: <0.4 (policy avoiding comfort zone)

3. **`monitor/reachability_bonus_mean`** - Workspace quality
   - Target: >2.0 (3x improvement from 8f's 0.64)
   - Alert if: <1.0 (insufficient improvement)

4. **`monitor/inner_margin_violations`** - Count of <0.35m episodes
   - Target: <5% of episodes
   - Alert if: >15% (policy too aggressive)

### Critical Checkpoints
- **10M:** Early trends, workspace should be converging
- **30M:** Workspace should stabilize in 0.50-0.70m range
- **50M:** DECISION POINT for Stage 2 transition
- **70M:** Stage 2 effects visible, precision improving
- **100M:** Final evaluation vs 8f baseline

---

## ⚠️ Risk Mitigation

### Risk 1: 0.7m Margin Still Too Tight
**Symptom:** Workspace drifts beyond 0.7m despite relaxation  
**Mitigation:** Monitor at 30M, if drifting >0.75m, pause and increase to 0.75m  
**Fallback:** Adaptive margin schedule (0.7m → 0.75m → 0.8m over training)

### Risk 2: Curriculum Transition Instability
**Symptom:** Performance drop when switching to full trajectories at 50M  
**Mitigation:** Gradual weight interpolation 45M-55M instead of hard switch  
**Fallback:** Extend Stage 1 to 60M if workspace not stable

### Risk 3: Gentler Penalties Reduce Urgency
**Symptom:** Policy too relaxed about workspace distance  
**Mitigation:** Workspace comfort observations provide explicit feedback  
**Fallback:** If reachability <1.0 at 50M, increase distance weight 30→40

### Risk 4: No Improvement Over 8f
**Symptom:** 50M eval shows similar or worse metrics than 8f  
**Mitigation:** Analyze where policy is spending time (workspace distribution)  
**Fallback:** Session 8h with recovery drills and harder curriculum

---

## 🔄 What We're NOT Doing

### ❌ NOT Increasing Reachability Maintenance to 80
**Reason:** Caused 8c/8d stall at 3m, huge negative gradient  
**Evidence:** 8f's lighter regime (40) achieved best position (3.08m)

### ❌ NOT Using Bell-Shaped Rewards
**Reason:** 8e catastrophically failed (4.08m, degraded 50M→100M)  
**Evidence:** Narrow Gaussian peaks cause policy abandonment

### ❌ NOT Targeting 8cv2's 20.5° Orientation Directly
**Reason:** Came from failure mode (gave up on position)  
**Alternative:** Improve orientation via better base positioning in reach

### ❌ NOT Tightening Constraints Further
**Reason:** Current 0.6m margin already too tight (excludes 50% of FK workspace)  
**Alternative:** Expand to 0.7m to match arm capabilities

---

## 🎯 Success Criteria

### Must Achieve (Minimum Bar)
✅ Position ≤ 310cm (match 8f, no regression)  
✅ Orientation ≤ 47° (match 8f, no regression)  
✅ Reachability ≥ 1.5 (2x improvement vs 8f's 0.64)  
✅ Workspace stable 0.50-0.75m (no catastrophic drift)  
✅ No 8e-style collapse (progressive degradation)

### Target Achievement (Realistic Goal)
🎯 Position ≤ 285cm (-7% vs 8f)  
🎯 Orientation ≤ 42° (-10% vs 8f)  
🎯 Reachability ≥ 2.5 (+291% vs 8f)  
🎯 Workspace 0.55-0.65m (tight around FK median)  
🎯 Stage 1 shows clear workspace learning

### Stretch Goals (Best Case)
🏆 Position ≤ 270cm (-12% vs 8f)  
🏆 Orientation ≤ 38° (-18% vs 8f, approaching 8e's 34°)  
🏆 Reachability ≥ 3.5 (+447% vs 8f, approaching 8d's 7.06)  
🏆 Mean reward ≤ -90k (+29% vs 8f's -126k)

---

## 📊 Comparison Matrix (Predicted)

| Metric | 8d | 8e@100M | 8f | 8g Target | Best |
|--------|----|---------|----|-----------|------|
| **Position (cm)** | 311 | 408 | **308** | 285 | 🎯 8g |
| **Orientation (°)** | 47.4 | **34.0** | 46.5 | 42 | 🏆 8e |
| **Reach penalty** | ? | 582 | **232** | 150 | 🎯 8g |
| **Reachability** | 7.06 | 0.58 | 0.64 | 2.5 | 🏆 8d |
| **Reward (k)** | -177 | -293 | **-126** | -105 | 🎯 8g |
| **Workspace (m)** | 0.40 | 0.52→0.58 | 0.42→0.60 | 0.55-0.65 | 🎯 8g |
| **Hard Margin** | ? | 0.6 | 0.6 | **0.7** | 🎯 8g |
| **FK Coverage** | ? | 50% | 50% | **65%** | 🎯 8g |

---

## 💭 Why This Should Work

**Evidence-based reasoning:**

1. **FK Workspace Reality:** 0.7m margin covers 65% of reachable space (vs 50% @ 0.6m)
   - **Data:** 1,677 reach points, median 0.594m, P75 = 0.78m
   - **8f drift:** Policy naturally sought 0.60m despite penalties

2. **Gentler Penalties:** Halving distance weight (60→30) reduces gradient
   - **8f success:** Lighter regime (40) beat heavy (80) in 8d
   - **Theory:** Lower penalty slope allows exploration without fear

3. **Explicit Observations:** Policy gets direct workspace feedback
   - **Missing in 8f:** No "am I in good reach?" signal
   - **Theory:** Observation-reward alignment accelerates learning

4. **Curriculum Structure:** Learn workspace before precision
   - **8e failure:** Tried everything simultaneously, collapsed
   - **Theory:** Staged complexity prevents objective competition

5. **Proven Foundation:** Keep all 8f structural fixes
   - **Atomic state, heading cue, distance gating:** All validated
   - **Theory:** Build on what works, don't throw away progress

---

## 🚀 Next Steps

1. **Review and approve** this implementation plan
2. **Phase 1:** Make code changes (config.py, observations.py, train.py)
3. **Phase 2:** Create launch_session_8g.ps1
4. **Phase 3:** Smoke test (verify 78 dims, curriculum active)
5. **Phase 4:** Launch Stage 1 training (0-50M)
6. **Phase 5:** Evaluate at 50M → Go/No-Go decision
7. **Phase 6:** Stage 2 training (50M-100M)
8. **Phase 7:** Final evaluation and comparison

**Ready to begin implementation?** 🎯
