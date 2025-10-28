# Session 7c Visualization Guide

**Purpose:** Observe base movement behavior and identify specific issues  
**Duration:** 10-15 minutes  
**Requirements:** Isaac Sim GUI (non-headless)

---

## 🎯 What We're Looking For

Based on Session 7c evaluation results, we need to observe:

1. **Base Movement Patterns:**
   - ✅ Does base move? (We know it does: 0.1-1.8m)
   - ❓ Does it move toward targets or randomly?
   - ❓ Does it oscillate/jitter?
   - ❓ Does it overshoot targets?

2. **Reachability Issues:**
   - ❓ Why are 93% of targets unreachable?
   - ❓ Does base stop too early?
   - ❓ Does base move in wrong direction?
   - ❓ Are trajectories inherently unreachable?

3. **Collision Behavior:**
   - ❓ Where do collisions occur? (logs show "body: base")
   - ❓ Base-ground contact?
   - ❓ Base-leg interference?
   - ❓ Arm self-collision?

4. **Tracking Strategy:**
   - ❓ How does policy balance arm vs base movement?
   - ❓ Does arm reach limits before base moves?
   - ❓ Is coordination smooth?

---

## 🚀 Visualization Commands

### Quick Test (1 Episode, 1 Env)

```bash
I:\isaaclab\isaaclab.bat -p scripts\reinforcement_learning\sb3\evaluate.py \
  --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251027_180246\final_model.zip \
  --num_envs 1 \
  --num_episodes 1 \
  --deterministic \
  --trajectory_type multi_recorded \
  --use_all_trajectories
```

**What to observe:**
- Does base move smoothly or jerky?
- Does end effector track the red target marker?
- Are there any collision warnings?
- Does base position change significantly?

---

### Multiple Trajectories (5 Episodes)

```bash
I:\isaaclab\isaaclab.bat -p scripts\reinforcement_learning\sb3\evaluate.py \
  --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251027_180246\final_model.zip \
  --num_envs 1 \
  --num_episodes 5 \
  --deterministic \
  --trajectory_type multi_recorded \
  --use_all_trajectories
```

**What to observe:**
- Consistency across different trajectories
- Which trajectory types work best?
- Which types fail (arc? crane? tracking_zigzag?)?

---

### Side-by-Side Comparison (4 Envs)

```bash
I:\isaaclab\isaaclab.bat -p scripts\reinforcement_learning\sb3\evaluate.py \
  --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251027_180246\final_model.zip \
  --num_envs 4 \
  --num_episodes 4 \
  --deterministic \
  --trajectory_type multi_recorded \
  --use_all_trajectories
```

**What to observe:**
- Variance between environments
- Best vs worst performers
- Common failure modes

---

## 🎥 Isaac Sim Camera Controls

### Navigation:
- **Left Mouse + Drag**: Rotate view
- **Right Mouse + Drag**: Pan view  
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse + Drag**: Pan view (alternative)

### Useful Views:
1. **Top-down view**: See base XY movement clearly
2. **Side view**: See Z-axis stability (should be ~0)
3. **Follow camera**: Track robot as it moves

### Visual Elements:
- **Red sphere/marker**: Target end-effector position
- **Robot base**: Should move toward target when out of reach
- **Green lines**: Contact forces (if enabled)

---

## 📊 What to Record

### Screenshots/Notes:

**Example 1: Good Base Movement**
```
Time: Step 50
Base Position: [1.2, 0.5, 0.0]
Target Position: [2.0, 0.3, 1.0]
Base-Target Distance: 0.9m (out of reach)
EE Error: 0.3m

Observation: Base moved 0.8m closer, arm extended, tracking good
Status: ✅ This is what we want!
```

**Example 2: Poor Base Movement**
```
Time: Step 100
Base Position: [0.5, 0.1, 0.0]
Target Position: [2.0, 0.3, 1.0]
Base-Target Distance: 1.5m (far out of reach)
EE Error: 1.2m

Observation: Base barely moved, arm fully extended but can't reach
Status: ❌ Base should have moved more!
```

**Example 3: Wrong Direction**
```
Time: Step 75
Base Position: [1.0, -0.5, 0.0] (was [1.2, 0.0, 0.0])
Target Position: [1.5, 0.5, 1.0]
Base-Target Distance: 1.3m (increased!)
EE Error: 1.5m

Observation: Base moved AWAY from target
Status: ❌ Policy navigating incorrectly!
```

---

## 🔍 Specific Tests

### Test 1: Base Z-Clamp Verification

**What to check:** Base Z should stay ~0.0 (not jump or drift)

**How to observe:**
1. Watch base from side view
2. Note Z position in logs
3. Should see: `Base Pos: [X, Y, 0.000]` consistently

**Expected:** Z within ±0.02m of 0.0  
**If different:** Z-clamp might need tuning

---

### Test 2: Reachability Correlation

**What to check:** Does base move more when targets are far?

**How to test:**
1. Run 5 episodes
2. Note when target is >1m from base
3. Observe if base velocity increases

**Expected:** Higher base velocity when target far  
**If different:** Policy not responding to distance

---

### Test 3: Collision Sources

**What to check:** Where do collision warnings occur?

**How to observe:**
1. Watch for red contact visualizations
2. Check console for collision warnings
3. Pause when collision occurs

**Common issues:**
- Base-ground: Normal (weight support), should be filtered
- Base-leg: Problem (mechanical interference)
- Arm-arm: Self-collision, should be penalized

---

### Test 4: Arm Limit Behavior

**What to check:** Does arm reach limits before base moves?

**How to observe:**
1. Watch arm joint angles
2. Note when arm stops extending
3. Check if base responds

**Expected:** When arm maxed out, base should start moving  
**If different:** Base mobilization reward too weak

---

## 📈 Decision Matrix

Based on observations, decide next steps:

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Base moves randomly | Weak alignment reward | ✅ Implement Session 7d changes |
| Base doesn't move enough | Mobilization reward too low | ⬆️ Increase to 250-300 |
| Base moves wrong direction | No directional guidance | ⭐ Add alignment reward |
| Collision spam | Over-sensitive threshold | ⬆️ Increase threshold |
| Arm doesn't use workspace | Arm reward too weak | ⬆️ Increase position_tracking |
| Trajectories impossible | Dataset issue | 🗑️ Filter bad trajectories |
| Smooth coordinated motion | Everything working! | 🎉 Just train longer |

---

## 🎬 Recording (Optional)

### Screen Recording:

**Windows:**
- Win+G → Game Bar → Record
- Or use OBS Studio for higher quality

**What to capture:**
- 30-60 seconds of best episode
- 30-60 seconds of worst episode
- Close-up of base movement
- Top-down view of navigation

### Useful for:
- Debugging later without re-running
- Sharing with team/advisor
- Before/after comparisons (Session 7c vs 7d)

---

## ⏱️ Time Budget

```
Quick test (1 episode):           3 minutes
Multiple trajectories (5 eps):    8 minutes  
Side-by-side (4 envs):            5 minutes
Specific tests:                   5 minutes
Documentation/notes:              5 minutes
-------------------------------------------
Total:                           26 minutes
```

**Minimum viable:** 8 minutes (run 1-2 episodes, take notes)  
**Recommended:** 15 minutes (cover main observations)  
**Thorough:** 30 minutes (all tests + recording)

---

## 📝 Observation Template

```markdown
# Session 7c Visualization Notes

**Date:** 2025-10-28
**Checkpoint:** Session 7c final_model.zip (100M steps)

## Episode 1 - [Trajectory Type]

**Performance:**
- Final EE Error: X.XXm
- Base Movement: X.XXm
- Episode Reward: XXXXX

**Observations:**
- Base movement pattern: [describe]
- Tracking quality: [describe]
- Issues noticed: [list]

**Screenshots:** [if any]

## Episode 2 - [Trajectory Type]

...

## Summary

**Key Findings:**
1. [Most important observation]
2. [Second important observation]
3. [Third important observation]

**Confirmed Issues:**
- [ ] Base moves randomly (not goal-directed)
- [ ] Base doesn't move enough
- [ ] Collisions occurring at: [location]
- [ ] Other: [describe]

**Recommendations:**
- [ ] Implement Session 7d reward changes
- [ ] Adjust specific weights: [which ones]
- [ ] Filter trajectories: [which types]
- [ ] Continue training without changes

**Ready for Session 7d:** [YES/NO/MAYBE]
```

---

## 🚀 Next Steps After Visualization

### If Observations Match Predictions:
1. ✅ Implement Session 7d reward changes
2. ✅ Launch 200M timestep training
3. ✅ Monitor for improvements

### If Unexpected Issues Found:
1. 🔍 Document specific failure modes
2. 🔧 Adjust reward proposal accordingly
3. 🧪 Test changes on short run (10M steps)
4. ✅ Launch full Session 7d if test passes

### If Everything Looks Good:
1. 🎉 Maybe Session 7c is better than metrics suggest!
2. 📊 Run longer evaluation (500 episodes)
3. 📈 Check if mean error drops with more episodes
4. 🤔 Consider if 100M is insufficient training

---

**Goal:** Understand WHY reachability is only 6%, so we can fix it in Session 7d! 🎯
