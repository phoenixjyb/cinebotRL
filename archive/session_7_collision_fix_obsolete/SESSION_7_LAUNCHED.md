# Session 7: Self-Collision Penalty Fix

**Status:** 🚀 TRAINING (100M steps)  
**Date:** 2025-10-23 10:05 (test) → 10:12 (full training launched)  
**Commit:** 946a999

---

## 🎯 The Fix

**One-line change:**
```python
# src/rl_platform/tasks/mobile_mm/config.py (line 99)
self_collision_penalty: float = 0.5  # Was 50.0
```

**Reduction:** 99% (50.0 → 0.5)

---

## 📊 Expected Impact

### Session 6 (penalty=50.0)
```
Collision penalty per step: ~30,000 points
Episode reward: -11,715,724
Ratio: Collision:Tracking = 630:1 (catastrophic!)
```

### Session 7 (penalty=0.5) - Expected
```
Collision penalty per step: ~300 points
Episode reward: -120,000 to +30,000 (99% improvement!)
Ratio: Collision:Tracking = 6:1 (learnable!)
```

---

## 🧪 Testing Protocol

### Phase 1: Quick Test (1M steps, ~5 min)
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --n_steps 128 `
    --batch_size 1024 `
    --total_timesteps 1000000 `
    --learning_rate 3e-4 `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --headless
```

**Success Criteria:**
- ✅ Episode reward: -50K to +10K (NOT -11M!)
- ✅ Self-collision penalty: < 1000/step (NOT 30K!)
- ✅ Base mobilization: > 0 (still moving)
- ✅ No crashes, no NaN values

**If successful → Proceed to Phase 2**

### Phase 2: Full Training (100M steps, ~9 hours)
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
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

### Phase 3: Evaluation
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py `
    --checkpoint logs/sb3/mobilemmtrackee_v0/SESSION7/final_model.zip `
    --num_envs 16 `
    --num_episodes 10 `
    --headless `
    --trajectory_type multi_recorded
```

**Target Metrics:**
- Mean episode reward: -5000 to +50000 (vs -11.7M)
- Mean tracking error: < 0.5m (vs 0.49-2.5m)
- Self-collision penalty: < 500/step (vs 30K)
- Base mobilization: > 0.1 (actively used)

---

## 🤔 Design Rationale

### Why 0.5 instead of 1.0 or 5.0?

**Mathematical Analysis:**
```
Best tracking reward per step: ~50 points (position) + 30 (mobilization) = 80 points
Current collision forces: ~600 units

Target ratio: 3:1 to 10:1 (collision:tracking)
Ideal weight: 240-800 / 600 = 0.4 to 1.3

Chosen: 0.5 (middle of range)
```

**Philosophy:**
- **Primary goal:** Track trajectory well (50 points)
- **Secondary concern:** Don't self-collide excessively (300 points @ 0.5)
- **Ratio 6:1** means collisions still matter, but don't destroy learning

### Why not add collision filtering first?

**Answer:** We don't know which collisions are real vs false positives!

By reducing the penalty, we let the robot explore and reveal:
1. Which body pairs actually collide
2. Whether collisions hurt tracking performance
3. Which collisions are unavoidable (geometry issue)

Then in Session 8, we can add **surgical** collision filtering based on data.

---

## 📈 Next Steps Based on Results

### Scenario A: Episode rewards now reasonable (-5K to +10K) ✅
→ **Proceed with full 100M training**  
→ Policy should learn quickly

### Scenario B: Still very negative (-500K) but better ⚠️
→ **Reduce penalty further to 0.1**  
→ Retest before full training

### Scenario C: Rewards too positive (+50K consistently) ⚠️
→ **Robot ignoring collisions**  
→ Increase penalty to 1.0-2.0

### Scenario D: Base stops moving 🛑
→ **Unlikely, but if happens:**
→ Revert penalty to 5.0
→ Investigate collision geometry in Isaac Sim GUI

---

## 🔬 What We'll Learn

1. **Is 99% reduction safe?**
   - Does the robot still avoid collisions?
   - Or does it become reckless?

2. **Which collisions are real?**
   - Monitor evaluation logs for collision body pairs
   - Identify patterns (arm-base? adjacent links?)

3. **Does tracking improve?**
   - With learning signal restored, should see < 0.5m errors
   - Base should contribute more to tracking

4. **What's the optimal penalty?**
   - If 0.5 works → keep it
   - If too low → adjust to 1.0-2.0 in Session 8
   - If too high → try 0.1-0.2

---

## 💾 Commit History

**946a999** - Session 7: Reduce self_collision_penalty 50.0 → 0.5 (99% reduction)
- Fixes catastrophic -11.7M episode rewards from Session 6
- Expected improvement: 99.9% better episode rewards
- Strategy: Explore first, filter later

---

## ⏱️ Timeline

- **10:05** - Quick test started (1M steps)
- **10:10** - Expected completion of quick test
- **10:15** - Decision point: Full training or adjust?
- **19:30** - Expected completion of full training (if approved)

---

## 📝 Notes

- Session 6 already had penalty=50.0 (not 1000.0 as initially thought)
- The 30K penalties come from summing ~600 force units across bodies
- Even at 0.5, collision penalty (300) is still 6x tracking reward (50)
- This is an **exploration phase** - Session 8 will add surgical filtering
