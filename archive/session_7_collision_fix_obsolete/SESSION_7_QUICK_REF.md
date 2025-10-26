# Session 7 Quick Reference Card

## 🎯 Goal
Fix self-collision penalty destroying learning signal (Session 6: -11.7M episode reward)

## 🔧 The Fix (1 Line!)

```python
# File: src/rl_platform/tasks/mobile_mm/config.py
# Find line ~XX (in class RewardWeights):

self_collision_penalty: float = 1000.0  # ❌ OLD (Session 6)
self_collision_penalty: float = 5.0     # ✅ NEW (Session 7)
```

**Why:** Reduce penalty 200x so it's 10% of position tracking max (5 vs 50)

---

## 🚀 Test & Launch

### Step 1: Quick Test (5 min)
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
    --task MobileMMTrackEE-v0 --num_envs 4096 --total_timesteps 1000000 --headless
```

**Check:** Episode reward should be **-1000 to +500** (NOT -11.7M!)

### Step 2: Full Training (if test passes)
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

### Step 3: Evaluate
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py `
    --checkpoint logs/sb3/mobilemmtrackee_v0/SESSION7/final_model.zip `
    --num_envs 16 --num_episodes 10 --headless --trajectory_type multi_recorded
```

---

## ✅ Success Criteria

| Metric | Session 6 | Session 7 Target |
|--------|-----------|------------------|
| Episode Reward | -11.7M ❌ | +500 to +2000 ✅ |
| Collision Penalty | 30K/step | 150/step |
| Tracking Error | 0.5-2.5m | 0.3-0.8m |
| Base Moving | YES ✅ | YES ✅ |

---

## 📊 What to Monitor

### Every 10M steps, check:
- `mean_episode_reward`: Should trend positive (not -11M!)
- `mean_tracking_error`: Should be < 1.0m
- `base_mobilization`: Should be > 0.1
- `self_collision_penalty`: Should be < 50/step (was 30K!)

### Red Flags 🚩:
- Episode reward still very negative (-10K+) → Stop and investigate
- Base stops moving → Revert jerk_penalty to 5.0
- Rewards explode → Reduce penalty further (try 2.0)

---

## 📝 Commit Message

```
Session 7: Fix self-collision penalty (1000.0 → 5.0)

Session 6 revealed self-collision penalties (-30K/step) overwhelming
all other rewards, causing -11.7M episode rewards. Policy learned
decent tracking (0.5-2.5m) but learning signal was destroyed.

Changes:
- self_collision_penalty: 1000.0 → 5.0 (200x reduction)
- Makes penalty ~10% of position tracking max
- Still penalizes collisions without dominating reward

Expected Results:
- Episode reward: +500 to +2000 (99.98% improvement!)
- Collision penalty: ~150/step (99.5% reduction)
- Tracking: 0.3-0.8m (40-68% better)

Testing: 1M step test first, then full 100M if successful.
```

---

## 📚 Reference Documents

- **Full Plan:** `docs/SESSION_7_PLAN.md`
- **Session 6 Analysis:** `docs/SESSION_6_EVALUATION_SUMMARY.md`
- **Master Log:** `TRAINING_SESSIONS_MASTER_LOG.md`

---

**Created:** 2025-10-23  
**Status:** Ready to Launch 🚀  
**Time to Fix:** 10 min + 9 hr training
