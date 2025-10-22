# Training Hierarchy and Episode Management Explained

This document explains the complete training hierarchy for the mobile manipulator RL system, from trajectories to rollout buffers.

---

## 📊 Training Hierarchy Overview

### **1. Dataset Level** (Your Recordings)
- **~1000+ trajectory files** in `trajectoryToLearn/world_json/*.json`
- Each trajectory file contains **~200-400 waypoints** (poses)
- Example: `1_pull_world_scaled.json` has 2734 lines ≈ **273 waypoints** (each pose is ~10 lines of JSON)
- Waypoints are typically spaced by **0.1 seconds** in the original recording

### **2. Environment Level** (Parallel Training)
```bash
--num_envs 4096
```
- **4096 parallel environments** running simultaneously
- Each environment tracks **one trajectory** at a time
- When an env resets, it picks a **new random trajectory** from your 1000+ files
- Environments run **independently** and **asynchronously**

### **3. Episode Level** (Within Each Environment)
- **One episode** = tracking one complete trajectory from start to finish
- **Episode length** varies based on number of waypoints in the selected trajectory
- Trajectory advances through waypoints at **20Hz** (dt=0.05s per step)
- Episode terminates when:
  - Trajectory completes (all waypoints tracked) ✅
  - OR collision detected 💥
  - OR max episode length reached ⏱️

### **4. Training Step Level** (PPO Algorithm)
```bash
--n_steps 128          # Rollout buffer size per environment
--batch_size 1024      # Mini-batch size for gradient updates
--total_timesteps 100000000  # Total training steps across all envs
```

**Training Loop**:
1. **Collect rollout**: Run 4096 envs for **128 steps** each = **524,288 samples**
2. **Update policy**: Split 524k samples into mini-batches of **1024**, do gradient descent
3. **Repeat** until 100M total timesteps reached

---

## 🔄 What is a "Step"?

A **step** is **one control cycle** through the entire RL loop:

### **One Step = One Complete Cycle:**

```
1. Policy observes current state (robot pose, EE position, target, etc.)
   ↓
2. Policy outputs action (6 arm joint targets + 2 base velocities)
   ↓
3. Actions applied to robot in simulation
   ↓
4. Physics simulation runs (Isaac Sim steps forward in time)
   ↓
5. Robot moves, new state observed
   ↓
6. Reward calculated based on new state
   ↓
7. Check if episode should terminate (collision, success, timeout)
   ↓
8. Store (state, action, reward, done) in rollout buffer
```

### **Timing:**
- **One step = 0.05 seconds of simulated time** (20 Hz control frequency)
- Physical time: depends on GPU speed (typically 100-1000+ FPS with 4096 envs)

---

## 📦 Steps per Rollout (n_steps = 128)

```bash
--n_steps 128
```

This means: **Collect 128 steps of experience before updating the policy**

### **What happens:**

```
For each of 4096 environments:
  Step 1:  observe → act → simulate → reward → store
  Step 2:  observe → act → simulate → reward → store
  Step 3:  observe → act → simulate → reward → store
  ...
  Step 128: observe → act → simulate → reward → store

Result: 4096 × 128 = 524,288 samples in rollout buffer

Then: Policy update using these 524k samples (split into mini-batches of 1024)
```

---

## 🎯 Example: Tracking a 300-Waypoint Trajectory

### **Math:**
```
Trajectory: 300 waypoints × 0.1s spacing = 30 seconds
Control frequency: 20Hz (0.05s per step)
Steps needed: 30s ÷ 0.05s = 600 steps
Rollouts needed: 600 ÷ 64 ≈ 10 rollouts
```

### **Timeline for One Environment:**

| Step | Simulated Time | Waypoint Index | What Happens |
|------|----------------|----------------|--------------|
| 1    | 0.00s         | 0              | Start tracking, observe, act |
| 2    | 0.05s         | 1              | Move toward waypoint 1 |
| 3    | 0.10s         | 2              | Advance to waypoint 2 |
| ...  | ...           | ...            | Continue tracking |
| 64   | 3.20s         | ~32            | At waypoint 32 (out of 300) |
| ...  | ...           | ...            | Continue tracking |
| 600  | 30.00s        | 299            | Complete trajectory! |

**After step 64**: Policy update happens, then continue collecting next rollout.

---

## 🔄 Asynchronous Episode Handling

### **Critical Insight: Environments DON'T wait for each other!**

PPO uses **asynchronous episode handling**:

```python
# Simplified PPO rollout collection
for step in range(n_steps):  # 128 steps
    for env in range(num_envs):  # 4096 envs
        action = policy(obs[env])
        obs[env], reward[env], done[env] = env.step(action)
        
        if done[env]:  # Environment finished its trajectory
            # IMMEDIATELY reset this env with a NEW trajectory
            obs[env] = env.reset()
            # Continue collecting - don't wait for others!
```

### **What Actually Happens:**

| Time | Env 0 (short traj, 500 steps) | Env 1 (long traj, 2000 steps) | Rollout Buffer |
|------|-------------------------------|--------------------------------|----------------|
| Step 1-128 | Tracking trajectory A | Tracking trajectory B | Collect rollout 1 |
| Step 129-256 | Tracking trajectory A | Tracking trajectory B | Collect rollout 2 |
| Step 257-384 | Tracking trajectory A | Tracking trajectory B | Collect rollout 3 |
| Step 385-512 | **DONE!** Reset → Traj C | Tracking trajectory B | Collect rollout 4 |
| Step 513-640 | Tracking trajectory C | Tracking trajectory B | Collect rollout 5 |
| ... | ... | ... | ... |

**Key points:**
1. ✅ Each env runs **independently**
2. ✅ When an env finishes, it **immediately resets** with a new trajectory
3. ✅ The rollout buffer collects **128 steps from ALL envs**, regardless of resets
4. ❌ **NO** - we do NOT wait for all envs to complete their trajectories

---

## 📦 Rollout Buffer with Episode Boundaries

The rollout buffer stores **transitions**, not complete episodes:

```python
# Rollout buffer contains 4096 × 128 = 524,288 transitions:
[
  # Env 0, steps 1-128 (might span 2 different trajectories!)
  (obs_0_1, action_0_1, reward_0_1, done_0_1),
  (obs_0_2, action_0_2, reward_0_2, done_0_2),
  ...
  (obs_0_100, action_0_100, reward_0_100, done=True),  # Trajectory A ends
  (obs_0_101, action_0_101, reward_0_101, done=False), # NEW trajectory B starts!
  ...
  
  # Env 1, steps 1-128 (might be middle of one long trajectory)
  (obs_1_1, action_1_1, reward_1_1, done_1_1),
  ...
]
```

**PPO handles episode boundaries using the `done` flag:**
- When computing returns (discounted rewards), it **resets** at `done=True`
- Value function learns to predict "return from this state until episode ends"
- Episode boundaries are naturally handled in the advantage calculation

---

## 📊 Complete Training Numbers

| Concept | Value | Meaning |
|---------|-------|---------|
| **One step** | 0.05s sim time | One control cycle (observe→act→simulate→reward) |
| **n_steps** | 64 | Steps per rollout buffer |
| **One rollout** | 3.20s sim time | 64 steps × 0.05s |
| **Rollout samples** | 262,144 | 4096 envs × 64 steps |
| **Policy updates** | Every 64 steps | After each rollout completes |
| **Total training** | 100M steps | Across all 4096 environments |
| **Total rollouts** | ~1,562,500 | 100M ÷ 64 |
| **Updates per env** | ~381 | 1,562,500 ÷ 4096 |
| **Training duration** | ~24,414 steps/env | 100M ÷ 4096 |

---

## 🎯 Key Questions Answered

### **Q1: Do we wait till the last env completes its trajectory?**
**A: NO!** Each env resets independently when its trajectory finishes.

### **Q2: Does each env see the completion of whole trajectories?**
**A: YES, but not synchronized!** 

- Short trajectories (500 steps): Complete **~49 times** per env during 100M training steps
- Long trajectories (2000 steps): Complete **~12 times** per env during 100M training steps
- Each env experiences **hundreds of complete trajectories** over training

### **Q3: What does the "grand iteration number" mean?**
```
Iteration 1000 | Total timesteps: 524,288,000
```

This counts **total steps across ALL environments**, not complete trajectories:
- **Total steps** = sum of all control cycles across all 4096 envs
- **Iteration** = number of policy updates (every 128 steps per env)

### **Q4: What does the tracking output show?**
```
[TRACKING Step 250] Env 0:
  🎯 Target (WORLD):  [2.741, 0.681, 0.789]
```

This shows:
- **Step 250** of training (total control cycles across all envs)
- **Env 0** is currently at waypoint position `[2.741, 0.681, 0.789]`
- This waypoint is **somewhere in the middle** of whichever trajectory Env 0 is currently tracking
- **NOT** the 250th waypoint of the trajectory!
- **NOT** the final waypoint!
- Printed every **50 steps** for debugging

---

## 📈 Example Timeline with Multiple Episodes

```
Training with 4096 envs, n_steps=128:

Rollout 1 (steps 1-128):
  - Env 0: Steps 1-128 of trajectory A (traj length: 500 steps)
  - Env 1: Steps 1-128 of trajectory B (traj length: 1500 steps)
  - Env 2: Steps 1-128 of trajectory C (traj length: 800 steps)
  → Collect 524k samples → Policy update

Rollout 2 (steps 129-256):
  - Env 0: Steps 129-256 of trajectory A
  - Env 1: Steps 129-256 of trajectory B
  - Env 2: Steps 129-256 of trajectory C
  → Collect 524k samples → Policy update

Rollout 3 (steps 257-384):
  - Env 0: Steps 257-384 of trajectory A
  - Env 1: Steps 257-384 of trajectory B
  - Env 2: Steps 257-384 of trajectory C
  → Collect 524k samples → Policy update

Rollout 4 (steps 385-512):
  - Env 0: Steps 385-500 of traj A, then RESET → steps 1-12 of traj D
  - Env 1: Steps 385-512 of trajectory B (still going!)
  - Env 2: Steps 385-512 of trajectory C
  → Collect 524k samples → Policy update (handles episode boundaries)

Rollout 5 (steps 513-640):
  - Env 0: Steps 13-140 of trajectory D
  - Env 1: Steps 513-640 of trajectory B
  - Env 2: Steps 513-640 of trajectory C
  → Collect 524k samples → Policy update
```

---

## 🎓 Summary

1. ✅ **Environments run asynchronously** - no waiting for others to finish
2. ✅ **Each env completes many full trajectories** during training
3. ✅ **Rollout buffer spans episode boundaries** - PPO handles this correctly with `done` flags
4. ✅ **Steps needed = (waypoints × spacing) ÷ dt** - your math is correct!
5. ✅ **Training progress counts total steps**, not completed trajectories
6. ✅ **One step = one control cycle** (0.05s @ 20Hz), not one waypoint
7. ✅ **Tracking output shows current state**, not final waypoint

The beauty of PPO: It learns from **all transitions** regardless of episode boundaries! 🚀

---

## 📖 Related Documentation

- [Training Architecture](../docs/architecture/training_architecture.md)
- [Reward System](../docs/reference/reward_system.md)
- [Daily Workflow](../docs/workflows/daily_workflow.md)
