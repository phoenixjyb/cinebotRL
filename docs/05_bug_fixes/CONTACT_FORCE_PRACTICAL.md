# Contact Force API - Test Results & Recommendations

**Date:** October 17, 2025  
**Status:** ⚠️ Unable to verify with standalone test script

---

## 🚧 Test Script Issues

We attempted to create a standalone test script but encountered Isaac Lab initialization issues when running outside of the full training context. This is a known limitation when trying to use Isaac Lab components in isolation.

---

## ✅ **RECOMMENDED: Check During Your Actual Training**

Since we can't easily test in isolation, the **best approach** is to add diagnostics to your actual training run and monitor the first few thousand steps.

### Add This to `env.py` in the `_get_rewards()` function:

```python
# After getting net_contact_forces (around line 660)
net_contact_forces = self.robot.root_physx_view.get_net_contact_forces()
# ... or fallback code ...

# ADD THIS DIAGNOSTIC CODE:
if not hasattr(self, '_contact_force_checked'):
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)
    max_force = contact_force_mag.max().item()
    print(f"\n{'='*80}")
    print(f"CONTACT FORCE API CHECK (First Step)")
    print(f"{'='*80}")
    print(f"Contact forces shape: {net_contact_forces.shape}")
    print(f"Max contact force: {max_force:.4f} N")
    if max_force < 0.001:
        print(f"⚠️  WARNING: Contact forces appear to be zero!")
        print(f"   Self-collision detection may NOT be working!")
    else:
        print(f"✓ Contact forces detected - API appears to be working")
    print(f"{'='*80}\n")
    self._contact_force_checked = True

# ADD TO EXTRAS (around line 707):
contact_force_mag = torch.norm(net_contact_forces, dim=-1)
max_contact_force_per_env = torch.max(contact_force_mag, dim=-1)[0]

self.extras["collision_diagnostics"] = {
    "max_contact_force_mean": max_contact_force_per_env.mean().item(),
    "max_contact_force_max": max_contact_force_per_env.max().item(),
    "max_contact_force_std": max_contact_force_per_env.std().item(),
    "num_envs_with_contact": (max_contact_force_per_env > 1.0).sum().item(),
    "num_severe_collisions": (max_contact_force_per_env > 10.0).sum().item(),
}
```

---

## 📊 What to Look For During Training

### **Start a short training run (1M steps):**

```powershell
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --total_timesteps 1000000 `
    --headless
```

### **Check the console output:**

**✅ GOOD (API Working):**
```
================================================================================
CONTACT FORCE API CHECK (First Step)
================================================================================
Contact forces shape: torch.Size([4096, 15, 3])
Max contact force: 2.3421 N
✓ Contact forces detected - API appears to be working
================================================================================
```

**❌ BAD (API NOT Working):**
```
================================================================================
CONTACT FORCE API CHECK (First Step)
================================================================================
Contact forces shape: torch.Size([4096, 15, 3])
Max contact force: 0.0000 N
⚠️  WARNING: Contact forces appear to be zero!
   Self-collision detection may NOT be working!
================================================================================
```

---

## 📈 Monitor in Tensorboard

After adding the diagnostics, watch these metrics:

```bash
tensorboard --logdir logs/
```

**Key metrics to watch:**

1. **`collision_diagnostics/max_contact_force_mean`**
   - Should be > 0.5 N in early training (random exploration causes collisions)
   - Should decrease as training progresses (agent learns to avoid collisions)
   - ❌ If always 0.0 → API is broken!

2. **`collision_diagnostics/num_envs_with_contact`**
   - Should be 50-500 envs initially (out of 4096)
   - Should decrease to < 10 in late training
   - ❌ If always 0 → API is broken!

3. **`reward_components/self_collision_penalty`**
   - Should be negative (-5 to -50) in early training
   - Should approach 0 as training progresses
   - ❌ If always 0.0 → API is broken!

---

## 🔍 Alternative: Check Your Previous 10M Training Logs

If you still have logs from your previous 10M timestep training, search for:

```powershell
# Search for the warning message
Select-String -Path "logs\*.log" -Pattern "WARNING.*Contact forces API not found"

# Search for contact force values
Select-String -Path "logs\*.log" -Pattern "contact.*force"
```

**If you see:**
```
[WARNING] Contact forces API not found - collision detection disabled!
```
→ ❌ API was not available during your training!

**If you don't see that warning:**
→ ✅ API exists, but we still need to verify it returns non-zero values

---

## 🎯 What This Means for Your 100M Training

### Scenario 1: API is Working ✅
- Contact forces are detected
- Self-collision penalty applies correctly
- Episodes terminate on severe collisions
- Agent learns safe motion patterns
- **Action:** Proceed with 100M training confidently!

### Scenario 2: API Returns Zeros ❌
- No collisions detected ever
- Self-collision penalty always 0
- Agent may learn unsafe behaviors
- **Action:** Consider alternatives:

#### Alternative 1: Joint Limit Heuristics
```python
def heuristic_collision_check(joint_pos):
    """Simple geometric collision check."""
    joint2 = joint_pos[:, 1]  # Shoulder
    joint3 = joint_pos[:, 2]  # Elbow
    
    # Dangerous: shoulder down + elbow up = arm hits base
    collision = (joint2 < -1.0) & (joint3 > 1.5)
    return collision.float() * 50.0  # Penalty
```

#### Alternative 2: Link Distance Check
```python
def distance_based_collision(body_positions):
    """Check minimum distance between links."""
    base_pos = body_positions[:, 0]
    ee_pos = body_positions[:, -1]
    
    distance = torch.norm(ee_pos - base_pos, dim=-1)
    collision = distance < 0.25  # 25cm threshold
    return collision.float() * 50.0
```

---

## 📋 Immediate Action Plan

**Step 1:** Add diagnostic code to `env.py` (5 minutes)
**Step 2:** Run short training (1M steps, ~10 minutes)
**Step 3:** Check console output for contact force check message
**Step 4:** Monitor Tensorboard metrics for first 100k steps
**Step 5:** Make decision:
   - ✅ API works → Proceed with 100M training
   - ❌ API broken → Implement alternative solution

---

## 💡 My Recommendation

Based on your system (Isaac Sim on Windows, successful 10M training):

**Probability Assessment:**
- 70% chance: API is working fine, just needs verification
- 20% chance: API works but insensitive (low force values)
- 10% chance: API is completely broken (all zeros)

**Recommended Path:**
1. Add the diagnostics (copy-paste the code above)
2. Run 1M training to verify
3. If API works → full speed ahead with 100M! 🚀
4. If API broken → quick fix with heuristics, then proceed

**Total time investment:** ~30 minutes to be 100% certain before your multi-day 100M training run!

---

## 🔧 Where to Add the Code

### File: `src/rl_platform/tasks/mobile_mm/env.py`

### Location 1: After line 659 (contact force acquisition)
```python
# EXISTING CODE:
net_contact_forces = self.robot.root_physx_view.get_net_contact_forces()

# ADD DIAGNOSTIC HERE:
if not hasattr(self, '_contact_force_checked'):
    # ... diagnostic code from above ...
```

### Location 2: After line 707 (in _get_rewards, after reward calculation)
```python
# EXISTING CODE:
self.extras["reward_components"] = {...}

# ADD DIAGNOSTIC HERE:
self.extras["collision_diagnostics"] = {
    # ... diagnostic code from above ...
}
```

---

## ✅ Summary

**Question:** "Are contact forces working?"

**Answer:** Unknown - needs verification during actual training

**Solution:** Add 15 lines of diagnostic code + run 1M test

**Confidence After Test:** 100% certain whether API works or not

**Risk if skipped:** Potentially wasting days on 100M training with broken collision detection

**Time to verify:** 30 minutes total

→ **STRONGLY RECOMMEND running the verification before 100M training!**

