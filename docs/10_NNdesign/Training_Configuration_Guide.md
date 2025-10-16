# Training Configuration Guide for 2048 Environments

## ✅ UPDATED Default Settings

The training script has been updated with **proper defaults** for 2048 environments:

```python
--total_timesteps  10,000,000   # 10M timesteps for good learning
--learning_rate    0.0003       # Standard PPO learning rate
--n_steps          32           # ← KEY FIX: Was 4096 (BAD), now 32 (GOOD)
--batch_size       512          # Proper mini-batch size
--n_epochs         10           # Standard PPO epochs
--save_freq        100,000      # Checkpoint every 100K steps
```

---

## 🎯 Why n_steps=32 is Correct

### **The Formula:**
```
Timesteps per iteration = n_steps × num_envs
                       = 32 × 2048
                       = 65,536 timesteps
```

### **Training Progress:**
```
Total timesteps:        10,000,000
Per iteration:          65,536
─────────────────────────────────
Total iterations:       ~153 iterations ✓
```

---

## ❌ Why n_steps=4096 Was Wrong

### **Old (BAD) Configuration:**
```
n_steps = 4,096
num_envs = 2,048
─────────────────────────
Per iteration = 8,388,608 timesteps
Iterations to 10M = 2 iterations only!
```

**Problems:**
- ❌ Only 2 policy updates for entire training
- ❌ No iterative improvement
- ❌ No gradient accumulation
- ❌ Robot can't learn from mistakes
- ❌ Massive sample waste

---

## ✅ New (GOOD) Configuration

### **Updated Settings:**
```
n_steps = 32
num_envs = 2,048
─────────────────────────
Per iteration = 65,536 timesteps
Iterations to 10M = 153 iterations ✓
```

**Benefits:**
- ✅ 153 policy updates (vs 2!)
- ✅ Iterative improvement every ~2 seconds
- ✅ Policy can adapt and refine
- ✅ Proper gradient-based learning
- ✅ See progress in real-time

---

## 📊 Comparison Table

| Config | n_steps | Timesteps/Iter | Iterations | Learning Quality | Speed/Iter |
|--------|---------|---------------|------------|------------------|------------|
| **Old (BAD)** | 4096 | 8.4M | 2 | ❌ No learning | ~260s |
| **New (GOOD)** | 32 | 65K | 153 | ✅ Excellent | ~2s |
| Alternative | 16 | 33K | 305 | ✅ Good | ~1s |
| Alternative | 64 | 131K | 76 | ✅ Good | ~4s |

---

## 🚀 How to Use

### **Option 1: Use Defaults (Recommended)**
```bash
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 2048 \
    --headless
```

This will automatically use:
- `n_steps=32`
- `batch_size=512`
- `total_timesteps=10,000,000`

---

### **Option 2: Custom Settings**

```bash
# Faster iterations (more frequent updates):
.\isaaclab.bat -p train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 2048 \
    --n_steps 16 \
    --batch_size 256 \
    --total_timesteps 10000000 \
    --headless

# OR

# Larger batches (more stable training):
.\isaaclab.bat -p train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 2048 \
    --n_steps 64 \
    --batch_size 1024 \
    --total_timesteps 10000000 \
    --headless
```

---

## 🔬 Expected Performance

### **With n_steps=32 (New Default):**

```
Physics FPS:          ~32,000 (unchanged, still fast!)
Rollout time:         ~2 seconds per iteration
Training time:        ~3-5 seconds per iteration
Total time/iteration: ~5-7 seconds
─────────────────────────────────────────────────
Training 10M steps:   ~13-18 minutes total
Iterations:           153
Updates per second:   ~0.14 Hz (every ~7s)
```

### **What You'll See:**

```
Iteration 1:   fps=32000, total_timesteps=65536,   time_elapsed=7s
Iteration 2:   fps=31500, total_timesteps=131072,  time_elapsed=14s
Iteration 3:   fps=32200, total_timesteps=196608,  time_elapsed=21s
...
Iteration 153: fps=31800, total_timesteps=10027008, time_elapsed=18min
```

---

## 📈 Training Metrics to Watch

### **Good Signs (Learning is Working):**

1. **Explained Variance** → Should increase from ~0 to >0.7
2. **Episode Reward** → Should increase steadily
3. **Policy Loss** → Should decrease initially
4. **Value Loss** → Should decrease then stabilize
5. **Approx KL** → Should stay around target_kl (0.01)

### **Bad Signs (Need to Adjust):**

- ❌ Explained variance stays negative → Increase network size or learning rate
- ❌ Reward doesn't improve → Check reward function
- ❌ Training unstable (high variance) → Reduce learning rate
- ❌ Approx KL always triggers early stopping → Increase target_kl

---

## 💡 Rule of Thumb for n_steps

### **For Different num_envs:**

```
num_envs   →  Recommended n_steps  →  Timesteps/Iter
─────────────────────────────────────────────────────
1          →  2048-4096           →  2K-4K
4          →  512-1024            →  2K-4K
16         →  128-256             →  2K-4K
64         →  32-128              →  2K-8K
256        →  16-64               →  4K-16K
1024       →  8-32                →  8K-33K
2048       →  16-32               →  33K-65K  ← YOU ARE HERE
4096       →  8-16                →  33K-65K
```

**Target: 20K-100K timesteps per iteration for optimal learning**

---

## 🎯 Why This Range Works

### **Too Small (<10K timesteps/iter):**
- ❌ Too frequent updates
- ❌ High variance in gradients
- ❌ Unstable training
- ❌ Overhead from policy switching

### **Sweet Spot (20K-100K timesteps/iter):**
- ✅ Stable gradients
- ✅ Good sample efficiency
- ✅ Fast iteration cycle
- ✅ Proper credit assignment

### **Too Large (>500K timesteps/iter):**
- ❌ Infrequent updates
- ❌ Slow learning
- ❌ Can't adapt quickly
- ❌ Sample inefficiency

---

## 📝 Quick Reference

### **Your Settings (2048 envs):**

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `n_steps` | 4096 | **32** | Enables iterative learning |
| `batch_size` | 1024 | **512** | Matches smaller rollout buffer |
| `total_timesteps` | 1M | **10M** | More training time |
| Iterations | 2 | **153** | Proper learning curve |
| Time/iter | 260s | **~7s** | Much faster feedback |

---

## 🚀 Ready to Train!

### **Launch Training:**

```bash
cd I:\isaaclab

# Stop any running training first (Ctrl+C)

# Start with optimal settings:
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 2048 \
    --headless
```

### **Expected Output:**
```
Iteration 1:   fps=32154, total_timesteps=65536,   time=7s     ✓
Iteration 2:   fps=31892, total_timesteps=131072,  time=14s    ✓
Iteration 3:   fps=32245, total_timesteps=196608,  time=21s    ✓
...
Iteration 153: fps=31967, total_timesteps=10027008, time=18min ✓
```

---

## 🎓 Summary

**Key Takeaway:**
- ❌ **DON'T** use large n_steps with many environments
- ✅ **DO** scale n_steps inversely with num_envs
- 🎯 **Target**: 20K-100K timesteps per iteration

**Your Setup:**
- 2048 envs × 32 steps = **65,536 timesteps/iteration** ✓
- **153 iterations** to train 10M timesteps ✓
- **~18 minutes** total training time ✓
- **Real iterative learning** with proper convergence ✓

🚀 **Now you can train properly and see the policy actually learn!**
