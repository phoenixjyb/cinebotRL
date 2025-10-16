# CinebotRL Quick Start

**One-page reference for training on RTX 3090.**

---

## ⚡ Train Now (Conservative Start)

```powershell
# Navigate to Isaac Lab
cd I:\isaaclab

# Run training - UPDATED for proper iterative learning
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 2048 `
  --batch_size 512 `
  --n_steps 32 `
  --headless

# Monitor GPU in another terminal
nvidia-smi -l 1
```

**Expected Performance**:
- GPU Memory: 5-7GB (25%)
- Speed: 40K steps/sec
- Training Time: ~18 mins for 10M steps
- Iterations: **153** (proper iterative learning)
- **2x faster** than baseline

---

## 📊 Scaling Strategy

### Phase 1: Conservative (2x faster) ⭐ **START HERE** - UPDATED
```powershell
--num_envs 2048 --batch_size 512 --n_steps 32
```
- GPU: 5-7GB (25%), Speed: 40K steps/sec, Time: ~18 mins, **153 iterations**

### Phase 2: Aggressive (4-6x faster) - UPDATED
```powershell
--num_envs 4096 --batch_size 1024 --n_steps 32
```
- GPU: 10-15GB (60%), Speed: 80-120K steps/sec, Time: 5-10 mins, **76 iterations**

### Phase 3: Maximum (6-8x faster) - UPDATED
```powershell
--num_envs 6144 --batch_size 1536 --n_steps 32
```
- GPU: 18-22GB (80%), Speed: 120-160K steps/sec, Time: 3-5 mins, **51 iterations**

---

## 🔍 Monitoring Commands

```powershell
# GPU utilization
nvidia-smi -l 1

# Training logs
# Check console output for:
# - "fps: XXXXX" (target: 40,000+)
# - "GPU Memory: X.XGB" (target: 5-7GB Phase 1)
```

---

## ⚠️ Important Notes

### Why Low GPU Utilization is OK
- **RL is different from supervised learning**
- Expect: 10-30% compute, 25-60% memory
- NOT a problem - this is normal for PPO

### What to Avoid
- ❌ FP16 mixed precision (breaks PPO stability)
- ❌ Multi-GPU (not needed for single env)
- ❌ `torch.cuda.empty_cache()` (you have 21GB free)

### What to Do
- ✅ Increase batch_size and num_envs
- ✅ Use cudnn.benchmark (already enabled)
- ✅ Monitor with nvidia-smi

---

## 📚 More Details

- **GPU Optimization**: [`04_optimization/RTX3090_REFERENCE_CARD.md`](04_optimization/RTX3090_REFERENCE_CARD.md)
- **Critical Analysis**: [`04_optimization/RTX3090_CRITICAL_ANALYSIS.md`](04_optimization/RTX3090_CRITICAL_ANALYSIS.md)
- **Training Guide**: [`03_training/multi_trajectory_training.md`](03_training/multi_trajectory_training.md)
- **Troubleshooting**: [`07_reference/troubleshooting.md`](07_reference/troubleshooting.md)

---

## 🎯 Expected Results

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| GPU Memory | 2.5GB | 5-7GB | 10-15GB | 18-22GB |
| Speed | 20K/s | 40K/s | 80-120K/s | 120-160K/s |
| Time | 30-40m | 15-20m | 5-10m | 3-5m |
| Speedup | 1x | 2x | 4-6x | 6-8x |

**Start with Phase 1, then scale up once stable!**
