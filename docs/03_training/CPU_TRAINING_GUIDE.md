# CPU Training Configuration

## 🖥️ Running Training on CPU

### Why Train on CPU?

**Use CPU when:**
- Testing code changes quickly (faster startup, no GPU initialization)
- GPU is being used for something else (display, other training)
- Debugging issues (easier to debug on CPU)
- Small-scale experiments (few environments)

**Limitations:**
- ⚠️ **MUCH slower** than GPU training (10-100x slower)
- ⚠️ Only practical for **small num_envs** (1-16)
- ⚠️ Not recommended for full training runs

---

## 📝 Commands

### Test Run on CPU (Quick Code Testing)

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 4 `
  --total_timesteps 10000 `
  --n_steps 512 `
  --batch_size 64 `
  --device cpu `
  --headless
```

**Use case:** Test that code runs without errors (2-3 mins)

---

### Small Training Run on CPU

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 8 `
  --total_timesteps 100000 `
  --n_steps 2048 `
  --batch_size 128 `
  --device cpu `
  --headless
```

**Use case:** Small experiment or hyperparameter testing (10-15 mins)

---

### Default (Auto) - Use GPU if Available

```powershell
# This is the default behavior - no --device flag needed
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 64 `
  --total_timesteps 500000 `
  --n_steps 4096 `
  --batch_size 256 `
  --headless
  # Automatically uses GPU if available
```

---

### Force GPU (Fail if Not Available)

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --total_timesteps 5000000 `
  --n_steps 4096 `
  --batch_size 256 `
  --device cuda `
  --headless
```

---

## ⚖️ Performance Comparison

| Configuration | Speed (steps/sec) | Use Case |
|---------------|-------------------|----------|
| **CPU, 4 envs** | ~50-100 | Code testing |
| **CPU, 16 envs** | ~100-200 | Small experiments |
| **GPU, 64 envs** | ~5,000-10,000 | Test training |
| **GPU, 1024 envs** | ~20,000-40,000 | Full training |

---

## 🎯 Recommended Configurations

### Quick Code Test (CPU - 2 mins)
```powershell
--device cpu --num_envs 4 --total_timesteps 10000 --n_steps 512 --batch_size 64
```

### Hyperparameter Test (CPU - 10 mins)
```powershell
--device cpu --num_envs 8 --total_timesteps 100000 --n_steps 2048 --batch_size 128
```

### Test Training (GPU - 5-10 mins)
```powershell
--device cuda --num_envs 64 --total_timesteps 500000 --n_steps 4096 --batch_size 256
```

### Full Training (GPU - 30-40 mins)
```powershell
--device cuda --num_envs 1024 --total_timesteps 5000000 --n_steps 4096 --batch_size 256
```

---

## 📊 CPU vs GPU Training Trade-offs

### CPU Advantages ✅
- Faster startup (no GPU initialization)
- Easier debugging
- Leaves GPU free for display/other tasks
- More memory available
- Deterministic behavior

### GPU Advantages ✅
- **10-100x faster** training
- Can handle many parallel environments (1024+)
- Tensor operations optimized
- Essential for production training
- Better for large batch sizes

---

## 🔍 Checking Device Usage

**During training, you'll see:**

```
✓ PPO model created on cpu
```
or
```
✓ PPO model created on cuda:0
```

**In logs, check:**
```
Device:            cpu
```
or
```
Device:            cuda:0
```

---

## 💡 Best Practices

### For Development/Testing:
1. **Start with CPU** (4 envs, 10K steps) - verify code works
2. **Then try GPU** (64 envs, 500K steps) - verify performance
3. **Then scale up** (1024 envs, 5M steps) - full training

### For Production Training:
- **Always use GPU** with `--device cuda`
- Use **1024 envs** minimum
- Train for **5M+ steps**
- Monitor via TensorBoard

---

## 🚨 Common Issues

### Issue: "CUDA requested but not available"
**Solution:** GPU not detected, will fall back to CPU
```powershell
# Check CUDA availability:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Issue: CPU training too slow
**Solution:** Reduce num_envs and timesteps
```powershell
--num_envs 4 --total_timesteps 10000
```

### Issue: Out of memory on CPU
**Solution:** Reduce num_envs or batch_size
```powershell
--num_envs 4 --batch_size 64
```

---

## 📖 Device Parameter Options

| Value | Behavior |
|-------|----------|
| `--device auto` | Use GPU if available, else CPU (default) |
| `--device cpu` | Force CPU, even if GPU available |
| `--device cuda` | Force GPU, error if not available |
| *(no flag)* | Same as `auto` |

---

## ✅ Summary

**Quick test?** → `--device cpu --num_envs 4 --total_timesteps 10000`

**Real training?** → `--device cuda --num_envs 1024 --total_timesteps 5000000`

**The flag is now available - use it when needed!** 🎉

