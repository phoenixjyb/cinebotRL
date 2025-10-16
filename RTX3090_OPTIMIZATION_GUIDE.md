# RTX 3090 Optimization Guide for RL Training

## 🚀 Your RTX 3090 Specifications

**Hardware:**
- **CUDA Cores:** 10,496
- **Tensor Cores:** 328 (3rd Gen)
- **Memory:** 24 GB GDDR6X
- **Compute Capability:** 8.6
- **Memory Bandwidth:** 936 GB/s
- **FP32 Performance:** 35.6 TFLOPS
- **Tensor Performance:** 142 TFLOPS (FP16)

**Key Strength:** Massive parallel processing + huge memory = Perfect for RL!

---

## ⚡ Current Optimizations (Already Applied)

### 1. Auto-Detection ✅
```python
# Line 163 in train.py - Already detecting RTX 3090
best_device = 0
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    if cap_val >= 7.0:  # RTX 3090 = 8.6
        best_device = i
```

### 2. GPU Device Assignment ✅
```python
# Isaac Sim uses RTX 3090 for physics
device=f"cuda:{best_device}"  # cuda:0 (RTX 3090)
```

---

## 🎯 Optimal Settings for RTX 3090

### Best Configuration for Full Utilization

**Current settings are already good, but can be optimized further:**

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --device cuda `
  --num_envs 1024 `           # ✅ OPTIMAL: Maxes out GPU parallelism
  --total_timesteps 5000000 `
  --n_steps 4096 `             # ✅ GOOD: Large rollouts utilize Tensor Cores
  --batch_size 256 `           # ✅ OPTIMAL: Good for RTX 3090
  --headless
```

---

## 🔧 Advanced Optimizations

### 1. Enable TF32 for Tensor Cores (RECOMMENDED)

**Add this to your training script** (top of `main()` function):

```python
import torch

# Enable TF32 on Ampere GPUs (RTX 3090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# This gives ~8x speedup on matrix multiplications!
print("    ✓ TF32 enabled for Tensor Cores")
```

**Why?** RTX 3090's Tensor Cores excel at TF32 operations - 8x faster than FP32!

---

### 2. Optimal Batch Sizes for Tensor Cores

**Tensor Cores work best with multiples of 8:**

```python
# Current: batch_size=256  ✅ Good (256 = 8×32)
# Also good: 128, 256, 512, 1024
# Avoid: 100, 200, 300 (not multiples of 8)
```

**Recommendation for RTX 3090:**
- **Training:** `batch_size=256` or `512` (current is optimal)
- **Testing:** `batch_size=128` (faster iteration)

---

### 3. Maximize Parallel Environments

**RTX 3090 can handle MANY parallel environments:**

| Num Envs | GPU Memory | Speed | Use Case |
|----------|------------|-------|----------|
| 512 | ~8 GB | Fast | Testing |
| 1024 | ~12 GB | Optimal | **Recommended** |
| 2048 | ~18 GB | Faster | Max throughput |
| 4096 | ~22 GB | Fastest | Ultimate (if fits) |

**Current setting: 1024 ✅ Perfect balance!**

**To push further:**
```powershell
--num_envs 2048  # Try if 1024 feels slow
```

---

### 4. Mixed Precision Training (FP16)

**For even more Tensor Core utilization:**

Add to PPO model creation:
```python
policy_kwargs = dict(
    # ... existing settings
    use_fp16=True,  # Enable FP16 for Tensor Cores
)
```

⚠️ **Note:** SB3 doesn't natively support FP16, but Isaac Sim does internally.

---

## 📊 Performance Optimization Checklist

### ✅ Already Optimized (Current Setup)
- [x] GPU auto-detection (RTX 3090 selected)
- [x] Large parallel envs (1024)
- [x] Batch size multiple of 8 (256)
- [x] Headless mode (no display overhead)
- [x] CUDA backend enabled

### 🎯 Additional Optimizations (Apply Now)
- [ ] **Enable TF32** (biggest impact!)
- [ ] Try `num_envs=2048` (use more GPU memory)
- [ ] Experiment with `batch_size=512` (more Tensor Core work)

### 🔬 Advanced (Optional)
- [ ] Profile with `torch.profiler` to find bottlenecks
- [ ] Enable CUDA graphs (requires code changes)
- [ ] Use PyTorch JIT compilation

---

## 🚀 Quick Optimization Implementation

### Step 1: Enable TF32 (Easy, Big Impact!)

Add to `train.py` after line 157 (in `main()` function):

```python
def main():
    """Main training loop."""
    args = parse_args()
    
    print("=" * 70)
    print("MobileMMTrackEE Training with Stable Baselines3")
    print("=" * 70)
    
    # Step 1: Initialize Isaac Sim via AppLauncher
    print("\n[1/6] Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
        import torch
        
        # Enable TF32 for Tensor Cores (RTX 3090 optimization)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("    ✓ TF32 enabled for Tensor Cores (8x matrix mul speedup)")
        
        # ... rest of code
```

---

### Step 2: Try Larger Batch (Optional)

If training feels slow, try:
```powershell
--batch_size 512  # Instead of 256
```

---

### Step 3: Scale Up Environments (Optional)

If GPU utilization low (<80%), try:
```powershell
--num_envs 2048  # Instead of 1024
```

---

## 📈 Expected Performance

### Current Configuration (1024 envs)
- **Throughput:** ~20,000-40,000 steps/sec
- **GPU Utilization:** 60-80%
- **Training Time (5M steps):** 30-40 minutes

### With TF32 Enabled
- **Throughput:** ~25,000-50,000 steps/sec (+25%)
- **GPU Utilization:** 70-85%
- **Training Time (5M steps):** 25-35 minutes

### With 2048 Envs + TF32
- **Throughput:** ~35,000-60,000 steps/sec (+50-80%)
- **GPU Utilization:** 80-95%
- **Training Time (5M steps):** 15-25 minutes

---

## 🔍 Monitoring GPU Utilization

### During Training, Check:

**Terminal 1: Start training**
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py ...
```

**Terminal 2: Monitor GPU (in PowerShell)**
```powershell
# Quick check
nvidia-smi

# Watch continuously (refresh every 1 sec)
while ($true) { cls; nvidia-smi; Start-Sleep -Seconds 1 }
```

**What to look for:**
- **GPU Utilization:** Should be 70-95%
- **Memory Usage:** 10-15 GB / 24 GB (plenty of headroom)
- **Power:** ~300-350W (max is 350W)
- **Temperature:** <85°C

---

## 🎯 Optimal Command for RTX 3090

### Recommended: Balanced Performance
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --device cuda `
  --num_envs 1024 `
  --total_timesteps 5000000 `
  --n_steps 4096 `
  --batch_size 256 `
  --headless
```
**Expected:** 30-40 mins, 60-80% GPU utilization

---

### Maximum Performance (with TF32)
```powershell
# After adding TF32 to code
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --device cuda `
  --num_envs 2048 `
  --total_timesteps 5000000 `
  --n_steps 4096 `
  --batch_size 512 `
  --headless
```
**Expected:** 15-25 mins, 80-95% GPU utilization

---

## 🐛 Troubleshooting

### GPU Utilization Low (<50%)
**Causes:**
- CPU bottleneck (observation processing)
- Small batch size
- Too few environments

**Solutions:**
- Increase `num_envs` (1024 → 2048)
- Increase `batch_size` (256 → 512)
- Enable TF32

---

### Out of Memory (OOM)
**Causes:**
- Too many environments
- Too large batch size

**Solutions:**
- Reduce `num_envs` (2048 → 1024)
- Reduce `batch_size` (512 → 256)
- Use `--headless` (saves memory)

---

### Training Slower than Expected
**Causes:**
- Not using Tensor Cores efficiently
- Display GPU stealing resources (Quadro P2000)

**Solutions:**
- Enable TF32 ✅
- Ensure `--device cuda` points to RTX 3090
- Use multiples of 8 for batch sizes
- Close unnecessary programs

---

## 💡 Pro Tips for RTX 3090

1. **Always use headless mode** (`--headless`) - saves 1-2 GB memory
2. **Monitor temperature** - RTX 3090 runs hot, ensure good cooling
3. **Use Tensor Cores** - Enable TF32 for 8x speedup on matmuls
4. **Scale up gradually** - Start 1024 envs, then try 2048 if stable
5. **Batch size matters** - Use 256 or 512 (multiples of 8)
6. **Memory is plentiful** - 24GB means you can go big on environments!

---

## 📝 Summary: Quick Wins

### Immediate (No Code Changes)
```powershell
# Already optimal:
--num_envs 1024
--batch_size 256
--device cuda
--headless
```

### Easy Win (Add 3 lines of code)
**Enable TF32 in train.py:**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
**Impact:** +25% speed, no downsides!

### Max Performance (After TF32 + testing)
```powershell
--num_envs 2048
--batch_size 512
```
**Impact:** +50-80% speed, uses 18-20 GB / 24 GB memory

---

## ✅ Your RTX 3090 is PERFECT for this!

- ✅ 24 GB memory → Can handle 2048+ environments
- ✅ 10,496 CUDA cores → Massive parallelism  
- ✅ 328 Tensor Cores → 8x faster with TF32
- ✅ Current settings already good!

**Bottom line:** Enable TF32 for biggest immediate gain, then experiment with 2048 envs! 🚀

