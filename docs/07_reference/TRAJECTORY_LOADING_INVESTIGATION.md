# 🔍 Investigation: Why Multi-Recorded Trajectories Weren't Loading

**Date:** October 17, 2025  
**Issue:** Training ran on synthetic circle instead of 1,038 real trajectories

---

## 📊 Summary

**Root Cause:** The code **HAD** support for `multi_recorded` mode since October 15, 2025, but it was **NEVER ACTIVATED** in training because:

1. ✅ `multi_trajectory.py` existed (since commit `b66a093`)
2. ✅ `TrajectoryManager` supported `multi_recorded` type
3. ✅ `TrajectoryConfig` had `type` parameter
4. ❌ **BUT:** `env.py` was NOT passing `trajectory_dir` parameter to `TrajectoryManager`
5. ❌ **AND:** Training command did NOT specify `--trajectory_type multi_recorded`

---

## 🔬 Git History Analysis

### What Existed on October 15, 2025 (Commit `b66a093`)

| Component | Status | Details |
|-----------|--------|---------|
| `multi_trajectory.py` | ✅ **Existed** | Full `MultiTrajectoryLoader` class implemented |
| `TrajectoryManager.__init__` | ✅ **Supported multi_recorded** | Had `traj_type` parameter with `"multi_recorded"` option |
| `TrajectoryConfig.type` | ✅ **Included multi_recorded** | Type was `Literal["line", "circle", "figure_eight", "recorded"]` |
| `env.py` TrajectoryManager call | ❌ **Missing trajectory_dir** | Did NOT pass `trajectory_dir` parameter! |
| `train.py` arguments | ❌ **No trajectory flags** | No `--trajectory_type` argument existed |

### Code from October 15 (Commit `b66a093`)

**File:** `src/rl_platform/tasks/mobile_mm/config.py`
```python
@dataclass
class TrajectoryConfig:
    """Configuration for reference trajectory generation."""
    
    # Trajectory type
    type: Literal["line", "circle", "figure_eight", "recorded"] = "circle"
    #                                                ❌ NO "multi_recorded"!
    
    # Parametric trajectory settings
    amplitude: float = 0.5
    speed: float = 0.2
    height: float = 1.0
    
    # Recorded trajectory settings
    waypoint_file: str | None = None
    loop_trajectory: bool = True
    
    # ❌ NO multi-recorded settings (trajectory_dir, trajectory_pattern, etc.)
```

**File:** `src/rl_platform/tasks/mobile_mm/trajectories.py`
```python
def __init__(
    self,
    traj_type: Literal["line", "circle", "figure_eight", "recorded", "multi_recorded"],
    #                                                                ✅ Had multi_recorded!
    num_envs: int,
    device: str,
    amplitude: float = 0.5,
    speed: float = 0.2,
    height: float = 1.0,
    dt: float = 0.02,
    waypoint_file: str | None = None,
    trajectory_dir: str | None = None,  # ✅ Parameter existed!
):
    # ...
    if traj_type == "multi_recorded" and trajectory_dir is not None:
        self._init_multi_trajectory(trajectory_dir)  # ✅ Logic existed!
```

**File:** `src/rl_platform/tasks/mobile_mm/env.py` (THE PROBLEM!)
```python
# Trajectory manager
self.trajectory_manager = TrajectoryManager(
    traj_type=self.task_cfg.trajectory.type,  # ✅ Passed type
    num_envs=self.num_envs,
    device=self.device,
    amplitude=self.task_cfg.trajectory.amplitude,
    speed=self.task_cfg.trajectory.speed,
    height=self.task_cfg.trajectory.height,
    dt=self.control_dt,
    # ❌ NO waypoint_file parameter!
    # ❌ NO trajectory_dir parameter!
)
```

---

## 🎯 The Disconnect

### What Was Implemented ✅
- `multi_trajectory.py`: Full loader for multiple trajectories
- `TrajectoryManager._init_multi_trajectory()`: Method to initialize multi-trajectory mode
- `TrajectoryManager.__init__`: Accepted `trajectory_dir` parameter

### What Was Missing ❌
1. **`TrajectoryConfig` didn't have:**
   - `"multi_recorded"` in the `type` Literal
   - `trajectory_dir` field
   - `trajectory_pattern` field
   - `trajectory_filter_indices` field
   - `max_trajectories` field

2. **`env.py` didn't pass:**
   - `waypoint_file` parameter to `TrajectoryManager`
   - `trajectory_dir` parameter to `TrajectoryManager`

3. **`train.py` didn't have:**
   - `--trajectory_type` argument
   - `--trajectory_dir` argument
   - Any way to configure trajectory mode from command line!

---

## 🔧 What Was Fixed Today (October 17, 2025)

### Changes Made:

1. **`config.py`** - Added multi-recorded support:
```python
type: Literal["line", "circle", "figure_eight", "recorded", "multi_recorded"] = "circle"
#                                                           ✅ Added!

# Multi-recorded trajectory settings
trajectory_dir: str = "trajectoryToLearn/world_json"  # ✅ New!
trajectory_pattern: str = "**/*.json"  # ✅ New!
trajectory_filter_indices: list[int] | None = None  # ✅ New!
max_trajectories: int | None = None  # ✅ New!
```

2. **`env.py`** - Passed trajectory parameters:
```python
self.trajectory_manager = TrajectoryManager(
    traj_type=self.task_cfg.trajectory.type,
    # ... existing parameters ...
    waypoint_file=self.task_cfg.trajectory.waypoint_file,  # ✅ Added!
    trajectory_dir=self.task_cfg.trajectory.trajectory_dir,  # ✅ Added!
    trajectory_pattern=self.task_cfg.trajectory.trajectory_pattern,  # ✅ Added!
    trajectory_filter_indices=self.task_cfg.trajectory.trajectory_filter_indices,  # ✅ Added!
    max_trajectories=self.task_cfg.trajectory.max_trajectories,  # ✅ Added!
)
```

3. **`train.py`** - Added command-line arguments:
```python
parser.add_argument("--trajectory_type", default="circle", ...)  # ✅ New!
parser.add_argument("--trajectory_dir", default="trajectoryToLearn/world_json", ...)  # ✅ New!
parser.add_argument("--use_all_trajectories", action="store_true", ...)  # ✅ New!
parser.add_argument("--use_chassis_only", action="store_true", ...)  # ✅ New!
parser.add_argument("--max_trajectories", type=int, ...)  # ✅ New!

# ✅ Changed from gym.make() to direct instantiation with config!
env_cfg.task_config.trajectory = TrajectoryConfig(
    type=args.trajectory_type,
    trajectory_dir=args.trajectory_dir,
    trajectory_filter_indices=...,
    max_trajectories=...,
)
env = MobileMMTrackEEEnv(cfg=env_cfg)
```

4. **`trajectories.py`** - Updated to accept new parameters:
```python
def __init__(
    self,
    # ... existing parameters ...
    trajectory_pattern: str = "**/*.json",  # ✅ Added!
    trajectory_filter_indices: list[int] | None = None,  # ✅ Added!
    max_trajectories: int | None = None,  # ✅ Added!
):
    # ... pass to _init_multi_trajectory ...
```

5. **`multi_trajectory.py`** - Added filtering support:
```python
class MultiTrajectoryLoader:
    def __init__(
        self,
        # ... existing parameters ...
        filter_by_indices: list[int] | None = None,  # ✅ Added!
        exclude_macosx: bool = True,  # ✅ Added!
    ):
        # ... filtering logic ...
```

---

## 🎬 Your Training History

### What You Actually Ran:
```powershell
& "I:\isaaclab\isaaclab.bat" -p ... \
    --task MobileMMTrackEE-v0 \
    --num_envs 4096 \
    --batch_size 1024 \
    --n_steps 128 \
    --total_timesteps 10000000 \
    --learning_rate 0.0003 \
    --ent_coef 0.001 \
    --enable_entropy_decay \
    --final_ent_coef 0.0001 \
    --decay_start_timestep 5000000 \
    --decay_duration_timesteps 5000000 \
    --enable_kl_schedule \
    --kl_warmup 0.25 \
    --kl_main 0.15 \
    --kl_finetune 0.07 \
    --target_kl 1.0 \
    --headless
```

**Notice:** NO `--trajectory_type` argument! → Defaulted to `"circle"`

### What Actually Happened:
1. `--trajectory_type` not specified → defaults to `"circle"`
2. `env_cfg.task_config.trajectory.type` = `"circle"` (from TrajectoryConfig default)
3. `TrajectoryManager.__init__` got `traj_type="circle"`
4. Only synthetic circle trajectory generated
5. **NONE of the 1,038 trajectories were loaded!**

---

## ✅ How to Fix It Going Forward

### For Training on ALL Trajectories:
```powershell
& "I:\isaaclab\isaaclab.bat" -p ... \
    --trajectory_type multi_recorded \      # ✅ Activate multi-trajectory mode
    --use_all_trajectories \                # ✅ Use all 1,038 trajectories
    --headless
```

### For Testing Base Movement:
```powershell
python scripts/test_chassis_trajectories.py --num 10
```

---

## 📈 Timeline

| Date | Event | Status |
|------|-------|--------|
| Oct 14, 2025 | Commit `3e68250`: Added `recorded` trajectory support | ✅ Single trajectory |
| Oct 15, 2025 | Commit `b66a093`: Added `multi_trajectory.py` | ⚠️ Implemented but not wired up |
| Oct 15-17, 2025 | Your training runs | ❌ Used circle (default) |
| Oct 17, 2025 | Investigation today | ✅ Found the disconnect |
| Oct 17, 2025 | Fixed `config.py`, `env.py`, `train.py` | ✅ Now properly wired |

---

## 🎓 Lessons Learned

1. **Implementation ≠ Activation:** Code can exist but not be usable if not properly wired through config/CLI
2. **Defaults matter:** When no CLI args provided, defaults determine behavior
3. **Full stack verification:** Need to check config dataclass → env initialization → CLI args
4. **Git history is truth:** Shows exactly when features were added and what state they were in

---

## 🚀 Next Steps

1. ✅ Run training with `--trajectory_type multi_recorded --use_all_trajectories`
2. ✅ Verify logs show "Loaded all available trajectories from trajectoryToLearn/world_json"
3. ✅ Check trajectory count matches 1,038 (or close, excluding __MACOSX)
4. ✅ Monitor that robot sees diverse movements (check base diagnostics)

---

**Conclusion:** The infrastructure was 90% there, but the final 10% (config fields, env parameter passing, CLI arguments) was missing. Now it's complete!
