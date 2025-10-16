# Cinebot RL Roadmap

## 0. Environment Roles, Tooling, and Assets

### Windows (Primary RL Training Platform) ✅ **FULLY OPERATIONAL**
**Status (2025-10-15):** Training verified working with all compatibility fixes applied

- **Isaac Lab 2.2.0** at `I:\isaaclab` with Python 3.11.13, torch 2.7.0+cu128, Isaac Sim 5.0.0-rc.45
- **Dual-GPU Configuration:**
  - RTX 3090 (Device 0, compute 8.6) → RL training (auto-detected)
  - Quadro P2000 (Device 1, compute 6.1) → display/GUI
- **Training Framework:** Stable Baselines3 PPO with custom `IsaacLabToSB3VecEnvWrapper` bridging Isaac Lab → SB3
- **Launch Method:** PowerShell scripts with automatic GPU detection (no manual CUDA_VISIBLE_DEVICES needed)
  ```powershell
  .\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless
  ```
- **ROS 2 Humble** at `I:\ros2humble\ros2-windows` (Python 3.10) for Isaac Sim topic bridging (Domain 55)
- **Asset Management:** Convert URDF/meshes from `assets_own/` to USD format in `assets_own/usd/` before training

**Migration Complete:** All WSL-specific code removed. Training runs natively on Windows with 12+ compatibility fixes documented in [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md). Quick start guide available in [START_TRAINING_NOW.md](START_TRAINING_NOW.md).

### WSL2 (Ubuntu 22.04, Optional Support)
**Not required for training** - use only for data analysis, monitoring, or ROS 2 automation

- **ROS 2 Humble** (system Python 3.10, `/opt/ros/humble`) for topic monitoring and automation scripts
- **`.venv_rl311`** (Python 3.11, PyTorch 2.6.0+cu124, SB3 2.7.0) for data analysis and visualization (not training)
- CUDA 12.6.85 available via GPU passthrough, but Windows handles all training
- Fast DDS bridge config: `scripts/networking/configure_fastdds_wsl.sh` (Domain 55)
- Optional headless Isaac Sim install: `scripts/install_isaacsim_headless_wsl.sh` (not needed for current workflow)

### Asset Provenance
- `assets_own/`: single source of truth for URDFs and original meshes; keep under version control.
- `assets_own/usd/`: tracked exports used by Isaac Lab.
- `assets/processed/`: inspection reports or reduced meshes generated from the source assets.

## 1. Robot Configuration & Isaac Lab Setup

The calibrated URDF is in `assets_own/mobile_manipulator_PPR_base_corrected.urdf` with STL meshes (note millimetre scale). During conversion:
- Apply uniform `0.001` mesh scale in the Isaac Sim Asset Converter so USD lands under `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`.
- Validate the generated USD with the asset inspector (`src/asset_inspector`) and capture reports in `assets/processed/mobile_arm_whole_body/`.

### Articulation Configuration Snippet (to be added)
```python
# src/rl_platform/robots/mobile_mm.py
from rl_platform.robots.mobile_mm import get_mobile_mm_assets

assets = get_mobile_mm_assets()
assets.validate()

# Later, when defining the ArticulationCfg:
# spawn=sim_utils.UsdFileCfg(usd_path=str(assets.usd_path), ...)
```

## 2. RL Environment Scaffold

### 2.1 Module Layout
Create a task-specific package to keep code organized:
```
src/rl_platform/tasks/mobile_mm/
    __init__.py
    config.py           # dataclasses / OmegaConf bindings for tunables
    trajectories.py     # reference generators and waypoint loaders
    scene.py            # obstacle spawning, sensor setup
    observations.py     # functions that assemble observation tensors/dicts
    rewards.py          # reusable reward terms & penalty helpers
    env.py              # DirectRLEnv subclass (MobileMMTrackEE)
```
Register the task ID (e.g., `MobileMMTrackEE-v0`) in the training registry (`src/task_spec.py` or equivalent) so SB3/RL-Games launchers can resolve it.

### 2.2 Trajectory Library
- Implement parametric references (line, circle, figure-eight) with controllable speed and amplitude.
- Allow playback of recorded waypoints stored under `assets/processed/trajectories/` (preprocess data via WSL scripts if convenient).
- Provide a `TrajectoryManager` that yields the current target pose plus optional lookahead samples for preview control.
- Include curriculum knobs to vary trajectory length/speed as training progresses.

### 2.3 Scene & Collision Instrumentation
- Instantiate the mobile manipulator using `rl_platform.robots.mobile_mm.get_mobile_mm_assets()`.
- Spawn static or randomized obstacles via `InteractiveSceneCfg` primitives; maintain per-env seeds for reproducibility.
- Attach contact sensors (base, arm) and, if needed, distance/ray sensors for early collision warnings.
- Track obstacle metadata (AABBs, safety radii) so rewards/terminations can reference minimum distance values.

### 2.4 Observation Composition
- In `_get_obs()`, include base pose/velocity, joint states, end-effector pose/velocity, and tracking error vectors.
- Add lookahead reference deltas, contact flags, minimum obstacle distance, and action history when rate penalties are planned.
- Return a dictionary/tensor compatible with the chosen backend (SB3 expects numpy arrays; RL-Games uses torch tensors).

### 2.5 Reward & Penalty Design
- Build reusable terms in `rewards.py`: position/orientation tracking, progress bonuses, action magnitude/rate penalties, collision penalties, stability terms.
- Keep scale factors in `config.py` so they can be tuned via YAML/CLI without code edits.
- In `_compute_reward()`, combine the scaled components and log diagnostics (e.g., collision counts) to `self.extras`.

### 2.6 Reset, Randomization, Curriculum
- `_reset_idx()` should reset the trajectory phase, randomize obstacle placements (if enabled), and jitter initial joint states.
- Apply domain randomization to physics parameters (mass, friction, torque limits) using Isaac Lab RNG utilities.
- Drive curriculum staging via config flags—start with sparse obstacles and short trajectories, then increase difficulty once reward thresholds are met.

### 2.7 Integration Plan
- Feed `assets.usd_path` from `get_mobile_mm_assets()` into the `ArticulationCfg` spawn block.
- Update SB3/RL-Games training configs to reference the new task ID.
- Document environment parameters and reward settings in a new `docs/tracking/phase1_task_design.md` once the scaffold is implemented.
## 3. Training Workflow

### ROS 2 Bridge Health Check
- Windows: `scripts\networking\setup_ros2_humble_windows.ps1 -RosInstall I:\ros2\ros2-windows`, then `ros2 run demo_nodes_cpp listener`.
- WSL: `scripts/networking/configure_fastdds_wsl.sh`, export DDS env vars, run `ros2 run demo_nodes_cpp talker`.
- Verified 2025‑10‑13: Windows listener received `/chatter` (`Hello World: 427-441`) from WSL talker across domain 55.

### Launch Commands
```bash
# Windows (SB3 example)
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py \
  --task MobileMMTrackEE-v0 --num_envs 1024 --headless

## 3. Running RL Training

### Quick Start (Windows Training)

**Option 1: Quick Start (Recommended)**
```powershell
# Launch training with default settings (64 envs, headless, 5M steps)
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless

# Or with custom parameters
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 128 -TotalTimesteps 10000000
```

**Option 2: Direct Invocation**
```powershell
# Direct Isaac Lab launcher (auto GPU detection)
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
  --task MobileMMTrackEE-v0 --num_envs 64 --headless true --total_timesteps 5000000
```

**Option 3: Combined Commit & Train**
```powershell
# Commit all changes and start training in one command
.\scripts\commit_and_start_training.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless
```

### Monitoring Training

```powershell
# In a separate PowerShell window - view logs
.\scripts\monitor_training.ps1 -Mode logs

# Watch GPU usage
.\scripts\monitor_training.ps1 -Mode gpu

# Launch TensorBoard
.\scripts\monitor_training.ps1 -Mode tensorboard

# Show all monitoring options
.\scripts\monitor_training.ps1 -Mode all
```

**Expected Timeline:**
- 64 envs: ~60 minutes for 100K steps, ~8-10 hours for 5M steps
- 128 envs: ~30 minutes for 100K steps, ~4-5 hours for 5M steps

**Checkpoints:** Saved to `I:\isaaclab\logs\sb3\MobileMMTrackEE-v0\<timestamp>/checkpoints/`

**Architecture:** Custom `IsaacLabToSB3VecEnvWrapper` handles all conversions between Isaac Lab (dict observations, torch tensors, GPU) and Stable Baselines3 (numpy arrays, CPU). See [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md) for technical details.

Monitor progress via TensorBoard (logs at `I:\isaaclab\logs\sb3`), and keep checkpoints under `I:\isaaclab\logs/` (gitignored).
```

Monitor progress via TensorBoard (`logs/sb3` or `logs/rl_games`), and keep checkpoints under `outputs/` (gitignored).

## 4. Export & Deployment

- After convergence, export policies using `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`.
- Convert to TensorRT with `trtexec`, deploy to Jetson Orin with TensorRT EP or ONNX Runtime.
- Store packaged artifacts under `outputs/exports/` and document deployment steps in `docs/tracking/phase1_deployment.md` (TBD).

## 5. Next Stage Checklist

1. **Assets**
   - [x] Stage `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd` and supporting configuration (2025-10-13)
   - [ ] Regenerate `assets/processed/mobile_arm_whole_body/inspection_report.json` after any mesh/URDF edits

2. **Code & Environment**
   - [x] ✅ **Windows Training Operational** (2025-10-15)
     - [x] Isaac Lab 2.2.0 + Stable Baselines3 integration complete
     - [x] All 12+ compatibility issues resolved (see [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md))
     - [x] Custom `IsaacLabToSB3VecEnvWrapper` implemented for Isaac Lab ↔ SB3 bridging
     - [x] GPU auto-detection implemented (no manual CUDA_VISIBLE_DEVICES needed)
     - [x] PowerShell launcher scripts created ([launch_training_windows.ps1](scripts/launch_training_windows.ps1))
     - [x] Monitoring utilities created ([monitor_training.ps1](scripts/monitor_training.ps1))
     - [x] Training verified working with `MobileMMTrackEE-v0` task
   - [x] Scaffold `src/rl_platform/tasks/mobile_mm/` (config, trajectories, scene, observations, rewards, env)
   - [x] Implement `_get_obs`, `_compute_reward`, `_reset_idx` in `MobileMMTrackEE`
   - [x] Register `MobileMMTrackEE-v0` in training registry
   - [ ] Add obstacle randomization inside `MobileMMTrackEE` environment

3. **Testing**
   - [x] Run asset inspector + visualization pipeline to validate USD import
   - [x] Execute RL training runs (SB3 with 64-128 envs) to validate reward stability
   - [ ] Script a collision-avoidance scenario to confirm contact sensors and distance penalties

4. **Automation & Integration**
   - [ ] Optional: Finish WSL headless Isaac Sim install (`scripts/install_isaacsim_headless_wsl.sh`) - not required for training
   - [ ] Extend ROS 2 bridge validation to the new task (ensure Isaac Sim publishes expected topics on domain 55)
   - [ ] Add CI/local hooks (e.g., lint/preflight invoking `scripts/wsl/check_phase0_prereqs.sh`)

**Priority:** Training infrastructure complete ✅ Next focus: Reward tuning, obstacle avoidance, deployment pipeline
