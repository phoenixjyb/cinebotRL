# Cinebot RL Roadmap

## 0. Environment Roles, Tooling, and Assets

### Windows (Primary Isaac Lab Runtime)
- Launch Isaac Sim 5.0.0 RC and Isaac Lab via `I:\isaaclab\isaaclab-3090.bat`, ensuring `CUDA_VISIBLE_DEVICES=0` pins workloads to the RTX 3090.
- Convert authoritative URDF/mesh assets from `assets_own/` into Omniverse USDs (store generated USDs under `assets_own/usd/`) before registering them in lab configs; archive validation or reduction outputs under `assets/processed/`.
- Run RL training scripts (SB3, RL-Games), manage TensorBoard/Torch checkpoints, and validate GUI scenes. Isaac Lab packages are already installed editable against the bundled Python 3.11 environment (`isaaclab`, `isaaclab_tasks`, etc.).
- Maintain ROS 2 Humble listener sessions using `scripts\networking\setup_ros2_humble_windows.ps1 -RosInstall I:\ros2\ros2-windows`, keeping `ROS_DOMAIN_ID=55` and Fast DDS bridge ready for cross-host telemetry.

### WSL2 (Ubuntu 22.04, Supporting Automation)
- Use for scripting, data preprocessing, ROS 2 publishers, and future headless Isaac Sim automation. CUDA 12.6 is verified (`nvcc --version` -> 12.6.85).
- Source `.venv_rl311` via `scripts/wsl/activate_rl_env_wsl.sh` for CUDA-enabled PyTorch experiments that do not require the Windows kit.
- Keep Fast DDS bridge aligned with `scripts/networking/configure_fastdds_wsl.sh`, exporting `ROS_DOMAIN_ID=55`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and `FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml`; confirmed listener/talker exchange on 2025‑10‑13.
- Optional headless Isaac Sim install (Linux) remains pending; run `scripts/install_isaacsim_headless_wsl.sh` and `scripts/wsl/check_phase0_prereqs.sh` once ready.

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

# Windows (RL-Games example)
I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/rl_games/train.py \
  task=MobileMMTrackEE-v0 num_envs=1024 headless=true max_iterations=1000
```

Monitor progress via TensorBoard (`logs/sb3` or `logs/rl_games`), and keep checkpoints under `outputs/` (gitignored).

## 4. Export & Deployment

- After convergence, export policies using `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`.
- Convert to TensorRT with `trtexec`, deploy to Jetson Orin with TensorRT EP or ONNX Runtime.
- Store packaged artifacts under `outputs/exports/` and document deployment steps in `docs/tracking/phase1_deployment.md` (TBD).

## 5. Next Stage Checklist

1. **Assets**
   - [x] Stage `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd` and supporting configuration (2025-10-13).
   - [ ] Regenerate `assets/processed/mobile_arm_whole_body/inspection_report.json` after any mesh/URDF edits.
2. **Code**
   - [ ] Scaffold `src/rl_platform/tasks/mobile_mm/` (config, trajectories, scene, observations, rewards, env).
   - [ ] Implement `_get_obs`, `_compute_reward`, `_reset_idx`, and obstacle randomization inside `MobileMMTrackEE`.
   - [ ] Register `MobileMMTrackEE-v0` (or chosen ID) in the training registry and expose config knobs.
3. **Testing**
   - [ ] Run asset inspector + visualization pipeline to sanity-check the USD import.
   - [ ] Execute a smoke RL run (e.g., RL-Games with 128 envs for 200 iterations) to validate reward stability.
   - [ ] Script a collision-avoidance scenario to confirm contact sensors and distance penalties fire correctly.
4. **Automation & Integration**
   - [ ] Decide whether to finish the WSL headless Isaac Sim install (`scripts/install_isaacsim_headless_wsl.sh`).
   - [ ] Extend ROS 2 bridge validation to the new task (ensure Isaac Sim publishes expected topics on domain 55).
   - [ ] Add CI/local hooks (e.g., lint/preflight invoking `scripts/wsl/check_phase0_prereqs.sh`).
