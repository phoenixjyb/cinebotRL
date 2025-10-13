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

### Articulation Configuration Snippet
```python
# src/isaaclab_assets/robots/mobile_mm.py
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = PROJECT_ROOT / "assets_own"

MOBILE_MM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSETS_ROOT / "usd" / "mobile_manipulator_PPR_base_corrected.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=50.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint_x": 0.0, "joint_y": 0.0, "joint_theta": 0.0,
            "left_arm_joint1": 0.0, "left_arm_joint2": 0.0,
            "left_arm_joint3": 0.0, "left_arm_joint4": 0.0,
            "left_arm_joint5": 0.0, "left_arm_joint6": 0.0,
        },
    ),
    actuators={
        "base_xy": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=0.0, damping=300.0,
        ),
        "base_yaw": ImplicitActuatorCfg(
            joint_names_expr=["joint_theta"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
            stiffness=0.0, damping=50.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=2.5,
            stiffness=0.0, damping=25.0,
        ),
    },
)
```

## 2. RL Environment Scaffold

- Observation, reward, and reset logic remain TODOs in `MobileMMTrackEE(DirectRLEnv)`.
- Action mapping: base `vx`/`ω` plus 6 arm joint velocities; nonholonomic constraints handled via kinematics inside `_apply_action`.
- Curriculum ideas and staged reward tweaks captured in `assets/raw/robot_spec.md`.

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
   - [ ] Re-export current URDF to USD under `assets_own/usd/`.
   - [ ] Update inspection report in `assets/processed/` once USD is regenerated.
2. **Code**
   - [ ] Implement `_reset_idx`, `_get_obs`, `_desired_ee_pose` in `MobileMMTrackEE`.
   - [ ] Add reward shaping & collision handling.
3. **Testing**
   - [ ] Run smoke training (SB3/RL-Games) with 128 envs for 200 iters.
   - [ ] Confirm ROS bridge picks up Isaac Sim topics when `--/exts/ros2_bridge/useDomainID=55` is enabled.
4. **Automation**
   - [ ] Decide whether to finish WSL headless Isaac Sim install; if yes, run `scripts/install_isaacsim_headless_wsl.sh`.
   - [ ] Wire CI or local linting using `scripts/wsl/check_phase0_prereqs.sh` and future `scripts/wsl/setup_ci_env.sh` (TBD).
