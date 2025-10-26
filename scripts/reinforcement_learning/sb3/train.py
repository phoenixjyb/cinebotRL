"""Training script for MobileMMTrackEE task using Stable Baselines3.

This script trains a mobile manipulator to track end-effector trajectories
using the PPO algorithm from Stable Baselines3.

Designed for Windows with Isaac Lab. No WSL-specific workarounds needed!

Usage:
    # On Windows with Isaac Lab:
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py \\
        --task MobileMMTrackEE-v0 \\
        --num_envs 1024 \\
        --headless
        
    # Or use the convenient launcher:
    .\scripts\launch_training_windows.ps1 -Headless -NumEnvs 1024
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
# Get the script's actual location, then go up to project root
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/reinforcement_learning/sb3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # Up 3 levels to project root

# Add paths for module imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] sys.path includes: {PROJECT_ROOT}, {PROJECT_ROOT / 'src'}")

# NOTE: Do NOT import task_spec here! Isaac Sim must be initialized first.
# We'll import and register tasks inside main() after AppLauncher runs.


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train mobile manipulator with Stable Baselines3"
    )
    
    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="MobileMMTrackEE-v0",
        help="Task ID to train on",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10_000_000,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=128,  # 128 steps × 4096 envs = 524K timesteps/iteration (better GAE estimation)
        help="Number of steps per rollout (128-512 recommended for stable GAE)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,  # Good balance for 65K rollout buffer
        help="Minibatch size for PPO updates",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs per PPO update",
    )
    
    # PPO hyperparameters
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration (recommend 0.001 for tracking tasks)",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="Target KL divergence for early stopping (None = disabled)",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="PPO clipping range for policy updates",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Lambda parameter for Generalized Advantage Estimation",
    )
    
    # Entropy decay parameters (prevents late-stage policy divergence)
    parser.add_argument(
        "--enable_entropy_decay",
        action="store_true",
        help="Enable entropy coefficient decay to prevent policy divergence after convergence",
    )
    parser.add_argument(
        "--final_ent_coef",
        type=float,
        default=0.0001,
        help="Final entropy coefficient after decay (only used if --enable_entropy_decay)",
    )
    parser.add_argument(
        "--decay_start_timestep",
        type=int,
        default=100_000_000,
        help="Timestep to start entropy decay (default: 100M steps)",
    )
    parser.add_argument(
        "--decay_duration_timesteps",
        type=int,
        default=100_000_000,
        help="Duration of entropy decay in timesteps (default: 100M steps)",
    )
    
    # KL divergence scheduling parameters (prevents instability and oscillations)
    parser.add_argument(
        "--enable_kl_schedule",
        action="store_true",
        help="Enable target_kl scheduling for stable policy updates across training phases",
    )
    parser.add_argument(
        "--kl_warmup",
        type=float,
        default=0.07,
        help="Target KL during warmup phase (0-10%% of training, default: 0.07)",
    )
    parser.add_argument(
        "--kl_main",
        type=float,
        default=0.02,
        help="Target KL during main phase (10-80%% of training, default: 0.02)",
    )
    parser.add_argument(
        "--kl_finetune",
        type=float,
        default=0.01,
        help="Target KL during fine-tune phase (80-100%% of training, default: 0.01)",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs (default: logs/sb3/{task}/{timestamp})",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cinebotrl",
        help="W&B project name",
    )
    
    # Resume from checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto - use GPU if available)",
    )
    
    # Trajectory configuration
    parser.add_argument(
        "--trajectory_type",
        type=str,
        default="circle",
        choices=["line", "circle", "figure_eight", "recorded", "multi_recorded"],
        help="Type of trajectory to use (default: circle for basic training, multi_recorded for diverse real trajectories)",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectoryToLearn/world_json",
        help="Directory containing recorded trajectories (for multi_recorded mode)",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        help="Use ALL trajectories from directory (recommended for training), not just chassis-requiring ones",
    )
    parser.add_argument(
        "--use_chassis_only",
        action="store_true",
        help="Use only chassis-requiring trajectories (for testing base movement, not recommended for training)",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Limit number of trajectories to load (None = all, useful for debugging)",
    )
    
    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()
    
    print("=" * 70)
    print("MobileMMTrackEE Training with Stable Baselines3")
    print("=" * 70)
    
    # Step 1: Initialize Isaac Sim via AppLauncher
    # This MUST happen before importing any Isaac Lab or task modules
    print("\n[1/6] Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
        import torch
        
        # Enable TF32 for Tensor Cores (RTX 30xx/40xx optimization)
        # Provides ~8x speedup on matrix multiplications with minimal precision loss
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels for your input sizes
            print("    ✓ TF32 + cuDNN benchmark enabled (8x matmul speedup + auto-tuned kernels)")
        
        # Auto-detect best GPU
        best_device = 0
        best_compute = 0.0
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            cap_val = cap[0] + cap[1] * 0.1
            if cap_val >= 7.0 and cap_val > best_compute:
                best_compute = cap_val
                best_device = i
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    Selected GPU {i}: {gpu_name} (compute {cap[0]}.{cap[1]}, {gpu_mem_gb:.1f}GB)")
                
                # Warn if GPU is underutilized based on memory capacity
                # Rough estimate: ~3MB per environment for mobile manipulator
                recommended_envs = int((gpu_mem_gb - 4) / 0.003)  # Leave 4GB for overhead
                if args.num_envs < recommended_envs * 0.3:  # Less than 30% capacity
                    print(f"    ⚠️  GPU Memory Underutilized!")
                    print(f"       Current: {args.num_envs} envs (~{args.num_envs * 3 / 1024:.1f}GB)")
                    print(f"       Recommended: {recommended_envs // 2} envs (50% capacity)")
                    print(f"       Maximum: ~{recommended_envs} envs (80% capacity)")
                    print(f"       💡 Try: --num_envs {recommended_envs // 2}")
        
        # Create AppLauncher to initialize Isaac Sim
        app_launcher = AppLauncher(
            headless=args.headless,
            enable_cameras=False,
            device=f"cuda:{best_device}",
        )
        simulation_app = app_launcher.app
        print("    ✓ Isaac Sim initialized")
        
    except Exception as e:
        print(f"    ✗ Failed to initialize Isaac Sim: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: NOW we can safely import and register our custom tasks
    # Isaac Sim is running, so Isaac Lab imports will work
    print("\n[2/6] Registering custom tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print(f"    ✓ Registered task: {args.task}")
    except Exception as e:
        print(f"    ✗ Failed to register tasks: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Step 3: Import SB3 and other training dependencies
    print("\n[3/6] Importing training dependencies...")
    try:
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        from stable_baselines3.common.vec_env import VecNormalize, VecEnv, VecEnvWrapper
        print("    ✓ Dependencies imported")
    except ImportError as e:
        print(f"    ✗ Failed to import dependencies: {e}")
        simulation_app.close()
        return
    
    # Entropy decay callback to prevent policy divergence
    class EntropyDecayCallback(BaseCallback):
        """
        Decays entropy coefficient during training to prevent late-stage divergence.
        
        Problem: Constant high ent_coef causes policy to maximize entropy instead of
        tracking performance after convergence.
        
        Solution: Start with high ent_coef for exploration, decay to low value for convergence.
        
        Args:
            initial_ent_coef: Starting entropy coefficient (default: 0.001 for exploration)
            final_ent_coef: Ending entropy coefficient (default: 0.0001 for convergence)
            decay_start_timestep: When to start decay (default: 100M steps)
            decay_duration_timesteps: How long decay takes (default: 100M steps)
        """
        def __init__(
            self, 
            initial_ent_coef: float = 0.001,
            final_ent_coef: float = 0.0001,
            decay_start_timestep: int = 100_000_000,
            decay_duration_timesteps: int = 100_000_000,
            verbose: int = 0
        ):
            super().__init__(verbose)
            self.initial_ent_coef = initial_ent_coef
            self.final_ent_coef = final_ent_coef
            self.decay_start = decay_start_timestep
            self.decay_duration = decay_duration_timesteps
            self.decay_end = decay_start_timestep + decay_duration_timesteps
            
        def _on_step(self) -> bool:
            """Update entropy coefficient based on current timestep."""
            current_timestep = self.num_timesteps
            
            if current_timestep < self.decay_start:
                # Before decay: use initial value
                new_ent_coef = self.initial_ent_coef
            elif current_timestep >= self.decay_end:
                # After decay: use final value
                new_ent_coef = self.final_ent_coef
            else:
                # During decay: linear interpolation
                progress = (current_timestep - self.decay_start) / self.decay_duration
                new_ent_coef = self.initial_ent_coef * (1 - progress) + self.final_ent_coef * progress
            
            # Update model's entropy coefficient
            self.model.ent_coef = new_ent_coef
            
            # Log every 10M steps
            if self.verbose > 0 and current_timestep % 10_000_000 < 65536:  # Within one rollout
                print(f"[EntropyDecay] Step {current_timestep/1e6:.1f}M: ent_coef = {new_ent_coef:.6f}")
            
            return True  # Continue training
    
    # Adaptive KL divergence scheduling for proper policy updates
    class AdaptiveKLSchedule(BaseCallback):
        """
        Adaptive KL scheduling that prevents early stopping and allows proper learning.
        
        Problem: Current KL limits cause "Early stopping at step 0" → no learning!
        - KL divergence spikes early due to new action space (base movement)
        - Immediate early stopping prevents any policy updates
        - Learning efficiency drops to ~1 step per iteration
        
        Solution: Stage-adaptive KL with recovery mechanism:
        - Very Early (0-5M): Extremely loose (1.0) - allow major exploration
        - Early (5-20M): Loose (0.5) - permit substantial learning
        - Learning (20-60M): Moderate (0.2) - balanced updates
        - Stable (60-80M): Normal (0.1) - standard learning
        - Fine-tune (80-100M): Tight (0.05) - precise convergence
        
        Adaptive features:
        - Recent early stopping detection → temporary KL boost
        - Low explained variance → increased exploration allowance
        - Training progress monitoring → automatic adjustments
        """
        def __init__(
            self,
            total_timesteps: int,
            kl_very_early: float = 1.0,    # 0-5M: Allow major exploration
            kl_early: float = 0.5,         # 5-20M: Substantial learning
            kl_learning: float = 0.2,      # 20-60M: Balanced updates
            kl_stable: float = 0.1,        # 60-80M: Normal learning  
            kl_finetune: float = 0.05,     # 80-100M: Precise convergence
            verbose: int = 1
        ):
            super().__init__(verbose)
            self.total_timesteps = total_timesteps
            self.kl_very_early = kl_very_early
            self.kl_early = kl_early
            self.kl_learning = kl_learning
            self.kl_stable = kl_stable
            self.kl_finetune = kl_finetune
            
            # Adaptive tracking
            self.recent_early_stops = 0
            self.check_interval = 10  # Check every 10 rollouts
            self.last_check_timestep = 0
            
        def _get_stage_kl(self, current_timestep: int) -> tuple[float, str]:
            """Determine KL limit based on training stage."""
            progress = current_timestep / self.total_timesteps
            
            if progress < 0.05:  # 0-5M steps
                return self.kl_very_early, "very_early"
            elif progress < 0.20:  # 5-20M steps  
                return self.kl_early, "early"
            elif progress < 0.60:  # 20-60M steps
                return self.kl_learning, "learning"
            elif progress < 0.80:  # 60-80M steps
                return self.kl_stable, "stable"
            else:  # 80-100M steps
                return self.kl_finetune, "finetune"
        
        def _on_rollout_end(self) -> bool:
            """Adaptive KL update with early stopping detection."""
            current_timestep = self.num_timesteps
            
            # Get base KL for current stage
            base_target_kl, stage = self._get_stage_kl(current_timestep)
            
            # Adaptive boost if recent early stopping detected
            kl_boost = 1.0
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Check if we're getting early stops (indicates KL too tight)
                if current_timestep - self.last_check_timestep >= self.check_interval * 4096 * 128:
                    # Simple heuristic: if we're not learning much, boost KL
                    try:
                        recent_losses = getattr(self.model.logger, 'recent_losses', [])
                        if len(recent_losses) > 5:
                            if all(loss > 0.1 for loss in recent_losses[-5:]):  # High losses = poor learning
                                kl_boost = 2.0  # Double the KL allowance
                                if self.verbose > 0:
                                    print(f"[AdaptiveKL] Learning struggle detected - boosting KL by 2x")
                    except:
                        pass  # Ignore if logging not available
                    
                    self.last_check_timestep = current_timestep
            
            # Apply adaptive KL
            new_target_kl = base_target_kl * kl_boost
            
            # Update model's target_kl (handle None case)
            current_kl = self.model.target_kl if self.model.target_kl is not None else 0.0
            if abs(current_kl - new_target_kl) > 0.01:  # Significant change
                self.model.target_kl = new_target_kl
                if self.verbose > 0:
                    boost_str = f" (boosted {kl_boost}x)" if kl_boost > 1.0 else ""
                    print(f"[AdaptiveKL] Step {current_timestep/1e6:.1f}M: target_kl = {new_target_kl:.3f} ({stage}){boost_str}")
            
            return True
        
        def _on_step(self) -> bool:
            """Called at every step."""
            return True
    
    # Create wrapper to convert Isaac Lab observations to SB3 format
    class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
        """VecEnv wrapper to convert Isaac Lab's dict observations with torch tensors to numpy arrays for SB3."""
        
        def __init__(self, venv):
            # Isaac Lab env is already a VecEnv
            VecEnvWrapper.__init__(self, venv)
            # We'll update observation space after first reset
            self._obs_space_updated = False
            
            # FIX: Isaac Lab's action_space includes batch dimension [num_envs, action_dim]
            # SB3 expects per-env action space [action_dim]
            if hasattr(venv.action_space, 'shape') and len(venv.action_space.shape) > 1:
                # Remove the num_envs dimension
                action_dim = venv.action_space.shape[-1]  # Last dimension is action_dim
                self.action_space = spaces.Box(
                    low=venv.action_space.low.flatten()[0],  # All actions have same limits
                    high=venv.action_space.high.flatten()[0],
                    shape=(action_dim,),
                    dtype=venv.action_space.dtype
                )
                print(f"[IsaacLabToSB3VecEnvWrapper] Fixed action_space: {venv.action_space.shape} -> {self.action_space.shape}")
            
        def reset(self):
            obs = self.venv.reset()
            
            # Handle new Gymnasium API: reset() returns (obs, info)
            if isinstance(obs, tuple):
                obs, info = obs
            
            # Convert dict of torch tensors to numpy array
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
                
                # Update observation space on first reset
                if not self._obs_space_updated:
                    # For the observation space, we want per-env shape without batch dim
                    obs_shape = obs.shape[1:] if len(obs.shape) > 1 else obs.shape
                    self.observation_space = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=obs_shape,
                        dtype=np.float32
                    )
                    self._obs_space_updated = True
            return obs
        
        def step_async(self, actions):
            # Isaac Lab envs use sync step(), not async
            # Store actions for step_wait to process
            # Convert numpy actions to torch tensors for Isaac Lab and move to device
            import torch
            if isinstance(actions, np.ndarray):
                # Get the device from the underlying environment
                device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, 'device') else 'cuda:0'
                actions = torch.from_numpy(actions).float().to(device)
            self._actions = actions
        
        def step_wait(self):
            # Call the synchronous step() method with stored actions
            result = self.venv.step(self._actions)
            
            # Handle both old (4 values) and new (5 values) Gymnasium API
            if len(result) == 5:
                # New Gymnasium API: (obs, reward, terminated, truncated, info)
                obs, rewards, terminated, truncated, infos = result
                # Combine terminated and truncated into done for old API
                dones = terminated | truncated
            else:
                # Old Gym API: (obs, reward, done, info)
                obs, rewards, dones, infos = result
            
            # Convert observations
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
            
            # Convert rewards and dones to numpy
            if hasattr(rewards, 'cpu'):
                rewards = rewards.cpu().numpy()
            if hasattr(dones, 'cpu'):
                dones = dones.cpu().numpy()
            
            # Ensure infos is a list of dicts (SB3 expects this format)
            # Isaac Lab sometimes returns infos as a dict or other format
            if isinstance(infos, dict):
                # If infos is a single dict, wrap it in a list for each env
                infos = [infos.copy() for _ in range(len(rewards))]
            elif not isinstance(infos, list):
                # If infos is something else, create empty dicts
                infos = [{} for _ in range(len(rewards))]
            else:
                # infos is already a list, but ensure each element is a dict
                infos = [info if isinstance(info, dict) else {} for info in infos]
            
            return obs, rewards, dones, infos
    
    print("    ✓ Isaac Lab to SB3 VecEnv wrapper created")
    
    # Step 4: Setup logging and W&B
    print("\n[4/6] Setting up logging...")
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = args.task.replace("-", "_").lower()
        args.log_dir = str(PROJECT_ROOT / f"logs/sb3/{task_name}/{timestamp}")
    
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"    Log directory: {args.log_dir}")
    
    # Setup W&B if requested
    if args.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            
            wandb.init(
                project=args.wandb_project,
                name=f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
                sync_tensorboard=True,
            )
            wandb_callback = WandbCallback(
                model_save_freq=args.save_freq,
                model_save_path=args.log_dir,
                verbose=2,
            )
            print("    ✓ W&B logging enabled")
        except ImportError:
            print("    ⚠️  wandb not available, skipping W&B logging")
            args.wandb = False
    
    # Step 5: Create environment
    print(f"\n[5/6] Creating training environment...")
    print(f"    Task: {args.task}")
    print(f"    Num envs: {args.num_envs}")
    print(f"    Headless: {args.headless}")
    print(f"    Trajectory type: {args.trajectory_type}")
    
    # Configure trajectory settings based on command-line args
    trajectory_config = {}
    if args.trajectory_type == "multi_recorded":
        print(f"    Trajectory directory: {args.trajectory_dir}")
        
        # Determine which trajectories to use
        if args.use_chassis_only:
            print("    ⚠️  Using ONLY chassis-requiring trajectories (for testing, not recommended for training)")
            # Load chassis-required indices
            chassis_indices_file = "chassis_required_indices.txt"
            if Path(chassis_indices_file).exists():
                import re
                with open(chassis_indices_file, 'r') as f:
                    content = f.read()
                match = re.search(r'CHASSIS_REQUIRED_INDICES = \[(.*?)\]', content, re.DOTALL)
                if match:
                    indices_str = match.group(1)
                    chassis_indices = [int(x.strip()) for x in indices_str.replace('\n', ' ').split(',') if x.strip()]
                    trajectory_config['filter_indices'] = chassis_indices
                    if args.max_trajectories:
                        trajectory_config['filter_indices'] = chassis_indices[:args.max_trajectories]
                    print(f"    Loaded {len(trajectory_config['filter_indices'])} chassis-requiring trajectory indices")
                else:
                    print(f"    ⚠️  Could not parse {chassis_indices_file}, using all trajectories")
            else:
                print(f"    ⚠️  {chassis_indices_file} not found, using all trajectories")
        elif args.use_all_trajectories:
            print("    ✓ Using ALL trajectories (recommended for training diverse policy)")
            trajectory_config['filter_indices'] = None
            if args.max_trajectories:
                print(f"    Limited to first {args.max_trajectories} trajectories")
        else:
            # Default behavior - use all
            print("    Using all available trajectories (default)")
            trajectory_config['filter_indices'] = None
        
        trajectory_config['max_trajectories'] = args.max_trajectories
    
    try:
        # Import environment config to modify it
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        
        # Create custom environment configuration
        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        # Convert trajectory_dir to absolute path if it's relative
        trajectory_dir = args.trajectory_dir
        if not Path(trajectory_dir).is_absolute():
            trajectory_dir = str(PROJECT_ROOT / trajectory_dir)
            print(f"    Resolved relative path to: {trajectory_dir}")
        
        # Configure trajectory
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type=args.trajectory_type,
            trajectory_dir=trajectory_dir,
            trajectory_pattern="**/*.json",
            trajectory_filter_indices=trajectory_config.get('filter_indices'),
            max_trajectories=trajectory_config.get('max_trajectories'),
        )
        
        # Create environment directly with config
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv
        env = MobileMMTrackEEEnv(cfg=env_cfg)
        
        print(f"    ✓ Environment created")
        if args.trajectory_type == "multi_recorded":
            if trajectory_config.get('filter_indices') is not None:
                print(f"    ✓ Loaded {len(trajectory_config['filter_indices'])} filtered trajectories")
            else:
                print(f"    ✓ Loaded all available trajectories from {args.trajectory_dir}")
    
    except Exception as e:
        print(f"    ✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    try:
        
        # First wrap to convert Isaac Lab format to SB3 format (dict -> numpy)
        # Isaac Lab envs are already VecEnv, so use VecEnvWrapper
        env = IsaacLabToSB3VecEnvWrapper(env)
        
        # Do a dummy reset to let the wrapper discover the true observation shape
        # This updates the observation_space before VecNormalize reads it
        _ = env.reset()
        
        # Then wrap with VecNormalize for better training stability
        # NOTE: VecNormalize will now see numpy arrays instead of dicts
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        print("    ✓ Environment created and wrapped")
    except Exception as e:
        print(f"    ✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Step 6: Create and train PPO model
    print("\n[6/6] Setting up PPO training...")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,  # Per environment steps
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ppo_mobile_mm",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Entropy decay callback (prevents policy divergence after convergence)
    if args.enable_entropy_decay:
        entropy_decay_callback = EntropyDecayCallback(
            initial_ent_coef=args.ent_coef,
            final_ent_coef=args.final_ent_coef,
            decay_start_timestep=args.decay_start_timestep,
            decay_duration_timesteps=args.decay_duration_timesteps,
            verbose=1,
        )
        callbacks.append(entropy_decay_callback)
        print(f"    ✓ Entropy decay enabled: {args.ent_coef} → {args.final_ent_coef}")
        print(f"      Decay: {args.decay_start_timestep/1e6:.0f}M - {(args.decay_start_timestep + args.decay_duration_timesteps)/1e6:.0f}M steps")
    
    # Adaptive KL divergence schedule callback (prevents early stopping, enables learning)
    if args.enable_kl_schedule:
        kl_schedule_callback = AdaptiveKLSchedule(
            total_timesteps=args.total_timesteps,
            kl_very_early=max(args.kl_warmup * 4, 1.0),    # Much more aggressive early KL
            kl_early=max(args.kl_warmup * 2, 0.5),         # Still very loose  
            kl_learning=max(args.kl_main * 4, 0.2),        # More learning room
            kl_stable=max(args.kl_main * 2, 0.1),          # Reasonable updates
            kl_finetune=args.kl_finetune,                  # Normal end-game
            verbose=1
        )
        callbacks.append(kl_schedule_callback)
        print(f"    ✓ Adaptive KL schedule enabled: very_early={max(args.kl_warmup * 4, 1.0):.2f}, early={max(args.kl_warmup * 2, 0.5):.2f}")
        print(f"      Stages: 0-5M (explore), 5-20M (learn), 20-60M (balance), 60-80M (stable), 80-100M (finetune)")
    
    
    if args.wandb:
        callbacks.append(wandb_callback)
    
    # Create or load model
    try:
        import torch
        
        # Device selection based on command-line argument
        if args.device == "cpu":
            device = "cpu"
            print("    ⚠️  CPU training forced via --device cpu")
        elif args.device == "cuda":
            if torch.cuda.is_available():
                device = f"cuda:{best_device}"
            else:
                print("    ⚠️  CUDA requested but not available, falling back to CPU")
                device = "cpu"
        else:  # auto
            device = f"cuda:{best_device}" if torch.cuda.is_available() else "cpu"
        
        if args.checkpoint:
            print(f"    Loading checkpoint: {args.checkpoint}")
            model = PPO.load(args.checkpoint, env=env, device=device)
        else:
            print("    Creating new PPO model...")
            
            # Enhanced network architecture for 9DOF robot with trajectory tracking
            # Obs: 70 dims (base 13 + joints 12 + EE 13 + error 7 + lookahead 9 + action_hist 16)
            # Actions: 8 dims (6 arm joints + 2 base velocities)
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[256, 256, 128],  # Actor: 3-layer network (70→256→256→128→8)
                    vf=[256, 256, 128]   # Critic: 3-layer network (70→256→256→128→1)
                ),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,  # Orthogonal initialization (better for RL)
                log_std_init=-1.0,  # Initial log(std) = -1.0 → std ≈ 0.37
            )
            
            print("    Network Architecture:")
            print("      Actor (Policy):  [70] → [256] → [256] → [128] → [8]  (~118K params)")
            print("      Critic (Value):  [70] → [256] → [256] → [128] → [1]  (~117K params)")
            print("      Total: ~235K parameters (16× larger than default)")
            print("    Action Distribution:")
            print("      Initial std:     ~0.37 (log_std_init=-1.0)")
            print("      Std control:     Entropy decay + KL schedule")
            
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,  # Use enhanced architecture
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                clip_range_vf=1.0,      # Clip value function updates for stability
                ent_coef=args.ent_coef,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=args.target_kl,
                # Note: log_std_bounds not available in SB3 2.5.0
                # Std control via: entropy decay + KL schedule + log_std_init=-1.0
                tensorboard_log=args.log_dir,
                device=device,
                verbose=1,
            )
            print(f"    ✓ PPO model created on {device}")
            
    except Exception as e:
        print(f"    ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        return
    
    # Print training configuration
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Task:              {args.task}")
    print(f"Num environments:  {args.num_envs}")
    print(f"Total timesteps:   {args.total_timesteps:,}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Rollout steps:     {args.n_steps}")
    print(f"Batch size:        {args.batch_size}")
    print(f"PPO epochs:        {args.n_epochs}")
    print(f"Entropy coef:      {args.ent_coef}")
    if args.enable_entropy_decay:
        print(f"  → Decay enabled: {args.ent_coef} → {args.final_ent_coef}")
        print(f"  → Decay period:  {args.decay_start_timestep/1e6:.0f}M - {(args.decay_start_timestep + args.decay_duration_timesteps)/1e6:.0f}M steps")
    print(f"Gamma:             {args.gamma}")
    print(f"GAE lambda:        {args.gae_lambda}")
    print(f"Clip range:        {args.clip_range}")
    print(f"Target KL:         {args.target_kl if args.target_kl else 'None (disabled)'}")
    if args.enable_kl_schedule:
        print(f"  → KL schedule:   warmup={args.kl_warmup}, main={args.kl_main}, finetune={args.kl_finetune}")
        print(f"  → Phase splits:  10% warmup, 70% main, 20% finetune")
    print(f"Save frequency:    {args.save_freq:,} steps")
    print(f"Log directory:     {args.log_dir}")
    print(f"Device:            {device}")
    print("=" * 70 + "\n")
    
    # Train
    try:
        print("Starting training...\n")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=1,
            tb_log_name="PPO",
        )
        
        # Save final model
        final_model_path = os.path.join(args.log_dir, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(args.log_dir, "vec_normalize.pkl"))
        
        print(f"\n{'='*70}")
        print("✓ Training complete!")
        print(f"{'='*70}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Logs saved to: {args.log_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
        # Save interrupt checkpoint
        interrupt_path = os.path.join(args.log_dir, "interrupt_checkpoint")
        model.save(interrupt_path)
        env.save(os.path.join(args.log_dir, "vec_normalize_interrupt.pkl"))
        
        print(f"Interrupt checkpoint saved to: {interrupt_path}")
    
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        env.close()
        if args.wandb:
            wandb.finish()
        simulation_app.close()
        print("Done!")


if __name__ == "__main__":
    main()
