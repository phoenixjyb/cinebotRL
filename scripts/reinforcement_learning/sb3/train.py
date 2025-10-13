"""Training script for MobileMMTrackEE task using Stable Baselines3.

This script trains a mobile manipulator to track end-effector trajectories
using the PPO algorithm from Stable Baselines3.

Usage:
    # On Windows with Isaac Lab:
    I:\isaaclab\isaaclab-3090.bat -p scripts/reinforcement_learning/sb3/train.py \\
        --task MobileMMTrackEE-v0 \\
        --num_envs 1024 \\
        --headless
"""

import argparse
import os
from datetime import datetime

# Register our custom tasks
from src.task_spec import register_isaac_lab_tasks
register_isaac_lab_tasks()


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
        default=1_000_000,
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
        default=2048,
        help="Number of steps per rollout",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Minibatch size for PPO updates",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs per PPO update",
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
    
    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()
    
    # Import Isaac Lab and SB3 (must be after arg parsing for clean help message)
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import VecNormalize
        import torch
    except ImportError as e:
        print(f"Error: Required packages not available: {e}")
        print("Make sure you're running inside Isaac Lab environment")
        return
    
    # Setup logging directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = args.task.replace("-", "_").lower()
        args.log_dir = f"logs/sb3/{task_name}/{timestamp}"
    
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[train] Logging to: {args.log_dir}")
    
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
        except ImportError:
            print("[train] Warning: wandb not available, skipping W&B logging")
            args.wandb = False
    
    # Create environment
    print(f"[train] Creating environment: {args.task}")
    print(f"[train] Number of envs: {args.num_envs}")
    print(f"[train] Headless: {args.headless}")
    
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        render_mode=None if args.headless else "human",
    )
    
    # Wrap with VecNormalize for better training stability
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
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
    
    if args.wandb:
        callbacks.append(wandb_callback)
    
    # Create or load model
    if args.checkpoint:
        print(f"[train] Loading checkpoint: {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env=env,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        print("[train] Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=1,
        )
    
    # Print training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Task:              {args.task}")
    print(f"Num environments:  {args.num_envs}")
    print(f"Total timesteps:   {args.total_timesteps:,}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Rollout steps:     {args.n_steps}")
    print(f"Batch size:        {args.batch_size}")
    print(f"PPO epochs:        {args.n_epochs}")
    print(f"Save frequency:    {args.save_freq:,} steps")
    print(f"Log directory:     {args.log_dir}")
    print(f"Device:            {model.device}")
    print("=" * 60 + "\n")
    
    # Train
    try:
        print("[train] Starting training...")
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
        
        print(f"\n[train] Training complete!")
        print(f"[train] Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n[train] Training interrupted by user")
        
        # Save interrupt checkpoint
        interrupt_path = os.path.join(args.log_dir, "interrupt_checkpoint")
        model.save(interrupt_path)
        env.save(os.path.join(args.log_dir, "vec_normalize_interrupt.pkl"))
        
        print(f"[train] Interrupt checkpoint saved to: {interrupt_path}")
    
    finally:
        env.close()
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
