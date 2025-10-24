"""Analyze Session 6 training results from TensorBoard logs."""

import os
import sys
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    
    # Get all scalar tags
    tags = ea.Tags()['scalars']
    
    print(f"\n{'='*80}")
    print(f"Available metrics: {len(tags)}")
    print(f"{'='*80}")
    
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
        except Exception as e:
            print(f"Warning: Could not load {tag}: {e}")
    
    return data

def analyze_training_progress(data):
    """Analyze key training metrics."""
    
    print(f"\n{'='*80}")
    print("SESSION 6 TRAINING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Training Overview
    if 'rollout/ep_rew_mean' in data:
        rewards = data['rollout/ep_rew_mean']['values']
        steps = data['rollout/ep_rew_mean']['steps']
        
        print("📊 TRAINING OVERVIEW")
        print("-" * 80)
        print(f"Total timesteps: {steps[-1]:,}")
        print(f"Training updates: {len(steps):,}")
        print(f"Final episode reward: {rewards[-1]:.2f}")
        print(f"Initial reward: {rewards[0]:.2f}")
        print(f"Reward improvement: {rewards[-1] - rewards[0]:+.2f}")
        print()
    
    # Episode metrics
    print("📈 EPISODE METRICS (Final Values)")
    print("-" * 80)
    
    episode_metrics = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
    ]
    
    for metric in episode_metrics:
        if metric in data:
            final_val = data[metric]['values'][-1]
            initial_val = data[metric]['values'][0]
            print(f"{metric:40s}: {final_val:10.2f} (init: {initial_val:.2f}, Δ={final_val-initial_val:+.2f})")
    
    print()
    
    # Reward components (if available)
    print("💰 REWARD COMPONENTS (Final Values)")
    print("-" * 80)
    
    reward_components = [
        'reward_components/position_tracking',
        'reward_components/orientation_tracking',
        'reward_components/base_mobilization',
        'reward_components/target_distance_penalty',
        'reward_components/excessive_base_movement_penalty',
        'reward_components/action_magnitude_penalty',
        'reward_components/action_rate_penalty',
        'reward_components/jerk_penalty',
        'reward_components/self_collision_penalty',
        'reward_components/joint_limit_penalty',
    ]
    
    for component in reward_components:
        if component in data:
            vals = data[component]['values']
            final_val = vals[-1]
            mean_val = np.mean(vals[-100:])  # Last 100 updates
            print(f"{component:50s}: {final_val:10.4f} (avg: {mean_val:10.4f})")
    
    print()
    
    # Base diagnostics (CRITICAL for Session 6!)
    print("🚗 BASE MOVEMENT DIAGNOSTICS (Final Values)")
    print("-" * 80)
    
    base_metrics = [
        'base_diagnostics/base_vel_x_mean',
        'base_diagnostics/base_vel_x_max',
        'base_diagnostics/base_vel_z_mean',
        'base_diagnostics/base_vel_z_max',
        'base_diagnostics/base_action_x_mean',
        'base_diagnostics/base_action_x_std',
        'base_diagnostics/base_action_z_mean',
        'base_diagnostics/base_action_z_std',
    ]
    
    for metric in base_metrics:
        if metric in data:
            vals = data[metric]['values']
            final_val = vals[-1]
            mean_val = np.mean(vals[-100:])
            print(f"{metric:50s}: {final_val:10.4f} (avg: {mean_val:10.4f})")
    
    print()
    
    # Learning metrics
    print("🧠 LEARNING METRICS (Final Values)")
    print("-" * 80)
    
    learning_metrics = [
        'train/learning_rate',
        'train/entropy_loss',
        'train/policy_gradient_loss',
        'train/value_loss',
        'train/approx_kl',
        'train/clip_fraction',
        'train/explained_variance',
    ]
    
    for metric in learning_metrics:
        if metric in data:
            final_val = data[metric]['values'][-1]
            print(f"{metric:40s}: {final_val:12.6f}")
    
    print()
    
    # Critical Success Indicators
    print("✅ CRITICAL SUCCESS INDICATORS")
    print("-" * 80)
    
    success = True
    issues = []
    
    # Check 1: Base IS moving?
    if 'base_diagnostics/base_vel_x_max' in data:
        base_vel_max = np.mean(data['base_diagnostics/base_vel_x_max']['values'][-100:])
        if base_vel_max > 0.01:
            print(f"✅ Base IS moving! (max vel: {base_vel_max:.4f} m/s)")
        else:
            print(f"❌ Base FROZEN! (max vel: {base_vel_max:.6f} m/s)")
            success = False
            issues.append("Base not moving")
    
    # Check 2: Base mobilization reward positive?
    if 'reward_components/base_mobilization' in data:
        base_mob = np.mean(data['reward_components/base_mobilization']['values'][-100:])
        if base_mob > 0.0:
            print(f"✅ Base mobilization reward positive: {base_mob:.4f}")
        else:
            print(f"❌ Base mobilization reward negative: {base_mob:.4f}")
            success = False
            issues.append("Negative mobilization reward")
    
    # Check 3: Contact forces detected?
    if 'reward_components/self_collision_penalty' in data:
        collision_penalty = np.mean(np.abs(data['reward_components/self_collision_penalty']['values'][-100:]))
        if collision_penalty > 0.001:
            print(f"✅ Contact forces detected: {collision_penalty:.4f}")
        else:
            print(f"⚠️  No collisions detected (may be OK): {collision_penalty:.6f}")
    
    # Check 4: Reward improving?
    if 'rollout/ep_rew_mean' in data:
        rewards = data['rollout/ep_rew_mean']['values']
        initial_mean = np.mean(rewards[:10])
        final_mean = np.mean(rewards[-100:])
        improvement = final_mean - initial_mean
        
        if improvement > 0:
            print(f"✅ Reward improving: {improvement:+.2f}")
        else:
            print(f"❌ Reward declining: {improvement:+.2f}")
            success = False
            issues.append("Reward not improving")
    
    # Check 5: Jerk penalty reasonable?
    if 'reward_components/jerk_penalty' in data:
        jerk_penalty = np.mean(data['reward_components/jerk_penalty']['values'][-100:])
        if jerk_penalty > -200:  # Not catastrophically negative
            print(f"✅ Jerk penalty reasonable: {jerk_penalty:.2f}")
        else:
            print(f"❌ Jerk penalty too harsh: {jerk_penalty:.2f}")
            success = False
            issues.append("Jerk penalty too harsh")
    
    print()
    
    if success and not issues:
        print("🎉 SESSION 6 SUCCESS! All critical fixes working!")
    else:
        print(f"⚠️  Issues detected: {', '.join(issues)}")
    
    print()
    
    return data

def plot_key_metrics(data, output_dir):
    """Generate plots of key metrics."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Session 6 Training Analysis', fontsize=16)
        
        # Plot 1: Episode reward
        if 'rollout/ep_rew_mean' in data:
            ax = axes[0, 0]
            steps = data['rollout/ep_rew_mean']['steps']
            values = data['rollout/ep_rew_mean']['values']
            ax.plot(steps, values)
            ax.set_title('Episode Reward')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Mean Reward')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Base velocity
        if 'base_diagnostics/base_vel_x_max' in data:
            ax = axes[0, 1]
            steps = data['base_diagnostics/base_vel_x_max']['steps']
            values = data['base_diagnostics/base_vel_x_max']['values']
            ax.plot(steps, values)
            ax.set_title('Base Max Velocity (CRITICAL)')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Max Velocity (m/s)')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Base mobilization reward
        if 'reward_components/base_mobilization' in data:
            ax = axes[1, 0]
            steps = data['reward_components/base_mobilization']['steps']
            values = data['reward_components/base_mobilization']['values']
            ax.plot(steps, values)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Base Mobilization Reward')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Jerk penalty
        if 'reward_components/jerk_penalty' in data:
            ax = axes[1, 1]
            steps = data['reward_components/jerk_penalty']['steps']
            values = data['reward_components/jerk_penalty']['values']
            ax.plot(steps, values)
            ax.set_title('Jerk Penalty (Should be reasonable)')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Penalty')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Position tracking
        if 'reward_components/position_tracking' in data:
            ax = axes[2, 0]
            steps = data['reward_components/position_tracking']['steps']
            values = data['reward_components/position_tracking']['values']
            ax.plot(steps, values)
            ax.set_title('Position Tracking Reward')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: KL divergence
        if 'train/approx_kl' in data:
            ax = axes[2, 1]
            steps = data['train/approx_kl']['steps']
            values = data['train/approx_kl']['values']
            ax.plot(steps, values)
            ax.axhline(y=0.15, color='r', linestyle='--', alpha=0.5, label='Target KL')
            ax.set_title('KL Divergence')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Approx KL')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_dir / 'session6_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plots saved to: {output_path}")
        
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plots")

def main():
    log_dir = Path("c:/Users/yanbo/wSpace/cinebotRL/logs/sb3/mobilemmtrackee_v0/20251022_230622/PPO_1")
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Analyzing Session 6 Training Results")
    print(f"{'='*80}")
    print(f"Log directory: {log_dir}")
    
    # Load data
    data = load_tensorboard_data(str(log_dir))
    
    # Analyze
    analyze_training_progress(data)
    
    # Plot (if matplotlib available)
    output_dir = log_dir.parent
    plot_key_metrics(data, output_dir)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
