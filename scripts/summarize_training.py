"""
Summarize Session 7c training results from tensorboard logs
"""

import numpy as np
from pathlib import Path
import sys

def read_tensorboard_events(log_dir):
    """Read tensorboard event files to extract training metrics"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Get available tags
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            metrics[tag] = {'steps': steps, 'values': values}
        
        return metrics
    except ImportError:
        print("⚠️  tensorboard not installed. Install with: pip install tensorboard")
        return None
    except Exception as e:
        print(f"⚠️  Error reading tensorboard logs: {e}")
        return None

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    session7c_log = PROJECT_ROOT / "logs" / "sb3" / "mobilemmtrackee_v0" / "20251027_180246"
    
    print(f"\n{'='*80}")
    print(f"SESSION 7C TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"📁 Log directory: {session7c_log}\n")
    
    # Check if final model exists
    final_model = session7c_log / "final_model.zip"
    if final_model.exists():
        print(f"✅ Final model found: {final_model}")
        size_mb = final_model.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB\n")
    else:
        print(f"❌ Final model not found at: {final_model}\n")
    
    # Count checkpoints
    checkpoint_dir = session7c_log / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("ppo_mobile_mm_*_steps.zip"))
        print(f"📊 Checkpoints: {len(checkpoints)} saved")
        if checkpoints:
            first_ck = checkpoints[0].stem.split('_')[-2]
            last_ck = checkpoints[-1].stem.split('_')[-2]
            print(f"   First: {first_ck} steps")
            print(f"   Last:  {last_ck} steps")
        print()
    
    # Try to read tensorboard logs
    tb_dir = session7c_log / "PPO_1"
    if tb_dir.exists():
        print(f"📈 Reading tensorboard logs from: {tb_dir}")
        metrics = read_tensorboard_events(tb_dir)
        
        if metrics:
            # Print key metrics
            print(f"\n{'='*80}")
            print(f"KEY TRAINING METRICS")
            print(f"{'='*80}\n")
            
            # Look for common metric tags
            metric_keys = {
                'rollout/ep_rew_mean': 'Mean Episode Reward',
                'rollout/ep_len_mean': 'Mean Episode Length',
                'train/entropy_loss': 'Entropy Loss',
                'train/explained_variance': 'Explained Variance',
                'train/learning_rate': 'Learning Rate',
                'train/approx_kl': 'Approx KL Divergence',
                'train/clip_fraction': 'Clip Fraction',
                'train/value_loss': 'Value Loss',
            }
            
            for key, name in metric_keys.items():
                if key in metrics:
                    values = metrics[key]['values']
                    if len(values) > 0:
                        print(f"{name}:")
                        print(f"  Initial: {values[0]:.4f}")
                        print(f"  Final:   {values[-1]:.4f}")
                        print(f"  Mean:    {np.mean(values):.4f}")
                        print(f"  Std:     {np.std(values):.4f}")
                        print()
        else:
            print("   Could not read metrics")
    else:
        print(f"⚠️  Tensorboard logs not found at: {tb_dir}")
    
    # Training configuration
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}\n")
    print(f"Task:            MobileMMTrackEE-v0")
    print(f"Environments:    4096")
    print(f"Total timesteps: 100,000,000")
    print(f"Learning rate:   3e-4")
    print(f"Entropy coef:    0.001 → 0.0001 (decay after 50M)")
    print(f"KL schedule:     0.25 (warmup) → 0.15 (main) → 0.07 (finetune)")
    print(f"Base movement:   ENABLED (with Z-clamp fix)")
    print(f"Reachability:    ENABLED (guided mobilization)")
    print()
    
    print(f"{'='*80}\n")
    
    # Check if evaluation has been run
    eval_stats = session7c_log / "evaluation_stats.npz"
    if eval_stats.exists():
        print(f"✅ Evaluation statistics found: {eval_stats}")
        print(f"   Run: python scripts/compare_sessions.py")
        print(f"   to compare with Session 6\n")
    else:
        print(f"⏳ Evaluation not yet run")
        print(f"   Run: I:\\isaaclab\\isaaclab.bat -p scripts/evaluate_session7c.py --save_stats")
        print(f"   to generate evaluation statistics\n")

if __name__ == "__main__":
    main()
