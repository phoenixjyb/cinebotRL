"""Analyze Session 5b training results from TensorBoard logs."""
import os
import struct
from pathlib import Path

try:
    import tensorflow as tf
    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    print("WARNING: TensorFlow not available, will try alternative parsing")

log_dir = r"H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251021_230942\PPO_1"

print("\n=== SESSION 5B TRAINING RESULTS ===\n")

# Find event files
event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
if not event_files:
    print("ERROR: No TensorBoard event files found!")
    exit(1)

event_file = str(event_files[0])
print(f"Reading: {event_file}\n")

if HAVE_TF:
    # Use TensorFlow to read events
    metrics = {}
    for record in tf.data.TFRecordDataset(event_file):
        event = tf.compat.v1.Event.FromString(record.numpy())
        if event.summary:
            for value in event.summary.value:
                tag = value.tag
                if not hasattr(value, 'simple_value'):
                    continue
                val = value.simple_value
                step = event.step
                
                if tag not in metrics:
                    metrics[tag] = []
                metrics[tag].append((step, val))
    
    print(f"Found {len(metrics)} metric tags")
    
    # Sort by step and show final values
    print("\n=== FINAL METRICS (Last Recorded) ===")
    important_metrics = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean', 
        'train/learning_rate',
        'train/entropy_coef',
        'train/value_loss',
        'train/policy_loss',
    ]
    
    for metric in important_metrics:
        if metric in metrics:
            data = sorted(metrics[metric], key=lambda x: x[0])
            final_step, final_val = data[-1]
            print(f"{metric}: {final_val:.6f} (at step {final_step})")
            
            # Show last 5 values
            print(f"  Last 5 values:")
            for step, val in data[-5:]:
                print(f"    Step {step}: {val:.6f}")
    
    # Show all reward/penalty related metrics
    print("\n=== ALL REWARD/PENALTY METRICS ===")
    reward_metrics = {k: v for k, v in metrics.items() if 'reward' in k.lower() or 'penalty' in k.lower()}
    for metric, data in sorted(reward_metrics.items()):
        data_sorted = sorted(data, key=lambda x: x[0])
        final_step, final_val = data_sorted[-1]
        print(f"{metric}: {final_val:.6f}")
    
    # Environment status
    print("\n=== ENVIRONMENT/BROKEN METRICS ===")
    env_metrics = {k: v for k, v in metrics.items() if 'env' in k.lower() or 'broken' in k.lower()}
    for metric, data in sorted(env_metrics.items()):
        data_sorted = sorted(data, key=lambda x: x[0])
        final_step, final_val = data_sorted[-1]
        print(f"{metric}: {final_val:.6f}")
        
    max_step = max(max(steps for steps, _ in v) for v in metrics.values())
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Maximum training step: {max_step:,}")
    print(f"Target was: 100,000,000")
    print(f"Completion: {max_step/100_000_000*100:.1f}%")
    
else:
    print("ERROR: TensorFlow not available and no alternative parser implemented")
    print("Please install: pip install tensorflow")

# Check checkpoint files
checkpoint_dir = Path(log_dir).parent / "checkpoints"
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*_steps.pkl"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-2]))
        steps = int(latest.stem.split('_')[-2])
        print(f"\nLatest checkpoint: {latest.name}")
        print(f"Final training steps: {steps:,}")
