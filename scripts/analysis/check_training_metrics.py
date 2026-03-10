"""
Quick script to analyze training metrics from TensorBoard logs
to identify when instability occurred.
Uses tensorflow to read event files directly.
"""
import os
import sys
import struct

def read_tfevents_simple(filepath):
    """Simple parser for TensorBoard event files."""
    try:
        import tensorflow as tf
        
        metrics = {'kl': [], 'variance': [], 'steps': []}
        
        for event in tf.compat.v1.train.summary_iterator(filepath):
            step = event.step
            for value in event.summary.value:
                if value.tag == 'train/approx_kl':
                    metrics['kl'].append((step, value.simple_value))
                elif value.tag == 'train/explained_variance':
                    metrics['variance'].append((step, value.simple_value))
        
        return metrics
    except Exception as e:
        print(f"Error reading event file: {e}")
        return None

def analyze_tensorboard_logs(log_dir):
    """Extract key metrics from TensorBoard event files."""
    
    # Find event file
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"❌ No event files found in {log_dir}")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    print(f"Reading: {event_file}\n")
    
    metrics = read_tfevents_simple(event_file)
    if not metrics or not metrics['kl'] or not metrics['variance']:
        print("❌ Failed to read metrics from event file")
        return
    
    kl_events = metrics['kl']
    variance_events = metrics['variance']
    
    print(f"{'='*80}")
    print("TRAINING STABILITY ANALYSIS")
    print(f"{'='*80}\n")
    
    if not kl_events or not variance_events:
        print("❌ No training metrics found!")
        return
    
    print(f"Total KL records: {len(kl_events)}")
    print(f"Total Variance records: {len(variance_events)}")
    
    # Align events by step
    kl_dict = {step: val for step, val in kl_events}
    var_dict = {step: val for step, val in variance_events}
    common_steps = sorted(set(kl_dict.keys()) & set(var_dict.keys()))
    
    if not common_steps:
        print("❌ No common steps found!")
        return
    
    print(f"Common training steps: {len(common_steps)}")
    print(f"Final step: {common_steps[-1]/1_000_000:.2f}M\n")
    
    # Check last 20 iterations for instability
    print(f"\n{'='*80}")
    print("LAST 20 ITERATIONS (Most Recent First)")
    print(f"{'='*80}\n")
    print(f"{'Iteration':<10} {'Steps(M)':<12} {'KL Div':<12} {'Variance':<12} {'Status'}")
    print("-" * 70)
    
    unstable_count = 0
    first_unstable_step = None
    
    for i in range(min(20, len(kl_events))):
        idx = len(kl_events) - 1 - i
        kl_event = kl_events[idx]
        var_event = variance_events[idx]
        
        kl_val = kl_event.value
        var_val = var_event.value
        steps_m = kl_event.step / 1_000_000
        
        # Check stability
        status = "✅ OK"
        if kl_val > 0.1:
            status = f"🔴 KL>{kl_val:.3f}"
            unstable_count += 1
            if first_unstable_step is None:
                first_unstable_step = kl_event.step
        elif var_val < -0.3:
            status = f"🔴 VAR<{var_val:.3f}"
            unstable_count += 1
            if first_unstable_step is None:
                first_unstable_step = kl_event.step
        elif var_val < 0.0:
            status = f"⚠️  VAR={var_val:.3f}"
        
        print(f"{idx:<10} {steps_m:<12.2f} {kl_val:<12.4f} {var_val:<12.4f} {status}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if unstable_count > 0:
        print(f"🔴 INSTABILITY DETECTED!")
        print(f"   Unstable iterations: {unstable_count}/20")
        print(f"   First unstable at: {first_unstable_step/1_000_000:.2f}M steps")
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Roll back to checkpoint BEFORE {first_unstable_step/1_000_000:.2f}M")
        
        # Find nearest checkpoint
        checkpoint_step = (first_unstable_step // 2_000_000) * 2_000_000 - 2_000_000
        print(f"   Suggested checkpoint: ~{checkpoint_step/1_000_000:.0f}M steps")
    else:
        print(f"✅ Last 20 iterations look stable")
        print(f"   Final KL: {kl_events[-1].value:.4f}")
        print(f"   Final Variance: {variance_events[-1].value:.4f}")
        print(f"   Final Steps: {kl_events[-1].step/1_000_000:.2f}M")
    
    # Show trend over all training
    print(f"\n{'='*80}")
    print("OVERALL TRAINING TREND")
    print(f"{'='*80}\n")
    
    # Sample every 10% of training
    sample_indices = [int(i * len(kl_events) / 10) for i in range(11)]
    
    print(f"{'Steps(M)':<12} {'KL Div':<12} {'Variance':<12}")
    print("-" * 40)
    for idx in sample_indices:
        if idx < len(kl_events):
            steps_m = kl_events[idx].step / 1_000_000
            kl_val = kl_events[idx].value
            var_val = variance_events[idx].value
            print(f"{steps_m:<12.2f} {kl_val:<12.4f} {var_val:<12.4f}")

if __name__ == "__main__":
    log_dir = r"H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251107_164255\PPO_1"
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        sys.exit(1)
    
    try:
        analyze_tensorboard_logs(log_dir)
    except Exception as e:
        print(f"❌ Error analyzing logs: {e}")
        import traceback
        traceback.print_exc()
