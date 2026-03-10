"""
Analyze Session 8i training metrics to find instability.
Reads TensorBoard event files using tensorflow.
"""
import os
import sys

def analyze_training():
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ tensorflow not installed. Install with: pip install tensorflow")
        return
    
    log_dir = r"H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251107_164255\PPO_1"
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    
    if not event_files:
        print(f"❌ No event files found in {log_dir}")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    print(f"Reading: {event_file}\n")
    
    # Collect metrics
    records = []
    
    for event in tf.compat.v1.train.summary_iterator(event_file):
        step = event.step
        kl = var = None
        
        for value in event.summary.value:
            if value.tag == 'train/approx_kl':
                kl = value.simple_value
            elif value.tag == 'train/explained_variance':
                var = value.simple_value
        
        if kl is not None and var is not None:
            records.append((step, kl, var))
    
    if not records:
        print("❌ No metrics found!")
        return
    
    print(f"{'='*80}")
    print(f"TRAINING STABILITY ANALYSIS - Session 8i")
    print(f"{'='*80}\n")
    print(f"Total records: {len(records)}")
    print(f"Final step: {records[-1][0]/1_000_000:.2f}M\n")
    
    # Last 20 records
    print(f"{'='*80}")
    print(f"LAST 20 TRAINING ITERATIONS (Most Recent First)")
    print(f"{'='*80}\n")
    print(f"{'Steps(M)':<12} {'KL Div':<12} {'Variance':<12} {'Status'}")
    print("-" * 60)
    
    unstable_idx = None
    for i in range(min(20, len(records))):
        idx = len(records) - 1 - i
        step, kl, var = records[idx]
        steps_m = step / 1_000_000
        
        status = "✅ OK"
        if kl > 0.1:
            status = f"🔴 KL EXPLODED"
            if unstable_idx is None:
                unstable_idx = idx
        elif var < -0.3:
            status = f"🔴 VAR COLLAPSED"
            if unstable_idx is None:
                unstable_idx = idx
        elif var < 0.0:
            status = f"⚠️  VAR NEGATIVE"
        
        print(f"{steps_m:<12.2f} {kl:<12.4f} {var:<12.4f} {status}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    
    if unstable_idx is not None:
        unstable_step, unstable_kl, unstable_var = records[unstable_idx]
        print(f"🔴 INSTABILITY DETECTED!")
        print(f"   First unstable at: {unstable_step/1_000_000:.2f}M steps")
        print(f"   KL: {unstable_kl:.4f} (threshold: 0.1)")
        print(f"   Variance: {unstable_var:.4f} (threshold: -0.3)")
        
        # Find safe rollback point
        rollback_step = (unstable_step // 2_000_000 - 1) * 2_000_000
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Roll back to checkpoint: ~{rollback_step/1_000_000:.0f}M steps")
        print(f"   File: ppo_mobile_mm_{rollback_step}_steps.zip")
    else:
        final_step, final_kl, final_var = records[-1]
        print(f"✅ Last 20 iterations appear stable")
        print(f"   Final KL: {final_kl:.4f}")
        print(f"   Final Variance: {final_var:.4f}")
        print(f"   But emergency_pause was triggered, so check earlier...")
    
    # Show overall trend (every 10%)
    print(f"\n{'='*80}")
    print(f"OVERALL TRAINING TREND (Every 10%)")
    print(f"{'='*80}\n")
    print(f"{'Steps(M)':<12} {'KL Div':<12} {'Variance':<12}")
    print("-" * 40)
    
    for pct in range(0, 110, 10):
        idx = min(int(pct * len(records) / 100), len(records) - 1)
        step, kl, var = records[idx]
        print(f"{step/1_000_000:<12.2f} {kl:<12.4f} {var:<12.4f}")

if __name__ == "__main__":
    try:
        analyze_training()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
