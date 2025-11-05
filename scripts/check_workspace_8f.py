"""Check workspace distance from Session 8f TensorBoard logs."""
from tensorboard.backend.event_processing import event_accumulator
import sys

log_file = r'C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251101_013539\PPO_1\events.out.tfevents.1761932209.JiaFamily.34432.0'

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

ws_mean = ea.Scalars('monitoring/workspace_distance_mean')
ws_max = ea.Scalars('monitoring/workspace_distance_max')
soft_exceed = ea.Scalars('monitoring/workspace_soft_exceed_pct')
hard_exceed = ea.Scalars('monitoring/workspace_hard_exceed_pct')

print(f"\n{'='*60}")
print(f"Session 8f - Workspace Distance During Training")
print(f"{'='*60}")
print(f"\nTotal checkpoints logged: {len(ws_mean)}")
print(f"\nLatest 10 checkpoints:")
print(f"{'Step':>12} {'Mean':>8} {'Max':>8} {'Soft%':>8} {'Hard%':>8}")
print(f"{'-'*60}")

for i in range(max(-10, -len(ws_mean)), 0):
    step = ws_mean[i].step
    mean_val = ws_mean[i].value
    max_val = ws_max[i].value if i < len(ws_max) else 0
    soft_pct = soft_exceed[i].value if i < len(soft_exceed) else 0
    hard_pct = hard_exceed[i].value if i < len(hard_exceed) else 0
    
    # Color code based on target
    status = "✅" if 0.45 <= mean_val <= 0.55 else "⚠️" if 0.40 <= mean_val <= 0.60 else "🚨"
    
    print(f"{step:12,} {mean_val:8.4f} {max_val:8.4f} {soft_pct:7.1f}% {hard_pct:7.1f}% {status}")

print(f"\n{'='*60}")
print(f"Target: Mean ~0.5m, Soft margin at 0.7m, Hard margin at 0.9m")
print(f"Session 8e comparison (FAILURE): 0.55→0.34→0.58m (drifted)")
print(f"{'='*60}\n")

# Calculate statistics
if len(ws_mean) > 0:
    distances = [entry.value for entry in ws_mean]
    avg_distance = sum(distances) / len(distances)
    min_distance = min(distances)
    max_distance = max(distances)
    
    # Check for drift
    if len(distances) >= 10:
        early_avg = sum(distances[:5]) / 5
        late_avg = sum(distances[-5:]) / 5
        drift = late_avg - early_avg
        
        print(f"Analysis:")
        print(f"  Average workspace distance: {avg_distance:.4f}m")
        print(f"  Range: {min_distance:.4f}m - {max_distance:.4f}m")
        print(f"  Early training (first 5): {early_avg:.4f}m")
        print(f"  Late training (last 5): {late_avg:.4f}m")
        print(f"  Drift: {drift:+.4f}m", end="")
        
        if abs(drift) < 0.05:
            print(" ✅ STABLE!")
        elif abs(drift) < 0.10:
            print(" ⚠️ Slight drift")
        else:
            print(" 🚨 SIGNIFICANT DRIFT!")
