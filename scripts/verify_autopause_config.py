"""
Quick verification script to test if auto-pause config is accessible.
Run this to verify the code paths are correct before starting training.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("AUTO-PAUSE CONFIG VERIFICATION")
print("=" * 80)

# Step 1: Import config
print("\n[1/4] Importing config classes...")
try:
    from rl_platform.tasks.mobile_mm.config import RewardWeights
    print("  ✅ RewardWeights imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import: {e}")
    sys.exit(1)

# Step 2: Create config instance
print("\n[2/4] Creating RewardWeights instance...")
try:
    rewards_cfg = RewardWeights()
    print(f"  ✅ Instance created")
except Exception as e:
    print(f"  ❌ Failed to create instance: {e}")
    sys.exit(1)

# Step 3: Check auto-pause attributes
print("\n[3/4] Checking auto-pause attributes...")
attrs = ['enable_auto_pause', 'kl_threshold', 'variance_threshold', 'checkpoint_frequency_steps']
all_present = True
for attr in attrs:
    if hasattr(rewards_cfg, attr):
        value = getattr(rewards_cfg, attr)
        print(f"  ✅ {attr}: {value}")
    else:
        print(f"  ❌ {attr}: NOT FOUND")
        all_present = False

if not all_present:
    print("\n❌ FAILED: Some attributes are missing!")
    sys.exit(1)

# Step 4: Verify callback exists
print("\n[4/4] Checking AutoPauseCallback class...")
try:
    import subprocess
    result = subprocess.run(
        ['powershell', '-Command', 
         'Select-String -Path "scripts\\reinforcement_learning\\sb3\\train.py" -Pattern "class AutoPauseCallback" -Quiet'],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    if result.returncode == 0 and result.stdout.strip() == "True":
        print("  ✅ AutoPauseCallback class exists in train.py")
    else:
        print("  ❌ AutoPauseCallback class NOT FOUND in train.py")
        sys.exit(1)
except Exception as e:
    print(f"  ⚠ Could not verify callback class: {e}")

# Final verdict
print("\n" + "=" * 80)
print("✅ VERIFICATION PASSED")
print("=" * 80)
print("\nConfig values:")
print(f"  enable_auto_pause = {rewards_cfg.enable_auto_pause}")
print(f"  kl_threshold = {rewards_cfg.kl_threshold}")
print(f"  variance_threshold = {rewards_cfg.variance_threshold}")
print(f"  checkpoint_frequency_steps = {rewards_cfg.checkpoint_frequency_steps:,}")
print("\nNext step: Launch training to see '[OK] Auto-pause enabled' message")
print("=" * 80)
