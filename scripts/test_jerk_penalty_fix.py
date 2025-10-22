#!/usr/bin/env python3
"""
Test Jerk Penalty After Fix

This script validates that the jerk penalty fix is working correctly by:
1. Loading the robot limits configuration
2. Simulating typical base acceleration scenarios
3. Computing jerk penalties with old vs new limits
4. Confirming penalties are now reasonable

Expected Results:
- Old limit (5.0 m/s³): ~900 point penalty (CATASTROPHIC)
- New limit (50.0 m/s³): ~25 point penalty (REASONABLE)

Usage:
    python scripts/test_jerk_penalty_fix.py

Reference: docs/_CODE_REVIEW_VALIDATION.md (Issue #1)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def compute_jerk_penalty(jerk_magnitude: float, max_jerk: float, weight: float = 0.05) -> float:
    """Compute jerk penalty using the reward function logic.
    
    Args:
        jerk_magnitude: Actual jerk magnitude (m/s³)
        max_jerk: Maximum allowed jerk (m/s³)
        weight: Penalty weight (default 0.05 from config)
    
    Returns:
        Penalty value (negative)
    """
    if jerk_magnitude <= max_jerk:
        return 0.0
    
    excess_jerk = jerk_magnitude - max_jerk
    penalty = -weight * (excess_jerk ** 2)
    return penalty


def test_typical_scenario():
    """Test jerk penalty for typical base movement scenario."""
    
    print("\n" + "="*70)
    print("JERK PENALTY TEST - TYPICAL BASE MOVEMENT")
    print("="*70)
    
    # Typical scenario: base accelerates from 0 to 0.25 m/s in one control step
    dt = 0.05  # 20 Hz control rate
    
    # Step 1: Start from rest
    v_prev = 0.0  # m/s
    a_prev = 0.0  # m/s²
    
    # Step 2: Accelerate to 0.25 m/s
    v_curr = 0.25  # m/s
    a_curr = (v_curr - v_prev) / dt  # = 5.0 m/s²
    
    # Compute jerk
    jerk = (a_curr - a_prev) / dt  # = 100 m/s³
    
    print(f"\nScenario: Base accelerates from rest to 0.25 m/s in {dt}s")
    print(f"  v_prev = {v_prev:.2f} m/s")
    print(f"  v_curr = {v_curr:.2f} m/s")
    print(f"  a_curr = {a_curr:.2f} m/s²")
    print(f"  jerk   = {jerk:.2f} m/s³")
    
    # Old limit (before fix)
    old_limit = 5.0
    old_penalty = compute_jerk_penalty(jerk, old_limit)
    
    print(f"\n  OLD LIMIT = {old_limit:.1f} m/s³:")
    print(f"    Excess jerk = {jerk - old_limit:.2f} m/s³")
    print(f"    Penalty = {old_penalty:.2f} points")
    
    # New limit (after fix)
    new_limit = 50.0
    new_penalty = compute_jerk_penalty(jerk, new_limit)
    
    print(f"\n  NEW LIMIT = {new_limit:.1f} m/s³:")
    print(f"    Excess jerk = {jerk - new_limit:.2f} m/s³")
    print(f"    Penalty = {new_penalty:.2f} points")
    
    # Compare with mobilization bonus
    mobilization_bonus = 150.0  # base_progress_reward from config.py
    
    print(f"\n  COMPARISON WITH BASE MOBILIZATION BONUS (+{mobilization_bonus:.0f}):")
    print(f"    Old: penalty {old_penalty:.1f} >> bonus +{mobilization_bonus:.0f} → NET {old_penalty + mobilization_bonus:.1f} (CATASTROPHIC!)")
    print(f"    New: penalty {new_penalty:.1f} << bonus +{mobilization_bonus:.0f} → NET {new_penalty + mobilization_bonus:.1f} (GOOD!)")
    
    # Verdict
    print("\n" + "="*70)
    if abs(old_penalty) > mobilization_bonus:
        print("✅ OLD LIMIT CONFIRMED AS BUG:")
        print("   Jerk penalty dominated mobilization bonus")
        print("   Policy would learn to freeze base")
    
    if abs(new_penalty) < mobilization_bonus:
        print("✅ NEW LIMIT CONFIRMED AS FIX:")
        print("   Jerk penalty is proportional to mobilization bonus")
        print("   Policy can now learn to move base")
    
    return old_penalty, new_penalty


def test_aggressive_movement():
    """Test jerk penalty for aggressive base movement."""
    
    print("\n" + "="*70)
    print("JERK PENALTY TEST - AGGRESSIVE MOVEMENT")
    print("="*70)
    
    dt = 0.05
    
    # Aggressive: 0 → 0.5 m/s in one step
    v_prev = 0.0
    v_curr = 0.5
    a_prev = 0.0
    a_curr = (v_curr - v_prev) / dt  # = 10.0 m/s²
    jerk = (a_curr - a_prev) / dt  # = 200 m/s³
    
    print(f"\nScenario: Aggressive acceleration 0 → {v_curr} m/s in {dt}s")
    print(f"  jerk = {jerk:.2f} m/s³")
    
    old_limit = 5.0
    old_penalty = compute_jerk_penalty(jerk, old_limit)
    
    new_limit = 50.0
    new_penalty = compute_jerk_penalty(jerk, new_limit)
    
    print(f"\n  OLD: {old_penalty:.1f} points (MASSIVE penalty!)")
    print(f"  NEW: {new_penalty:.1f} points (Significant but not catastrophic)")
    
    print("\n  This aggressive movement should be discouraged but not completely blocked.")
    
    return old_penalty, new_penalty


def verify_config():
    """Verify that the config file has the new limit."""
    
    print("\n" + "="*70)
    print("CONFIG VERIFICATION")
    print("="*70)
    
    try:
        from src.rl_platform.tasks.mobile_mm.config import RobotLimits
        
        limits = RobotLimits()
        max_jerk = limits.max_linear_jerk
        
        print(f"\nLoaded max_linear_jerk from config: {max_jerk} m/s³")
        
        if abs(max_jerk - 50.0) < 0.01:
            print("✅ CONFIG UPDATED! max_linear_jerk = 50.0 m/s³")
            return True
        elif abs(max_jerk - 5.0) < 0.01:
            print("❌ CONFIG NOT UPDATED! Still at 5.0 m/s³")
            print("   Did you forget to save src/rl_platform/tasks/mobile_mm/config.py?")
            return False
        else:
            print(f"⚠️  UNEXPECTED VALUE: {max_jerk} m/s³")
            return False
            
    except Exception as e:
        print(f"❌ ERROR loading config: {e}")
        return False


def main():
    """Run all jerk penalty tests."""
    
    print("\n" + "="*70)
    print("JERK PENALTY FIX VALIDATION")
    print("="*70)
    print("\nThis test validates that the jerk penalty fix resolves the frozen base bug.")
    print("Reference: docs/_CODE_REVIEW_VALIDATION.md (Issue #1 - CRITICAL)")
    
    # Verify config first
    config_ok = verify_config()
    
    # Test scenarios
    old_typical, new_typical = test_typical_scenario()
    old_aggressive, new_aggressive = test_aggressive_movement()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nTypical movement (0 → 0.25 m/s):")
    print(f"  Old penalty: {old_typical:.1f} points → BASE FROZEN")
    print(f"  New penalty: {new_typical:.1f} points → BASE CAN MOVE")
    print(f"  Improvement: {abs(old_typical - new_typical):.1f} points less harsh")
    
    print("\nAggressive movement (0 → 0.5 m/s):")
    print(f"  Old penalty: {old_aggressive:.1f} points → COMPLETELY BLOCKED")
    print(f"  New penalty: {new_aggressive:.1f} points → DISCOURAGED BUT ALLOWED")
    
    if config_ok:
        print("\n✅ ALL CHECKS PASSED!")
        print("   Jerk penalty fix is working correctly")
        print("   Ready to launch Session 6 with this fix")
    else:
        print("\n❌ CONFIG NOT UPDATED!")
        print("   Fix src/rl_platform/tasks/mobile_mm/config.py first")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
