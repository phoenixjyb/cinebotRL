#!/usr/bin/env python3
"""
Quick arm reach calculator from URDF geometry.
Calculates theoretical maximum reach and compares with code assumptions.

Usage:
    python scripts/calculate_arm_reach.py
"""

import numpy as np
from pathlib import Path

# From URDF: assets_own/mobile_manipulator_PPR_base_corrected.urdf
SHOULDER_MOUNT = np.array([0.160, 0.0, 0.947])  # Relative to base center

# Link offsets from URDF joint origins
LINK_CHAIN = [
    np.array([-0.001, 0.045, 0.0]),   # joint1 (shoulder)
    np.array([0.0, 0.106, 0.0]),      # joint2
    np.array([-0.349, 0.020, 0.0]),   # joint3 (elbow, major link!)
    np.array([0.048, 0.071, 0.0]),    # joint4 (wrist)
    np.array([0.243, 0.002, 0.0]),    # joint5 (forearm, major link!)
    np.array([0.054, 0.004, 0.0]),    # joint6
    np.array([0.080, 0.0, 0.0]),      # EE offset (estimated)
]

def calculate_max_reach():
    """Calculate maximum theoretical reach by summing link magnitudes."""
    print("="*70)
    print("ARM WORKSPACE CALCULATION")
    print("="*70)
    print()
    
    # Shoulder position
    print(f"📍 Shoulder mount (from base center):")
    print(f"   X: {SHOULDER_MOUNT[0]:.3f}m (forward)")
    print(f"   Y: {SHOULDER_MOUNT[1]:.3f}m (lateral)")
    print(f"   Z: {SHOULDER_MOUNT[2]:.3f}m (height)")
    print()
    
    # Link chain
    print("🔗 Link chain:")
    cumulative = np.zeros(3)
    for i, link in enumerate(LINK_CHAIN):
        cumulative += link
        length = np.linalg.norm(link)
        print(f"   Link {i+1}: {link} | length={length:.3f}m | cumulative={cumulative}")
    print()
    
    # Maximum reach calculations
    print("📐 Maximum reach calculations:")
    print()
    
    # Method 1: Sum of X magnitudes (straight line)
    x_contributions = [abs(link[0]) for link in LINK_CHAIN]
    max_x = sum(x_contributions)
    print(f"1️⃣  X-axis sum: {max_x:.3f}m")
    print(f"   (If arm extended straight forward)")
    print()
    
    # Method 2: Sum of all link lengths (spherical)
    link_lengths = [np.linalg.norm(link) for link in LINK_CHAIN]
    max_spherical = sum(link_lengths)
    print(f"2️⃣  Spherical (sum of lengths): {max_spherical:.3f}m")
    print(f"   (If arm could bend freely in 3D)")
    print()
    
    # Method 3: Major axes combined (realistic)
    major_x = sum([abs(link[0]) for link in LINK_CHAIN])
    major_y = sum([abs(link[1]) for link in LINK_CHAIN])
    max_radial = np.sqrt(major_x**2 + major_y**2)
    print(f"3️⃣  Radial (X² + Y²)½: {max_radial:.3f}m")
    print(f"   Major X: {major_x:.3f}m")
    print(f"   Major Y: {major_y:.3f}m")
    print(f"   (Realistic maximum in XY plane)")
    print()
    
    # From base center
    reach_from_base_xy = max_radial + SHOULDER_MOUNT[0]  # Add forward offset
    reach_from_base_3d = np.sqrt(
        (max_radial + SHOULDER_MOUNT[0])**2 + SHOULDER_MOUNT[2]**2
    )
    
    print("📏 Reach from BASE CENTER:")
    print(f"   XY plane: {reach_from_base_xy:.3f}m")
    print(f"   3D space: {reach_from_base_3d:.3f}m")
    print()
    
    # Compare with code
    print("="*70)
    print("COMPARISON WITH CODE ASSUMPTIONS")
    print("="*70)
    print()
    
    code_assumption = 0.6  # From rewards.py
    print(f"💻 Code assumes: {code_assumption:.2f}m (from base center, XY)")
    print(f"📐 URDF calculates: {max_radial:.2f}m (from shoulder)")
    print(f"                   {reach_from_base_xy:.2f}m (from base center, XY)")
    print()
    
    if code_assumption < reach_from_base_xy:
        diff = reach_from_base_xy - code_assumption
        print(f"✅ Code is CONSERVATIVE by {diff:.2f}m")
        print(f"   → Good! Encourages base movement before max extension")
    else:
        diff = code_assumption - reach_from_base_xy
        print(f"❌ Code is OPTIMISTIC by {diff:.2f}m")
        print(f"   → Bad! Assumes unreachable targets are reachable")
    print()
    
    # Observed data
    print("="*70)
    print("OBSERVED IN TRAINING LOGS")
    print("="*70)
    print()
    print("Session 6/7 observations:")
    print("  Best EE distance: 0.55m (from base center)")
    print("  Typical range: 0.57-0.80m")
    print("  Worst case: 1.07m")
    print()
    print("Example:")
    print("  EE:       [2.051, 0.044, 0.942]")
    print("  Base:     [1.050, 0.080, 0.000]")
    print("  Shoulder: [1.210, 0.080, 0.947]")
    print()
    
    # Calculate from example
    ee = np.array([2.051, 0.044, 0.942])
    base = np.array([1.050, 0.080, 0.000])
    shoulder = base + SHOULDER_MOUNT
    
    dist_from_shoulder = np.linalg.norm(ee - shoulder)
    dist_from_base_xy = np.linalg.norm((ee - base)[:2])
    dist_from_base_3d = np.linalg.norm(ee - base)
    
    print(f"  EE from shoulder: {dist_from_shoulder:.3f}m")
    print(f"  EE from base (XY): {dist_from_base_xy:.3f}m")
    print(f"  EE from base (3D): {dist_from_base_3d:.3f}m")
    print()
    
    if dist_from_shoulder > max_radial:
        excess = dist_from_shoulder - max_radial
        print(f"⚠️  WARNING: Observed distance EXCEEDS calculated max by {excess:.3f}m!")
        print(f"   Possible reasons:")
        print(f"   - Gripper/tool length not in URDF (adds ~0.1-0.15m)")
        print(f"   - Physics engine allows slight over-extension")
        print(f"   - Calculation missing some joint coupling effects")
    else:
        margin = max_radial - dist_from_shoulder
        print(f"✅ Observed within calculated max (margin: {margin:.3f}m)")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print()
    print(f"1. Optimal working radius: 0.3-0.6m from base (code: 0.6m ✅)")
    print(f"2. Maximum safe reach: ~{max_radial:.2f}m from shoulder")
    print(f"3. Maximum from base (XY): ~{reach_from_base_xy:.2f}m")
    print(f"4. Observed maximum: ~0.84m from shoulder")
    print()
    print("💡 Current code assumption (0.6m) is CORRECT for practical use!")
    print("   Targets beyond 0.6m should trigger base movement.")
    print()

if __name__ == "__main__":
    calculate_max_reach()
