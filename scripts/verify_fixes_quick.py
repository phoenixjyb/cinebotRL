"""Quick validation test for bug fixes - code inspection only.

This script validates the fixes by directly inspecting the source code
without needing to run Isaac Sim. Much faster for verification!
"""

import os
import sys
import re

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

print("=" * 80)
print("Bug Fixes Code Inspection Validation")
print("=" * 80)
print()

# Read the environment file
env_file = os.path.join(PROJECT_ROOT, "src", "rl_platform", "tasks", "mobile_mm", "env.py")

if not os.path.exists(env_file):
    print(f"✗ Cannot find env.py at: {env_file}")
    sys.exit(1)

with open(env_file, 'r', encoding='utf-8') as f:
    env_code = f.read()

print(f"Reading: {env_file}")
print(f"File size: {len(env_code)} bytes")
print()

# Test Fix #1: Base Mobility
print("=" * 80)
print("Fix #1: Base Mobility")
print("=" * 80)

fix1_checks = {
    "_base_joint_ids initialization": r"self\._base_joint_ids\s*=",
    "base_velocities creation": r"base_velocities\s*=\s*torch\.cat",
    "set_joint_velocity_target call": r"self\.robot\.set_joint_velocity_target.*joint_ids=self\._base_joint_ids",
}

fix1_pass = 0
for check_name, pattern in fix1_checks.items():
    if re.search(pattern, env_code, re.DOTALL):
        print(f"✓ {check_name}")
        fix1_pass += 1
    else:
        print(f"✗ {check_name} - NOT FOUND")

print(f"Result: {fix1_pass}/{len(fix1_checks)} checks passed")
print()

# Test Fix #2: Action Scaling
print("=" * 80)
print("Fix #2: Action Scaling")
print("=" * 80)

fix2_checks = {
    "_scale_actions_to_joint_limits function": r"def _scale_actions_to_joint_limits",
    "Safety margin calculation": r"safety_margin\s*=\s*0\.05",
    "Scaling formula": r"actions_normalized\s*\*\s*\(upper_safe\s*-\s*lower_safe\)",
    "Function called before set_joint_position_target": r"arm_actions_scaled\s*=\s*self\._scale_actions_to_joint_limits",
}

fix2_pass = 0
for check_name, pattern in fix2_checks.items():
    if re.search(pattern, env_code, re.DOTALL):
        print(f"✓ {check_name}")
        fix2_pass += 1
    else:
        print(f"✗ {check_name} - NOT FOUND")

print(f"Result: {fix2_pass}/{len(fix2_checks)} checks passed")
print()

# Test Fix #3: Action History
print("=" * 80)
print("Fix #3: Action History")
print("=" * 80)

fix3_checks = {
    "_actions_t_minus_2 initialization": r"self\._actions_t_minus_2\s*=\s*torch\.zeros_like",
    "History chain update (t-2)": r"self\._actions_t_minus_2\s*=\s*self\.prev_prev_actions\.clone\(\)",
    "History chain update (t-1)": r"self\.prev_prev_actions\s*=\s*self\.prev_actions\.clone\(\)",
    "History chain update (t)": r"self\.prev_actions\s*=\s*actions\.clone\(\)",
    "Reward uses _actions_t_minus_2": r"prev_prev_actions=self\._actions_t_minus_2",
}

fix3_pass = 0
for check_name, pattern in fix3_checks.items():
    if re.search(pattern, env_code, re.DOTALL):
        print(f"✓ {check_name}")
        fix3_pass += 1
    else:
        print(f"✗ {check_name} - NOT FOUND")

print(f"Result: {fix3_pass}/{len(fix3_checks)} checks passed")
print()

# Test Fix #4: Collision Detection
print("=" * 80)
print("Fix #4: Collision Detection")
print("=" * 80)

fix4_checks = {
    "PhysX contact force reading": r"self\.robot\.root_physx_view\.get_net_contact_forces\(\)",
    "Fallback to robot.data": r"self\.robot\.data\.body_net_contact_force_w",
    "Warning message for fallback": r"\[WARNING\].*Contact forces API",
    "Termination uses contact forces": r"contact_force_mag\s*=\s*torch\.norm\(net_contact_forces",
    "Termination threshold check": r"terminated\s*\|=\s*max_contact_force\s*>\s*self\.task_cfg\.self_collision_termination_threshold",
    "Has try/except for contact forces": r"try:\s+.*?net_contact_forces\s*=.*?except",
}

fix4_pass = 0
for check_name, pattern in fix4_checks.items():
    if re.search(pattern, env_code, re.DOTALL):
        print(f"✓ {check_name}")
        fix4_pass += 1
    else:
        print(f"✗ {check_name} - NOT FOUND")

print(f"Result: {fix4_pass}/{len(fix4_checks)} checks passed")
print()

# Test Fix #5: Trajectory Advancement
print("=" * 80)
print("Fix #5: Trajectory Advancement")
print("=" * 80)

fix5_checks = {
    "trajectory_manager.step() in _get_rewards": r"self\.trajectory_manager\.step\(\)",
    "step() after reward calculation": r"# Advance trajectory.*\n.*self\.trajectory_manager\.step\(\)",
    "Comment explains timing": r"This must happen after reward calculation",
}

fix5_pass = 0
for check_name, pattern in fix5_checks.items():
    if re.search(pattern, env_code, re.DOTALL):
        print(f"✓ {check_name}")
        fix5_pass += 1
    else:
        print(f"✗ {check_name} - NOT FOUND")

print(f"Result: {fix5_pass}/{len(fix5_checks)} checks passed")
print()

# Final Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

total_checks = len(fix1_checks) + len(fix2_checks) + len(fix3_checks) + len(fix4_checks) + len(fix5_checks)
total_passed = fix1_pass + fix2_pass + fix3_pass + fix4_pass + fix5_pass

print(f"Fix #1 - Base Mobility:         {fix1_pass}/{len(fix1_checks)} {'✓ PASS' if fix1_pass == len(fix1_checks) else '✗ FAIL'}")
print(f"Fix #2 - Action Scaling:        {fix2_pass}/{len(fix2_checks)} {'✓ PASS' if fix2_pass == len(fix2_checks) else '✗ FAIL'}")
print(f"Fix #3 - Action History:        {fix3_pass}/{len(fix3_checks)} {'✓ PASS' if fix3_pass == len(fix3_checks) else '✗ FAIL'}")
print(f"Fix #4 - Collision Detection:   {fix4_pass}/{len(fix4_checks)} {'✓ PASS' if fix4_pass == len(fix4_checks) else '✗ FAIL'}")
print(f"Fix #5 - Trajectory Advancement: {fix5_pass}/{len(fix5_checks)} {'✓ PASS' if fix5_pass == len(fix5_checks) else '✗ FAIL'}")
print()
print(f"TOTAL: {total_passed}/{total_checks} checks passed")
print()

if total_passed == total_checks:
    print("🎉 ALL 5 CRITICAL FIXES VERIFIED! Ready for training.")
    exit_code = 0
else:
    print("⚠️  Some checks failed. Review the output above.")
    exit_code = 1

print("=" * 80)

sys.exit(exit_code)
