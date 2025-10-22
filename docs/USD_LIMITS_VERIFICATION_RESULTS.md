# USD Limits Verification Results

**Date**: October 22, 2025  
**Script**: `scripts/verify_usd_limits.py`  
**Status**: ✅ **PASSED** - All critical joints have correct limits

---

## Summary

The USD file has correct joint limits. The "unexpected" format (degrees vs radians) is actually normal - USD stores in degrees, Isaac Lab converts to radians at runtime.

---

## Verification Results

### Critical Joints (PPR Base)

| Joint | Type | Limits (USD) | Limits (Radians) | Status |
|-------|------|--------------|------------------|---------|
| `joint_x` | Prismatic | ±50.000 m | ±50.000 m | ✅ CORRECT |
| `joint_y` | Prismatic | ±50.000 m | ±50.000 m | ✅ CORRECT |
| `joint_theta` | Revolute | ±360.000° | ±6.283 rad (±2π) | ✅ CORRECT |

### Arm Joints

| Joint | Type | Limits (USD) | Status |
|-------|------|--------------|---------|
| `left_arm_joint1` | Revolute | ±165.000° | ✅ Matches URDF |
| `left_arm_joint2` | Revolute | [0°, 185°] | ✅ Matches URDF |
| `left_arm_joint3` | Revolute | [-190°, 0°] | ✅ Matches URDF |
| `left_arm_joint4` | Revolute | ±165.000° | ✅ Matches URDF |
| `left_arm_joint5` | Revolute | ±95.002° | ✅ Matches URDF |
| `left_arm_joint6` | Revolute | ±165.000° | ✅ Matches URDF |

---

## Analysis

### Why Degrees in USD?

USD format stores revolute joint limits in **degrees** by convention, while Isaac Lab (and most robotics software) uses **radians**. The conversion happens automatically during USD loading.

**Verification**:
```python
import math
360° → radians = 6.283185 rad
2π rad = 6.283185 rad
✅ MATCH!
```

### Critical Finding: joint_theta is NOT locked!

**Expected Bug** (from code review):
- Infinite URDF limits → USD converter collapses to [0, 0] → locked joint

**Actual Reality**:
- URDF has finite limits: ±6.283185 rad (±2π)
- USD correctly stores: ±360° (equivalent to ±2π rad)
- Base CAN rotate freely within ±360°+ range

This was already fixed in the URDF physics corrections!

---

## Conclusion

✅ **ALL CHECKS PASSED**

1. ✅ `joint_theta` has proper rotation limits (±360° = ±2π rad)
2. ✅ `joint_x` and `joint_y` have large translation limits (±50m)
3. ✅ All arm joints match URDF specifications
4. ✅ No locked joints (no [0, 0] limits found)

**Impact**: Base can rotate and translate freely as designed. The USD → Isaac Lab conversion is working correctly.

---

## Next Step: Contact Forces Debug

With USD verification complete, proceed to contact forces investigation:

```powershell
I:\isaaclab\isaaclab.bat -p scripts/debug_contact_forces.py --num_envs 16 --steps 100
```

This is the last remaining critical issue before Session 6 launch.

---

## References

- **Verification Script**: `scripts/verify_usd_limits.py`
- **URDF Physics Fixes**: Previous commits fixed infinite limits → finite ±2π
- **USD File**: `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`
- **Code Review**: `docs/_CODE_REVIEW_VALIDATION.md` (Issue #3)

---

**Status**: ✅ Ready to proceed with contact forces debug!
