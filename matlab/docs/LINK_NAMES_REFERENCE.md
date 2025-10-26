# Quick Reference: Correct Link Names

## URDF Structure (Verified)

```
Mobile Base Chain:
  base
    └── base_link_x (PPR virtual)
          └── base_link_y (PPR virtual)
                └── abstract_chassis_link (main body)

Arm Chain (from abstract_chassis_link):
  left_arm_base_link (shoulder mount)
    └── left_arm_link1  ← NO underscore!
          └── left_arm_link2
                └── left_arm_link3
                      └── left_arm_link4
                            └── left_arm_link5
                                  └── left_arm_link6
                                        └── left_gripper_link (EE)
```

## Collision Pairs (17 total)

### Arm → Chassis (3)
```matlab
{'left_arm_link1', 'abstract_chassis_link'}
{'left_arm_link2', 'abstract_chassis_link'}
{'left_arm_link3', 'abstract_chassis_link'}
```

### Arm Self-Collisions (10)
```matlab
% Link 1 vs later links
{'left_arm_link1', 'left_arm_link3'}  % Skip link2 (adjacent)
{'left_arm_link1', 'left_arm_link4'}
{'left_arm_link1', 'left_arm_link5'}
{'left_arm_link1', 'left_arm_link6'}

% Link 2 vs later links
{'left_arm_link2', 'left_arm_link4'}  % Skip link3 (adjacent)
{'left_arm_link2', 'left_arm_link5'}
{'left_arm_link2', 'left_arm_link6'}

% Link 3 vs later links
{'left_arm_link3', 'left_arm_link5'}  % Skip link4 (adjacent)
{'left_arm_link3', 'left_arm_link6'}

% Link 4 vs link 6
{'left_arm_link4', 'left_arm_link6'}  % Skip link5 (adjacent)
```

### Gripper Collisions (4)
```matlab
{'left_gripper_link', 'abstract_chassis_link'}
{'left_gripper_link', 'left_arm_link1'}
{'left_gripper_link', 'left_arm_link2'}
{'left_gripper_link', 'left_arm_link3'}
```

## Common Mistakes ❌

```matlab
% WRONG ❌
'left_arm_link_1'           % Has underscore
'base_link'                 % Not the chassis
'lidar_link'                % Doesn't exist
'left_arm_eef_link'         % Wrong EE name
'left_arm_gripper_base_link' % Doesn't exist

% CORRECT ✅
'left_arm_link1'            % No underscore
'abstract_chassis_link'     % Actual chassis
(removed)                   % Not in URDF
'left_gripper_link'         % Real EE name
(removed)                   % Not in URDF
```

## Frame Names

```matlab
BASE_LINK = "left_arm_base_link";  % Arm shoulder (origin for FK)
EE_LINK = "left_gripper_link";     % End effector
```

---

**Memorize:** `left_arm_link1` (no underscore), `abstract_chassis_link`, `left_gripper_link`
