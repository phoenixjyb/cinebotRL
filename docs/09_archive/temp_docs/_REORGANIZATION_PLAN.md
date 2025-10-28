# Documentation Reorganization Plan
# Generated: October 22, 2025

## File Classification and Target Locations

### 1. SETUP & CONFIGURATION → 01_setup/
- QUICK_START.md → 01_setup/QUICK_START.md (Getting started guide)
- QUICK_START_Entropy_Decay_Training.md → 01_setup/QUICK_START_ENTROPY_TRAINING.md (Specific training setup)
- EVALUATION_SYSTEM_SETUP.md → 01_setup/EVALUATION_SETUP.md (Eval system configuration)

### 2. ARCHITECTURE & DESIGN → 02_architecture/
- PPR_CONTROL_ARCHITECTURE.md → 02_architecture/PPR_CONTROL_ARCHITECTURE.md (PPR joint control)
- training_hierarchy_explained.md → 02_architecture/TRAINING_HIERARCHY.md (Training structure)

### 3. TRAINING GUIDES → 03_training/
- READY_TO_TRAIN_Summary.md → 03_training/TRAINING_READINESS_CHECKLIST.md
- FINAL_TRAINING_COMMAND_With_All_Protections.md → 03_training/TRAINING_COMMAND_REFERENCE.md
- TRAINING_COMMAND_QUICK_REF.md → 03_training/COMMAND_QUICK_REF.md
- TRAINING_WITH_RECORDED_TRAJECTORIES.md → 03_training/RECORDED_TRAJECTORIES_GUIDE.md
- Why_Both_Entropy_Decay_AND_KL_Schedule.md → 03_training/ENTROPY_AND_KL_EXPLAINED.md

### 4. OPTIMIZATION & ANALYSIS → 04_optimization/
- Policy_Divergence_Analysis_200M_Training.md → 04_optimization/POLICY_DIVERGENCE_200M.md
- distance_penalty_implementation.md → 04_optimization/DISTANCE_PENALTY_IMPLEMENTATION.md
- TRAJECTORY_TRACKING_IMPROVEMENTS.md → 04_optimization/TRAJECTORY_TRACKING_IMPROVEMENTS.md
- IMPROVEMENT_CHECKLIST.md → 04_optimization/IMPROVEMENT_CHECKLIST.md

### 5. BUG FIXES & INVESTIGATIONS → 05_bug_fixes/
- BASE_MOVEMENT_BUG_ANALYSIS.md → 05_bug_fixes/BASE_MOVEMENT_BUG_ANALYSIS.md
- BASE_MOVEMENT_COMPREHENSIVE_ANALYSIS.md → 05_bug_fixes/BASE_MOVEMENT_COMPREHENSIVE.md
- BASE_FIX_ACTION_CHECKLIST.md → 05_bug_fixes/BASE_FIX_CHECKLIST.md
- BUG_REPORT_Frozen_Base.md → 05_bug_fixes/BUG_FROZEN_BASE.md
- Frozen_Base_Investigation_Summary.md → 05_bug_fixes/FROZEN_BASE_SUMMARY.md
- CRITICAL_CONTACT_FORCE_ISSUE.md → 05_bug_fixes/CONTACT_FORCE_CRITICAL.md
- CONTACT_FORCE_API_VERIFICATION.md → 05_bug_fixes/CONTACT_FORCE_API.md
- CONTACT_FORCE_VERIFICATION_PRACTICAL.md → 05_bug_fixes/CONTACT_FORCE_PRACTICAL.md
- SELF_COLLISION_IMPLEMENTATION_STATUS.md → 05_bug_fixes/SELF_COLLISION_STATUS.md
- TRAJECTORY_START_INSIDE_BODY_ANALYSIS.md → 05_bug_fixes/TRAJECTORY_INSIDE_BODY.md
- urdf_physics_analysis.md → 05_bug_fixes/urdf_physics_analysis.md
- URDF_PHYSICS_ISSUES_REMAINING.md → 05_bug_fixes/URDF_PHYSICS_REMAINING.md

### 6. WORKFLOWS & GUIDES → 06_workflows/
- EVALUATION_GUIDE.md → 06_workflows/EVALUATION_GUIDE.md
- EVALUATION_QUICKSTART.md → 06_workflows/EVALUATION_QUICKSTART.md
- TESTING_RECORDED_TRAJECTORIES.md → 06_workflows/TESTING_TRAJECTORIES.md
- VISUALIZATION_GUIDE.md → 06_workflows/VISUALIZATION_GUIDE.md

### 7. REFERENCE MATERIALS → 07_reference/
- ROBOT_HOME_POSITION.md → 07_reference/ROBOT_HOME_POSITION.md
- TRAJECTORY_INFO.md → 07_reference/TRAJECTORY_INFO.md
- CHASSIS_TRAJECTORY_QUICK_REF.md → 07_reference/CHASSIS_TRAJECTORY_REF.md
- TRAJECTORY_ANALYSIS_SUMMARY.md → 07_reference/TRAJECTORY_ANALYSIS.md
- TRAJECTORY_LOADING_INVESTIGATION.md → 07_reference/TRAJECTORY_LOADING_INVESTIGATION.md
- TRAJECTORY_LOADING_VERIFIED.md → 07_reference/TRAJECTORY_LOADING_VERIFIED.md

### 8. PROJECT HISTORY & MILESTONES → 08_project_history/
- ALL_TESTS_PASSED.md → 08_project_history/TESTS_PASSED_MILESTONE.md
- DOCS_UPDATE_SUMMARY.md → 08_project_history/DOCS_UPDATE_OCT17.md
- Critical_Analysis_Playbook_vs_Reality.md → 08_project_history/PLAYBOOK_VS_REALITY_ANALYSIS.md
- INVESTIGATION_PLAN.md → 08_project_history/INVESTIGATION_PLAN_OCT18.md

## Summary Statistics
- Total files to organize: 40
- Target directories: 8
- Files staying in root: 2 (README.md, _NAVIGATION_GUIDE.txt)

## Next Steps
1. Create any missing subdirectories
2. Move files to new locations
3. Update cross-references in documents
4. Update README.md with new paths
5. Update _NAVIGATION_GUIDE.txt
