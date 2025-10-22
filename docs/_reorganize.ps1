# Documentation Reorganization Script
# Auto-generated: October 22, 2025

$docsRoot = "c:\Users\yanbo\wSpace\cinebotRL\docs"
$moved = 0
$notFound = 0

# Helper function to move file if it exists
function Move-IfExists {
    param($Source, $Dest, $Category)
    
    $sourcePath = Join-Path $docsRoot $Source
    $destPath = Join-Path $docsRoot $Dest
    
    if (Test-Path $sourcePath) {
        $destDir = Split-Path $destPath -Parent
        if (!(Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Move-Item -Path $sourcePath -Destination $destPath -Force
        Write-Host "✓ [$Category] $Source → $Dest" -ForegroundColor Green
        $script:moved++
    } else {
        Write-Host "⊘ [$Category] $Source (not found)" -ForegroundColor Yellow
        $script:notFound++
    }
}

Write-Host "`n=== DOCUMENTATION REORGANIZATION ===" -ForegroundColor Cyan
Write-Host "Starting at: $docsRoot`n" -ForegroundColor Cyan

# 1. SETUP & CONFIGURATION → 01_setup/
Write-Host "`n[1/8] SETUP & CONFIGURATION" -ForegroundColor Magenta
Move-IfExists "QUICK_START.md" "01_setup/QUICK_START.md" "SETUP"
Move-IfExists "QUICK_START_Entropy_Decay_Training.md" "01_setup/QUICK_START_ENTROPY_TRAINING.md" "SETUP"
Move-IfExists "EVALUATION_SYSTEM_SETUP.md" "01_setup/EVALUATION_SETUP.md" "SETUP"

# 2. ARCHITECTURE & DESIGN → 02_architecture/
Write-Host "`n[2/8] ARCHITECTURE & DESIGN" -ForegroundColor Magenta
Move-IfExists "PPR_CONTROL_ARCHITECTURE.md" "02_architecture/PPR_CONTROL_ARCHITECTURE.md" "ARCH"
Move-IfExists "training_hierarchy_explained.md" "02_architecture/TRAINING_HIERARCHY.md" "ARCH"

# 3. TRAINING GUIDES → 03_training/
Write-Host "`n[3/8] TRAINING GUIDES" -ForegroundColor Magenta
Move-IfExists "READY_TO_TRAIN_Summary.md" "03_training/TRAINING_READINESS_CHECKLIST.md" "TRAIN"
Move-IfExists "FINAL_TRAINING_COMMAND_With_All_Protections.md" "03_training/TRAINING_COMMAND_REFERENCE.md" "TRAIN"
Move-IfExists "TRAINING_COMMAND_QUICK_REF.md" "03_training/COMMAND_QUICK_REF.md" "TRAIN"
Move-IfExists "TRAINING_WITH_RECORDED_TRAJECTORIES.md" "03_training/RECORDED_TRAJECTORIES_GUIDE.md" "TRAIN"
Move-IfExists "Why_Both_Entropy_Decay_AND_KL_Schedule.md" "03_training/ENTROPY_AND_KL_EXPLAINED.md" "TRAIN"

# 4. OPTIMIZATION & ANALYSIS → 04_optimization/
Write-Host "`n[4/8] OPTIMIZATION & ANALYSIS" -ForegroundColor Magenta
Move-IfExists "Policy_Divergence_Analysis_200M_Training.md" "04_optimization/POLICY_DIVERGENCE_200M.md" "OPT"
Move-IfExists "distance_penalty_implementation.md" "04_optimization/DISTANCE_PENALTY_IMPLEMENTATION.md" "OPT"
Move-IfExists "TRAJECTORY_TRACKING_IMPROVEMENTS.md" "04_optimization/TRAJECTORY_TRACKING_IMPROVEMENTS.md" "OPT"
Move-IfExists "IMPROVEMENT_CHECKLIST.md" "04_optimization/IMPROVEMENT_CHECKLIST.md" "OPT"

# 5. BUG FIXES & INVESTIGATIONS → 05_bug_fixes/
Write-Host "`n[5/8] BUG FIXES & INVESTIGATIONS" -ForegroundColor Magenta
Move-IfExists "BASE_MOVEMENT_BUG_ANALYSIS.md" "05_bug_fixes/BASE_MOVEMENT_BUG_ANALYSIS.md" "BUG"
Move-IfExists "BASE_MOVEMENT_COMPREHENSIVE_ANALYSIS.md" "05_bug_fixes/BASE_MOVEMENT_COMPREHENSIVE.md" "BUG"
Move-IfExists "BASE_FIX_ACTION_CHECKLIST.md" "05_bug_fixes/BASE_FIX_CHECKLIST.md" "BUG"
Move-IfExists "BUG_REPORT_Frozen_Base.md" "05_bug_fixes/BUG_FROZEN_BASE.md" "BUG"
Move-IfExists "Frozen_Base_Investigation_Summary.md" "05_bug_fixes/FROZEN_BASE_SUMMARY.md" "BUG"
Move-IfExists "CRITICAL_CONTACT_FORCE_ISSUE.md" "05_bug_fixes/CONTACT_FORCE_CRITICAL.md" "BUG"
Move-IfExists "CONTACT_FORCE_API_VERIFICATION.md" "05_bug_fixes/CONTACT_FORCE_API.md" "BUG"
Move-IfExists "CONTACT_FORCE_VERIFICATION_PRACTICAL.md" "05_bug_fixes/CONTACT_FORCE_PRACTICAL.md" "BUG"
Move-IfExists "SELF_COLLISION_IMPLEMENTATION_STATUS.md" "05_bug_fixes/SELF_COLLISION_STATUS.md" "BUG"
Move-IfExists "TRAJECTORY_START_INSIDE_BODY_ANALYSIS.md" "05_bug_fixes/TRAJECTORY_INSIDE_BODY.md" "BUG"
Move-IfExists "urdf_physics_analysis.md" "05_bug_fixes/urdf_physics_analysis.md" "BUG"
Move-IfExists "URDF_PHYSICS_ISSUES_REMAINING.md" "05_bug_fixes/URDF_PHYSICS_REMAINING.md" "BUG"

# 6. WORKFLOWS & GUIDES → 06_workflows/
Write-Host "`n[6/8] WORKFLOWS & GUIDES" -ForegroundColor Magenta
Move-IfExists "EVALUATION_GUIDE.md" "06_workflows/EVALUATION_GUIDE.md" "WORKFLOW"
Move-IfExists "EVALUATION_QUICKSTART.md" "06_workflows/EVALUATION_QUICKSTART.md" "WORKFLOW"
Move-IfExists "TESTING_RECORDED_TRAJECTORIES.md" "06_workflows/TESTING_TRAJECTORIES.md" "WORKFLOW"
Move-IfExists "VISUALIZATION_GUIDE.md" "06_workflows/VISUALIZATION_GUIDE.md" "WORKFLOW"

# 7. REFERENCE MATERIALS → 07_reference/
Write-Host "`n[7/8] REFERENCE MATERIALS" -ForegroundColor Magenta
Move-IfExists "ROBOT_HOME_POSITION.md" "07_reference/ROBOT_HOME_POSITION.md" "REF"
Move-IfExists "TRAJECTORY_INFO.md" "07_reference/TRAJECTORY_INFO.md" "REF"
Move-IfExists "CHASSIS_TRAJECTORY_QUICK_REF.md" "07_reference/CHASSIS_TRAJECTORY_REF.md" "REF"
Move-IfExists "TRAJECTORY_ANALYSIS_SUMMARY.md" "07_reference/TRAJECTORY_ANALYSIS.md" "REF"
Move-IfExists "TRAJECTORY_LOADING_INVESTIGATION.md" "07_reference/TRAJECTORY_LOADING_INVESTIGATION.md" "REF"
Move-IfExists "TRAJECTORY_LOADING_VERIFIED.md" "07_reference/TRAJECTORY_LOADING_VERIFIED.md" "REF"

# 8. PROJECT HISTORY & MILESTONES → 08_project_history/
Write-Host "`n[8/8] PROJECT HISTORY & MILESTONES" -ForegroundColor Magenta
Move-IfExists "ALL_TESTS_PASSED.md" "08_project_history/TESTS_PASSED_MILESTONE.md" "HISTORY"
Move-IfExists "DOCS_UPDATE_SUMMARY.md" "08_project_history/DOCS_UPDATE_OCT17.md" "HISTORY"
Move-IfExists "Critical_Analysis_Playbook_vs_Reality.md" "08_project_history/PLAYBOOK_VS_REALITY_ANALYSIS.md" "HISTORY"
Move-IfExists "INVESTIGATION_PLAN.md" "08_project_history/INVESTIGATION_PLAN_OCT18.md" "HISTORY"

Write-Host "`n=== REORGANIZATION COMPLETE ===" -ForegroundColor Cyan
Write-Host "Files moved: $moved" -ForegroundColor Green
Write-Host "Files not found: $notFound" -ForegroundColor Yellow
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Review moved files" -ForegroundColor White
Write-Host "2. Update cross-references" -ForegroundColor White
Write-Host "3. Update README.md and _NAVIGATION_GUIDE.txt" -ForegroundColor White
