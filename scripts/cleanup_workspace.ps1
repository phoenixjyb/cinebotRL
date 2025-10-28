# Workspace Cleanup Script
# Organizes loose files and creates proper structure

$ErrorActionPreference = "Stop"

Write-Host "`n=== CinebotRL Workspace Cleanup ===" -ForegroundColor Cyan
Write-Host "Organizing loose files into proper structure...`n" -ForegroundColor Yellow

# === Step 1: Create organized structure ===
Write-Host "[1/5] Creating directory structure..." -ForegroundColor Green

$dirs = @(
    "docs/training_sessions/session_6",
    "docs/training_sessions/session_7c", 
    "docs/training_sessions/session_7d",
    "docs/09_archive/old_analysis",
    "docs/09_archive/temp_docs",
    "scripts/analysis",
    "data/trajectory_filters"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✅ Created: $dir" -ForegroundColor Gray
    }
}

# === Step 2: Move root analysis scripts ===
Write-Host "`n[2/5] Moving analysis scripts..." -ForegroundColor Green

$scripts = @{
    "analyze_session5b.py" = "scripts/analysis/"
    "reporting.py" = "scripts/analysis/"
}

foreach ($file in $scripts.Keys) {
    if (Test-Path $file) {
        $dest = $scripts[$file]
        Move-Item $file $dest -Force
        Write-Host "  ✅ $file → $dest" -ForegroundColor Gray
    }
}

# === Step 3: Move inspection/debug text files ===
Write-Host "`n[3/5] Moving debug/inspection files..." -ForegroundColor Green

$inspectFiles = @{
    "base_corrected_inspect.txt" = "docs/tracking/"
    "theta_before_x_inspect.txt" = "docs/tracking/"
}

foreach ($file in $inspectFiles.Keys) {
    if (Test-Path $file) {
        $dest = $inspectFiles[$file]
        Move-Item $file $dest -Force
        Write-Host "  ✅ $file → $dest" -ForegroundColor Gray
    }
}

# === Step 4: Move trajectory filter files ===
Write-Host "`n[4/5] Moving trajectory filter files..." -ForegroundColor Green

$trajFiles = @{
    "chassis_required_indices.txt" = "data/trajectory_filters/"
    "chassis_required_trajectories.txt" = "data/trajectory_filters/"
}

foreach ($file in $trajFiles.Keys) {
    if (Test-Path $file) {
        $dest = $trajFiles[$file]
        # Create data/trajectory_filters if doesn't exist
        if (-not (Test-Path "data/trajectory_filters")) {
            New-Item -ItemType Directory -Path "data/trajectory_filters" -Force | Out-Null
        }
        Move-Item $file $dest -Force
        Write-Host "  ✅ $file → $dest" -ForegroundColor Gray
    }
}

# === Step 5: Organize session documentation ===
Write-Host "`n[5/5] Organizing session documentation..." -ForegroundColor Green

# Session 6 docs
$session6Docs = @{
    "docs/SESSION_6_EVALUATION_SUMMARY.md" = "docs/training_sessions/session_6/"
}

# Session 7c docs
$session7cDocs = @{
    "docs/SESSION_7C_VISUALIZATION_GUIDE.md" = "docs/training_sessions/session_7c/"
    "docs/SESSION_7C_VS_SESSION_6_COMPARISON.md" = "docs/training_sessions/session_7c/"
}

# Session 7d docs
$session7dDocs = @{
    "docs/SESSION_7C_VS_7D_CHANGES.md" = "docs/training_sessions/session_7d/"
    "docs/SESSION_7D_ACCELERATED.md" = "docs/training_sessions/session_7d/"
    "docs/SESSION_7D_QUICK_START.md" = "docs/training_sessions/session_7d/"
    "docs/SESSION_7D_REWARD_TUNING_PROPOSAL.md" = "docs/training_sessions/session_7d/"
}

# Archive old analysis docs (root level)
$archiveDocs = @{
    "COORDINATE_TRANSFORM_ANSWER.md" = "docs/09_archive/old_analysis/"
    "REACHABILITY_BUILD_RUNNING.md" = "docs/09_archive/old_analysis/"
    "REACHABILITY_MAP_SUMMARY.md" = "docs/09_archive/old_analysis/"
    "REACHABILITY_QUICK_REF.md" = "docs/09_archive/old_analysis/"
}

# Archive temp/internal docs
$tempDocs = @{
    "docs/_CODE_REVIEW_VALIDATION.md" = "docs/09_archive/temp_docs/"
    "docs/_COMPLETION_SUMMARY.md" = "docs/09_archive/temp_docs/"
    "docs/_FIX_IMPLEMENTATION_PLAN.md" = "docs/09_archive/temp_docs/"
    "docs/_NAVIGATION_GUIDE.txt" = "docs/09_archive/temp_docs/"
    "docs/_REORGANIZATION_PLAN.md" = "docs/09_archive/temp_docs/"
    "docs/_REORGANIZATION_SUMMARY.md" = "docs/09_archive/temp_docs/"
}

$allDocs = $session6Docs + $session7cDocs + $session7dDocs + $archiveDocs + $tempDocs

foreach ($file in $allDocs.Keys) {
    if (Test-Path $file) {
        $dest = $allDocs[$file]
        Move-Item $file $dest -Force
        $fileName = Split-Path $file -Leaf
        Write-Host "  ✅ $fileName → $dest" -ForegroundColor Gray
    }
}

# Delete the reorganize script itself
if (Test-Path "docs/_reorganize.ps1") {
    Remove-Item "docs/_reorganize.ps1" -Force
    Write-Host "  ✅ Removed docs/_reorganize.ps1" -ForegroundColor Gray
}

# === Summary ===
Write-Host "`n=== Cleanup Complete! ===" -ForegroundColor Green
Write-Host "`n📁 New Structure:" -ForegroundColor Cyan
Write-Host "  • scripts/analysis/          - Analysis scripts" -ForegroundColor Gray
Write-Host "  • data/trajectory_filters/   - Trajectory filter configs" -ForegroundColor Gray
Write-Host "  • docs/training_sessions/    - Organized by session" -ForegroundColor Gray
Write-Host "  • docs/tracking/             - Debug/inspection files" -ForegroundColor Gray
Write-Host "  • docs/09_archive/           - Old/temp documentation" -ForegroundColor Gray

Write-Host "`n✅ Root directory cleaned!" -ForegroundColor Green
Write-Host "   Only essential files remain (README, pyproject.toml, etc.)`n" -ForegroundColor Yellow
