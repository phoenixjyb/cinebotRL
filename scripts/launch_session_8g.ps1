# Session 8g Launcher - Expanded Workspace + Curriculum Learning
# ==============================================================================
# Session 8g builds on 8f structural fixes with:
# 1. Expanded workspace comfort zone (0.6m → 0.7m hard margin)
# 2. Gentler penalties (60 → 30 distance weight)
# 3. Workspace comfort observations (+2 dims: comfort + dist_to_optimal)
# 4. Two-stage curriculum (0-50M: workspace, 50M-100M: precision)
#
# FK Workspace Reality (matlab/exports/reach_surface.mat):
#   1,677 points, median 0.594m, P75=0.78m
#   Current 0.6m margin excludes 50% of workspace!
#   New 0.7m margin covers 65% (P5-P95: 0.15-0.92m)
#
# Session 8f Baseline (BEST SO FAR):
#   Position: 3.08m, Orientation: 46.5°, Reach penalty: 232, Reward: -126k
#   But: Reachability still low (0.64), workspace drifted (0.42→0.60m)
#
# Session 8g Strategy:
#   Stage 1 (0-50M): Learn workspace positioning with easier trajectories
#     - Use chassis_required_trajectories.txt filtering
#     - Reduced position/orientation weights (50% of final)
#     - Focus: Get base to 0.5-0.7m zone
#   Stage 2 (50M-100M): Full precision tracking with all trajectories
#     - Restore full weights
#     - Build on stable workspace foundation
# ==============================================================================

param(
    [ValidateSet("smoke", "stage1", "stage2", "full")]
    [string]$Phase = "smoke",
    [switch]$Test,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

# Configuration
$IsaacLabPath = "I:\isaaclab"
$IsaacLabBat = Join-Path $IsaacLabPath "isaaclab.bat"
$ProjectRoot = "C:\Users\yanbo\wSpace\cinebotRL"
$TrainScript = Join-Path $ProjectRoot "scripts\reinforcement_learning\sb3\train.py"
$TestScript = Join-Path $ProjectRoot "scripts\test_mobile_mm_env.py"
$TaskName = "MobileMMTrackEE-v0"

# Session 8g configuration
$SessionConfig = @{
    smoke = @{
        NumEnvs = 64
        TotalSteps = 500000
        TrajectoryFilter = "chassis"
        Description = "Smoke test with curriculum Stage 1 settings"
    }
    stage1 = @{
        NumEnvs = 16384
        TotalSteps = 50000000
        TrajectoryFilter = "chassis"
        Description = "Stage 1: Learn workspace positioning (50M steps, ~7hr)"
    }
    stage2 = @{
        NumEnvs = 16384
        TotalSteps = 50000000
        TrajectoryFilter = "all"
        Description = "Stage 2: Full precision tracking (50M steps, ~7hr)"
    }
    full = @{
        NumEnvs = 16384
        TotalSteps = 100000000
        TrajectoryFilter = "all"
        Description = "Full run 0-100M with auto-curriculum (~14hr)"
    }
}

$Config = $SessionConfig[$Phase]

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Session 8g: Expanded Workspace + Curriculum" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase: $Phase" -ForegroundColor Yellow
Write-Host "Description: $($Config.Description)" -ForegroundColor Gray
Write-Host "Environments: $($Config.NumEnvs)" -ForegroundColor Gray
Write-Host "Total Steps: $($Config.TotalSteps)" -ForegroundColor Gray
Write-Host "Trajectory Filter: $($Config.TrajectoryFilter)" -ForegroundColor Gray
Write-Host ""

# Key changes from Session 8f
Write-Host "Key Changes from Session 8f:" -ForegroundColor Green
Write-Host "  1. Hard margin: 0.6m → 0.7m (now covers 65% of FK workspace)" -ForegroundColor White
Write-Host "  2. Distance penalty weight: 60 → 30 (gentler gradient)" -ForegroundColor White
Write-Host "  3. Optimal distance: 0.5m → 0.6m (FK workspace median)" -ForegroundColor White
Write-Host "  4. Workspace comfort observations (+2 dims)" -ForegroundColor White
Write-Host "  5. Two-stage curriculum (workspace → precision)" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT: Observation dimension changed 76 → 78 dims" -ForegroundColor Yellow
Write-Host "  Network architecture incompatible with Session 8f" -ForegroundColor Yellow
Write-Host "  Cannot resume from 8f checkpoints - MUST start fresh" -ForegroundColor Yellow
Write-Host ""
Write-Host "FK Workspace Data:" -ForegroundColor Magenta
Write-Host "  Median radius: 0.594m, P75: 0.78m" -ForegroundColor Gray
Write-Host "  Coverage <0.7m: 64.6%, <0.9m: 92.0%" -ForegroundColor Gray
Write-Host ""

# Curriculum info
if ($Phase -eq "stage1") {
    Write-Host "Curriculum Stage 1 (0-50M):" -ForegroundColor Yellow
    Write-Host "  Trajectories: chassis_required only (easier paths)" -ForegroundColor Gray
    Write-Host "  Position weight: 5.0 (50% of final)" -ForegroundColor Gray
    Write-Host "  Orientation weight: 15.0 (50% of final)" -ForegroundColor Gray
    Write-Host "  Goal: Converge workspace to 0.5-0.7m" -ForegroundColor Gray
    Write-Host ""
} elseif ($Phase -eq "stage2") {
    Write-Host "Curriculum Stage 2 (50M-100M):" -ForegroundColor Yellow
    Write-Host "  Trajectories: ALL (full cinematic paths)" -ForegroundColor Gray
    Write-Host "  Position weight: 10.0 (full)" -ForegroundColor Gray
    Write-Host "  Orientation weight: 30.0 (full)" -ForegroundColor Gray
    Write-Host "  Goal: Precision tracking on stable workspace" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Note: Must have completed Stage 1 first!" -ForegroundColor Red
    Write-Host ""
} elseif ($Phase -eq "full") {
    Write-Host "Full Training (0-100M):" -ForegroundColor Yellow
    Write-Host "  0-50M: Stage 1 (reduced weights: pos=5.0, ori=15.0)" -ForegroundColor Gray
    Write-Host "  50M-100M: Stage 2 (full weights: pos=10.0, ori=30.0)" -ForegroundColor Gray
    Write-Host "  Trajectories: ALL (complete dataset)" -ForegroundColor Gray
    Write-Host "  Automatic transition at 50M checkpoint" -ForegroundColor Gray
    Write-Host ""
}

if ($Test) {
    Write-Host "Running environment sanity test..." -ForegroundColor Yellow
    & $IsaacLabBat -p $TestScript --num_envs 4 --headless
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[SUCCESS] Environment test passed!" -ForegroundColor Green
        Write-Host "  Observation space: 78 dims (76 + 2 workspace comfort)" -ForegroundColor Gray
    } else {
        Write-Host "`n[ERROR] Environment test failed!" -ForegroundColor Red
        exit 1
    }
    exit 0
}

# Disable Gymnasium plugin entrypoints (Windows ale_py fix)
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

# Build training command
$TrainArgs = @(
    "-p"
    $TrainScript
    "--task"
    $TaskName
    "--num_envs"
    $Config.NumEnvs
    "--total_timesteps"
    $Config.TotalSteps
    "--headless"
)

# Add trajectory filtering based on phase
if ($Config.TrajectoryFilter -eq "chassis") {
    # Stage 1 only: Use chassis_required_trajectories.txt
    $TrainArgs += @("--use_chassis_only")
}
# For "all" or other modes: No flag = use all trajectories
# Curriculum weight switching happens automatically at 50M (handled in env.py)

if ($Resume) {
    Write-Host "Resume mode enabled" -ForegroundColor Yellow
    # Find latest checkpoint
    $LogDirs = Get-ChildItem -Path (Join-Path $ProjectRoot "logs\sb3\mobilemmtrackee_v0") -Directory | 
               Sort-Object Name -Descending
    
    if ($LogDirs.Count -gt 0) {
        $LatestLog = $LogDirs[0].FullName
        $Checkpoints = Get-ChildItem -Path (Join-Path $LatestLog "checkpoints") -Filter "*.zip" | 
                      Sort-Object Name -Descending
        
        if ($Checkpoints.Count -gt 0) {
            $LatestCheckpoint = $Checkpoints[0].FullName
            Write-Host "Resuming from: $LatestCheckpoint" -ForegroundColor Gray
            $TrainArgs += @("--resume", $LatestCheckpoint)
        } else {
            Write-Host "No checkpoints found, starting fresh" -ForegroundColor Yellow
        }
    }
}

Write-Host "Launching training..." -ForegroundColor Green
Write-Host "Command: $IsaacLabBat $($TrainArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Launch training
& $IsaacLabBat @TrainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    if ($Phase -eq "stage1") {
        Write-Host "`nNext Step: Evaluate Stage 1 checkpoint" -ForegroundColor Yellow
        Write-Host "  1. Run evaluation on final_model.zip" -ForegroundColor Gray
        Write-Host "  2. Check workspace: Should be 0.5-0.7m" -ForegroundColor Gray
        Write-Host "  3. Check reachability: Should be >1.5" -ForegroundColor Gray
        Write-Host "  4. If good: Run Stage 2 with .\scripts\launch_session_8g.ps1 -Phase stage2" -ForegroundColor Gray
    } elseif ($Phase -eq "stage2") {
        Write-Host "`nNext Step: Final evaluation" -ForegroundColor Yellow
        Write-Host "  1. Run evaluation on final_model.zip" -ForegroundColor Gray
        Write-Host "  2. Compare with Session 8f baseline" -ForegroundColor Gray
        Write-Host "  3. Check for improvements in all metrics" -ForegroundColor Gray
    }
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "Training failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}
