# Session 8h Launcher - Balanced Curriculum + Gradual Transition
# ==============================================================================
# Session 8h fixes Session 8g curriculum failures while preserving workspace improvements:
#
# Session 8g Results (Post-Mortem):
#   @ 31M: ✅ Workspace converged PERFECTLY (0.554m, 3% violations)
#   @ 40M: ✅ Position improved vs 8f (301cm vs 308cm)
#   @ 40M: ❌ Orientation catastrophic (130° vs 8f's 46.5°)
#   @ 100M: 💥 COLLAPSED (std exploded 19,000x, entropy=-82.8)
#
# Root Cause Analysis:
#   1. Curriculum weights (5.0, 15.0) under-trained orientation despite 1:3 ratio
#   2. Instant transition @ 50M shocked value function (variance=-0.241 @ 36M)
#   3. Training diverged catastrophically by 100M
#
# Session 8h Fixes:
#   1. Balanced curriculum: (4.0, 12.0) → (10.0, 30.0) maintains 1:3 ratio at 40%
#   2. Gradual transition: 45M-55M linear ramp (not instant @ 50M)
#   3. Trajectory curriculum: 4 stages (easy → recovery → moderate → full)
#   4. Auto-pause monitoring: KL>0.1 or variance<0 triggers halt
#   5. 20M gate: Validate approach before committing to 100M
#
# Expected Outcomes:
#   Conservative: 280-300cm position, 60-80° orientation
#   Optimistic: 250-280cm position, 45-60° orientation (match 8f!)
# ==============================================================================

param(
    [ValidateSet("smoke", "stage1", "full")]
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

# Session 8h configuration
$SessionConfig = @{
    smoke = @{
        NumEnvs = 64
        TotalSteps = 500000
        TrajectoryFilter = "chassis"  # Proxy for stage0_easy until trajectories populated
        TrajectoryStage = $null  # "stage0_easy" once trajectories are ready
        LearningRate = "2e-4"  # Lower LR for stability (Session 8g used 3e-4)
        Description = "Smoke test with curriculum Stage 1 (4.0, 12.0)"
    }
    stage1 = @{
        NumEnvs = 16384
        TotalSteps = 20000000
        TrajectoryFilter = "chassis"  # Proxy for stage0_easy until trajectories populated
        TrajectoryStage = $null  # "stage0_easy" once trajectories are ready
        LearningRate = "2e-4"  # Lower LR for stability (Session 8g used 3e-4)
        Description = "20M validation: Check orientation improvement"
    }
    full = @{
        NumEnvs = 16384
        TotalSteps = 100000000
        TrajectoryFilter = "chassis"  # Can switch to "all" or use stage curriculum
        TrajectoryStage = $null  # Can enable "stage0_easy" → "stage3_full" progression
        LearningRate = "2e-4"  # Lower LR for stability (Session 8g used 3e-4)
        Description = "Full 100M with gradual transition 45-55M (~14hr)"
    }
}

$Config = $SessionConfig[$Phase]

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Session 8h: Balanced Curriculum Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase: $Phase" -ForegroundColor Yellow
Write-Host "Description: $($Config.Description)" -ForegroundColor Gray
Write-Host "Environments: $($Config.NumEnvs)" -ForegroundColor Gray
Write-Host "Total Steps: $($Config.TotalSteps)" -ForegroundColor Gray
Write-Host "Trajectory Filter: $($Config.TrajectoryFilter)" -ForegroundColor Gray
Write-Host ""

# Key fixes from Session 8g
Write-Host "Key Fixes from Session 8g:" -ForegroundColor Green
Write-Host "  1. Curriculum weights: (5.0, 15.0) → (4.0, 12.0) - 40% scaling" -ForegroundColor White
Write-Host "     Fix: Orientation gets adequate signal from start (1:3 ratio maintained)" -ForegroundColor Cyan
Write-Host "  2. Gradual transition: 45M-55M linear ramp (not instant @ 50M)" -ForegroundColor White
Write-Host "     Fix: Prevents value function shock that killed 8g" -ForegroundColor Cyan
Write-Host "  3. Auto-pause monitoring: KL>0.1 or variance<0" -ForegroundColor White
Write-Host "     Fix: Catches divergence early, prevents catastrophic collapse" -ForegroundColor Cyan
Write-Host "  4. 20M validation gate: Check metrics before committing to 100M" -ForegroundColor White
Write-Host "     Fix: Validates approach without wasting 14 hours" -ForegroundColor Cyan
Write-Host ""

# What we're keeping from 8g (proven to work)
Write-Host "Proven Elements Retained from 8g:" -ForegroundColor Magenta
Write-Host "  ✅ Workspace margin: 0.7m (65% FK coverage, converged 0.554m @ 31M)" -ForegroundColor Gray
Write-Host "  ✅ Distance penalty: weight 30 (gentler gradient)" -ForegroundColor Gray
Write-Host "  ✅ Optimal distance: 0.6m (FK median)" -ForegroundColor Gray
Write-Host "  ✅ Workspace observations: +2 dims (comfort + normalized distance)" -ForegroundColor Gray
Write-Host ""

# Curriculum details
if ($Phase -eq "smoke") {
    Write-Host "Smoke Test:" -ForegroundColor Yellow
    Write-Host "  Validates: 78 dims, Stage 1 weights (4.0, 12.0), no crashes" -ForegroundColor Gray
    Write-Host "  Expected time: ~30 seconds" -ForegroundColor Gray
    Write-Host ""
} elseif ($Phase -eq "stage1") {
    Write-Host "20M Validation Run (Stage 1):" -ForegroundColor Yellow
    Write-Host "  Curriculum Stage 1 weights:" -ForegroundColor White
    Write-Host "    Position: 4.0 (40% of final 10.0)" -ForegroundColor Gray
    Write-Host "    Orientation: 12.0 (40% of final 30.0)" -ForegroundColor Gray
    Write-Host "    Ratio: 1:3 maintained throughout" -ForegroundColor Gray
    Write-Host "  Trajectories: chassis_required only (easier paths)" -ForegroundColor Gray
    Write-Host "  Expected time: ~5 hours" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Success Criteria @ 20M:" -ForegroundColor Cyan
    Write-Host "    Position error: <350cm mean" -ForegroundColor Gray
    Write-Host "    Orientation error: <80° mean (50% improvement vs 8g's 130°)" -ForegroundColor Gray
    Write-Host "    Workspace distance: 0.50-0.65m converged" -ForegroundColor Gray
    Write-Host "    Unreachable %: <10% (vs 8g's 78%!)" -ForegroundColor Gray
    Write-Host "    Explained variance: >0.3" -ForegroundColor Gray
    Write-Host "    KL divergence: 0.01-0.05" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  If ALL criteria pass → Proceed to full 100M run" -ForegroundColor Green
    Write-Host "  If ANY fail → Diagnose and adjust before 100M" -ForegroundColor Red
    Write-Host ""
} elseif ($Phase -eq "full") {
    Write-Host "Full Training Run (0-100M):" -ForegroundColor Yellow
    Write-Host "  0-45M: Stage 1 (reduced weights: pos=4.0, ori=12.0)" -ForegroundColor Gray
    Write-Host "  45-55M: Gradual transition (linear interpolation)" -ForegroundColor Gray
    Write-Host "  55-100M: Stage 2 (full weights: pos=10.0, ori=30.0)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Monitoring Thresholds:" -ForegroundColor White
    Write-Host "    Auto-pause if KL divergence > 0.1" -ForegroundColor Gray
    Write-Host "    Auto-pause if explained variance < 0" -ForegroundColor Gray
    Write-Host "    Checkpoint every 2M steps" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Expected time: ~14 hours" -ForegroundColor Gray
    Write-Host "  NOTE: Should only run after 20M validation passes!" -ForegroundColor Red
    Write-Host ""
}

# Session 8g lesson
Write-Host "Session 8g Lesson Learned:" -ForegroundColor Red
Write-Host "  Workspace expansion worked (position improved +2% vs 8f)" -ForegroundColor Gray
Write-Host "  BUT: Curriculum (5.0, 15.0) under-trained orientation" -ForegroundColor Gray
Write-Host "  AND: Instant 50M switch caused catastrophic collapse" -ForegroundColor Gray
Write-Host "  Session 8h fixes both with proportional weights + gradual transition" -ForegroundColor Gray
Write-Host ""

if ($Test) {
    Write-Host "Running environment sanity test..." -ForegroundColor Yellow
    & $IsaacLabBat -p $TestScript --num_envs 4 --headless
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[SUCCESS] Environment test passed!" -ForegroundColor Green
        Write-Host "  Observation space: 78 dims (76 + 2 workspace comfort)" -ForegroundColor Gray
        Write-Host "  Curriculum Stage 1: pos=4.0, ori=12.0" -ForegroundColor Gray
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
    "--learning_rate"
    $Config.LearningRate
    "--headless"
)

# Add trajectory filtering
if ($Config.TrajectoryStage) {
    # Session 8h: Use trajectory curriculum stages
    $TrainArgs += @("--trajectory_stage", $Config.TrajectoryStage)
    Write-Host "Using trajectory curriculum: $($Config.TrajectoryStage)" -ForegroundColor Cyan
} elseif ($Config.TrajectoryFilter -eq "chassis") {
    # Fall back to chassis-only as stage0 proxy
    $TrainArgs += @("--use_chassis_only")
    Write-Host "Using chassis-only trajectories (stage0 proxy until trajectories populated)" -ForegroundColor Yellow
}
# Note: Once trajectory stages are populated, set TrajectoryStage in config above

if ($Resume) {
    Write-Host "Resume mode enabled" -ForegroundColor Yellow
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

Write-Host "Launching Session 8h training..." -ForegroundColor Green
Write-Host "Command: $IsaacLabBat $($TrainArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Launch training
& $IsaacLabBat @TrainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    if ($Phase -eq "smoke") {
        Write-Host "`nNext Step: Run 20M validation" -ForegroundColor Yellow
        Write-Host "  .\scripts\launch_session_8h.ps1 -Phase stage1" -ForegroundColor Gray
    } elseif ($Phase -eq "stage1") {
        Write-Host "`nNext Step: Evaluate 20M checkpoint" -ForegroundColor Yellow
        Write-Host "  1. Run evaluation on final_model.zip (100 episodes)" -ForegroundColor Gray
        Write-Host "  2. Check ALL success criteria:" -ForegroundColor Gray
        Write-Host "     - Position: <350cm mean" -ForegroundColor Gray
        Write-Host "     - Orientation: <80° mean" -ForegroundColor Gray
        Write-Host "     - Workspace: 0.50-0.65m" -ForegroundColor Gray
        Write-Host "     - Unreachable: <10%" -ForegroundColor Gray
        Write-Host "  3. If ALL pass: Run full 100M with -Phase full" -ForegroundColor Gray
        Write-Host "  4. If ANY fail: Adjust config and restart stage1" -ForegroundColor Gray
    } elseif ($Phase -eq "full") {
        Write-Host "`nNext Step: Final evaluation" -ForegroundColor Yellow
        Write-Host "  1. Evaluate checkpoints: 40M, 60M, 80M, 100M" -ForegroundColor Gray
        Write-Host "  2. Compare with Session 8f baseline (308cm, 46.5°)" -ForegroundColor Gray
        Write-Host "  3. Check if gradual transition prevented collapse" -ForegroundColor Gray
        Write-Host "  4. Verify orientation improvement (target: <80°)" -ForegroundColor Gray
    }
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "Training failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}
