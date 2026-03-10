# Session 8f Launcher - Distance-Gated Penalty System + Playbook Fixes
# ==============================================================================
# Session 8f implements critical fixes from mobile_mm_training_playbook.md:
# 1. Atomic root state write (§1) - fixes control conflict
# 2. Distance-gated penalties (§2) - THE solution to workspace drift
# 3. Heading cue in observations (§3) - sin/cos yaw error (+2 obs dims)
# 4. Two-zone linear reachability - simpler than bell-shaped curve
#
# Session 8e Analysis (FAILURE):
# - Reachability bonus: 7.06 → 0.79 (89% drop!)
# - Workspace distance: 0.52m @ 50M → 0.58m @ 73M (drifting away)
# - Position error: 349cm (worse than 8d's 311cm)
# - Root cause: Bell-shaped peak too brittle, penalties fought mobilization
#
# Session 8f Strategy:
# - FAR mode (>0.55m): Penalties OFF, mobilization ON → "GO GET IT!"
# - NEAR mode (<0.55m): Penalties ON, precision mode → "BE PRECISE!"
# - Two-zone linear: 0.35-0.5m approach, 0.5-0.6m plateau, >0.6m decay
# ==============================================================================

param(
    [ValidateSet("smoke", "easy", "medium", "full", "complete")]
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

# Session 8f configuration (same as 8e but with code fixes)
$SessionConfig = @{
    smoke = @{
        NumEnvs = 64
        TotalSteps = 500000
        Description = "Quick smoke test (500K steps, ~2min)"
    }
    easy = @{
        NumEnvs = 4096
        TotalSteps = 2000000
        Description = "Phase 1: Observe mobilization behavior (2M steps, ~15min)"
    }
    medium = @{
        NumEnvs = 8192
        TotalSteps = 10000000
        Description = "Phase 2: Check distance gating (10M steps, ~1.5hr)"
    }
    full = @{
        NumEnvs = 16384
        TotalSteps = 100000000
        Description = "Phase 3: Full training (100M steps, ~14hr)"
    }
    complete = @{
        NumEnvs = 16384
        TotalSteps = 200000000
        Description = "Phase 4: Extended training (200M steps, ~28hr)"
    }
}

$Config = $SessionConfig[$Phase]

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Session 8f: Distance-Gated Penalty System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase: $Phase" -ForegroundColor Yellow
Write-Host "Description: $($Config.Description)" -ForegroundColor Gray
Write-Host "Environments: $($Config.NumEnvs)" -ForegroundColor Gray
Write-Host "Total Steps: $($Config.TotalSteps)" -ForegroundColor Gray
Write-Host ""

# Key changes from Session 8e
Write-Host "Key Changes from Session 8e:" -ForegroundColor Green
Write-Host "  1. Atomic root state write (fix control conflict)" -ForegroundColor White
Write-Host "  2. Distance-gated penalties (far=mobilize, near=precision)" -ForegroundColor White
Write-Host "  3. Heading cue +2 obs dims (sin/cos yaw error)" -ForegroundColor White
Write-Host "  4. Two-zone linear reachability (0.35-0.5-0.6m plateau)" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT: Observation dimension changed 49 → 51 dims" -ForegroundColor Yellow
Write-Host "  Network architecture incompatible with Session 8e" -ForegroundColor Yellow
Write-Host "  Cannot resume from 8e checkpoints - MUST start fresh" -ForegroundColor Yellow
Write-Host ""

if ($Test) {
    Write-Host "Running environment sanity test..." -ForegroundColor Yellow
    & $IsaacLabBat -p $TestScript --num_envs 4 --headless
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[SUCCESS] Environment test passed!" -ForegroundColor Green
    } else {
        Write-Host "`n[ERROR] Environment test failed!" -ForegroundColor Red
        exit 1
    }
    exit 0
}

# Prepare training command
$TrainArgs = @(
    "-p", $TrainScript,
    "--task", $TaskName,
    "--num_envs", $Config.NumEnvs,
    "--headless",
    "--trajectory_type", "multi_recorded",
    "--use_all_trajectories",
    "--total_timesteps", $Config.TotalSteps,
    "--enable_kl_schedule",
    "--enable_entropy_decay"
)

if ($Resume) {
    Write-Host "WARNING: Resume not implemented yet. Starting fresh training." -ForegroundColor Yellow
}

Write-Host "Launching Session 8f training..." -ForegroundColor Cyan
Write-Host "Command: $IsaacLabBat $($TrainArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Launch training
& $IsaacLabBat @TrainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Session 8f Phase $Phase Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Phase-specific next steps
    switch ($Phase) {
        "smoke" {
            Write-Host "`nNext Step: Run easy phase" -ForegroundColor Yellow
            Write-Host "  .\scripts\launch_session_8f.ps1 -Phase easy" -ForegroundColor White
        }
        "easy" {
            Write-Host "`nNext Step: Check metrics at 2M checkpoint" -ForegroundColor Yellow
            Write-Host "Expected:" -ForegroundColor Gray
            Write-Host "  - workspace_distance_mean: 0.48-0.52m (should be stable!)" -ForegroundColor White
            Write-Host "  - Explained variance: >0.85" -ForegroundColor White
            Write-Host "  - Entropy: -2.0 to -2.5 (healthy exploration)" -ForegroundColor White
            Write-Host ""
            Write-Host "If looks good: .\scripts\launch_session_8f.ps1 -Phase medium" -ForegroundColor White
        }
        "medium" {
            Write-Host "`nNext Step: Evaluate at 10M checkpoint" -ForegroundColor Yellow
            Write-Host "  I:\isaaclab\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate_quantitative.py --checkpoint <CHECKPOINT_PATH> --num_episodes 200 --num_envs 64 --output_dir evaluation_plots\session_8f_10M --headless" -ForegroundColor White
            Write-Host ""
            Write-Host "Expected improvements vs 8e @ 50M:" -ForegroundColor Gray
            Write-Host "  - reachability_bonus: >5.0 (vs 0.79 in 8e)" -ForegroundColor White
            Write-Host "  - position_error: <300cm (vs 349cm in 8e)" -ForegroundColor White
            Write-Host "  - orientation_error: <45° (vs 48.5° in 8e)" -ForegroundColor White
            Write-Host ""
            Write-Host "If improved: .\scripts\launch_session_8f.ps1 -Phase full" -ForegroundColor White
        }
        "full" {
            Write-Host "`nNext Step: Evaluate at 100M checkpoint" -ForegroundColor Yellow
            Write-Host "Compare with Session 8d @ 109M baseline:" -ForegroundColor Gray
            Write-Host "  8d: 311cm position, 47.4° orientation, workspace 0.402m (too close)" -ForegroundColor White
            Write-Host "  8f target: <280cm position, <40° orientation, workspace 0.48-0.52m" -ForegroundColor White
        }
        "complete" {
            Write-Host "`nSession 8f training complete!" -ForegroundColor Green
            Write-Host "Run final evaluation and compare with all previous sessions" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`n[ERROR] Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
