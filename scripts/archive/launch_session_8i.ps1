# Session 8i Launcher - Distance-Gated Orientation Rewards
# ==============================================================================
# Session 8i builds on Session 8h@40M baseline to fix orientation issues:
#
# Session 8h @ 40M Results (Baseline):
#   Position: 237.3 cm (EXCELLENT - best ever!)
#   Orientation: 135.1° (POOR - needs 60% improvement)
#   Workspace: Converged to 0.554m
#
# Session 8h @ 100M Results (Regression):
#   Position: 302.4 cm (REGRESSED +27%)
#   Orientation: 119.1° (Better but position collapsed)
#   Issue: Curriculum transition caused value function shock
#
# Session 8i Strategy (Distance-Gated Orientation):
#   1. Use 8h@40M as starting checkpoint (optional - can train from scratch)
#   2. Distance-gated rewards separate "reach" from "align" modes:
#      - Far (>0.7m): Low ori weight (4.0) - focus on base mobilization
#      - Close (<0.7m): High ori weight (30.0) - focus on precise alignment
#   3. Observation enhancements (70→73 dims):
#      - Axis-angle error: Shortest rotation path (+3 dims)
#      - EE angular velocity: Already computed, now explicit in obs
#   4. New reward components:
#      - orientation_progress_bonus: Reward ori error reduction
#      - angular_velocity_penalty: Penalize excessive angular velocity
#   5. Training strategy:
#      - Evaluate every 10M steps
#      - Stop if ori degrades >15% or position >250cm
#      - Target: 80-100° orientation, maintain ~237cm position
#
# Expected Outcomes:
#   Conservative: 80-100° orientation, 240-250cm position
#   Optimistic: 60-80° orientation, 230-240cm position
# ==============================================================================

param(
    [ValidateSet("smoke", "short", "full", "continuation")]
    [string]$Phase = "smoke",
    [switch]$Test,
    [switch]$Resume,
    [string]$CheckpointPath = ""  # Optional: Override default checkpoint
)

$ErrorActionPreference = "Stop"

# Configuration
$IsaacLabPath = "I:\isaaclab"
$IsaacLabBat = Join-Path $IsaacLabPath "isaaclab.bat"
$ProjectRoot = "C:\Users\yanbo\wSpace\cinebotRL"
$TrainScript = Join-Path $ProjectRoot "scripts\reinforcement_learning\sb3\train.py"
$TestScript = Join-Path $ProjectRoot "scripts\test_mobile_mm_env.py"
$TaskName = "MobileMMTrackEE-v0"

# Session 8i configuration
$SessionConfig = @{
    smoke = @{
        NumEnvs = 64
        TotalSteps = 500000
        TrajectoryFilter = "chassis"
        LearningRate = "2e-4"
        Description = "Smoke test with distance-gated orientation (73 dims)"
        CheckpointPath = $null  # Train from scratch for smoke test
        AutoFindCheckpoint = $false
    }
    short = @{
        NumEnvs = 16384
        TotalSteps = 40000000
        TrajectoryFilter = "chassis"
        LearningRate = "2e-4"
        Description = "40M run: Validate distance-gated approach (0→40M)"
        CheckpointPath = $null  # Train from scratch
        AutoFindCheckpoint = $false
    }
    continuation = @{
        NumEnvs = 16384
        TotalSteps = 120000000
        TrajectoryFilter = "chassis"
        LearningRate = "2e-4"
        Description = "Continue from 40M to 120M (auto-find checkpoint)"
        CheckpointPath = $null  # Will auto-find latest 40M checkpoint
        AutoFindCheckpoint = $true
        TargetSteps = 40000000  # Look for checkpoint near this step count
    }
    full = @{
        NumEnvs = 16384
        TotalSteps = 120000000
        TrajectoryFilter = "chassis"
        LearningRate = "2e-4"
        Description = "Full 120M from scratch (0→120M, independent run)"
        CheckpointPath = $null  # Train from scratch
        AutoFindCheckpoint = $false
    }
}

$Config = $SessionConfig[$Phase]

# Auto-find checkpoint if enabled
if ($Config.AutoFindCheckpoint -and -not $CheckpointPath) {
    Write-Host "`nAuto-finding checkpoint near $($Config.TargetSteps) steps..." -ForegroundColor Cyan
    $logsPath = Join-Path $ProjectRoot "logs\sb3\mobilemmtrackee_v0"
    if (Test-Path $logsPath) {
        $targetSteps = $Config.TargetSteps
        $tolerance = 5000000  # ±5M steps tolerance
        
        # Find checkpoints near target steps
        $foundCheckpoints = Get-ChildItem $logsPath -Recurse -Filter "*.zip" | Where-Object {
            $_.Name -match "(\d+)_steps\.zip$"
            $steps = [long]$matches[1]
            [Math]::Abs($steps - $targetSteps) -le $tolerance
        } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($foundCheckpoints) {
            $Config.CheckpointPath = $foundCheckpoints.FullName
            Write-Host "Found checkpoint: $($Config.CheckpointPath)" -ForegroundColor Green
            $steps = [long]($foundCheckpoints.Name -replace ".*_(\d+)_steps\.zip", '$1')
            Write-Host "  Steps: $($steps / 1000000)M" -ForegroundColor Gray
            Write-Host "  Will train from $($steps / 1000000)M to $($Config.TotalSteps / 1000000)M" -ForegroundColor Gray
        } else {
            Write-Host "WARNING: No checkpoint found near $($targetSteps / 1000000)M steps!" -ForegroundColor Red
            Write-Host "  Searched in: $logsPath" -ForegroundColor Gray
            Write-Host "  Please run 'short' phase first, or specify -CheckpointPath manually" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host "ERROR: Logs directory not found: $logsPath" -ForegroundColor Red
        exit 1
    }
}

# Override checkpoint if provided via parameter
if ($CheckpointPath) {
    $Config.CheckpointPath = $CheckpointPath
}

# Helper: Find available checkpoints from recent training runs
function Find-RecentCheckpoints {
    $logsPath = Join-Path $ProjectRoot "logs\sb3\mobilemmtrackee_v0"
    if (Test-Path $logsPath) {
        $recentDirs = Get-ChildItem $logsPath -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 3
        Write-Host "`nAvailable Recent Checkpoints:" -ForegroundColor Cyan
        foreach ($dir in $recentDirs) {
            $checkpointDir = Join-Path $dir.FullName "checkpoints"
            if (Test-Path $checkpointDir) {
                $checkpoints = Get-ChildItem $checkpointDir -Filter "*.zip" | Sort-Object Name -Descending | Select-Object -First 5
                if ($checkpoints.Count -gt 0) {
                    Write-Host "  $($dir.Name):" -ForegroundColor Yellow
                    foreach ($cp in $checkpoints) {
                        $relativePath = $cp.FullName.Replace("$ProjectRoot\", "")
                        Write-Host "    - $relativePath" -ForegroundColor Gray
                    }
                }
            }
        }
        Write-Host ""
    }
}

# Show checkpoint info if resuming or in non-smoke phase
if ($Phase -ne "smoke" -or $Resume) {
    Find-RecentCheckpoints
    if ($Config.CheckpointPath) {
        Write-Host "Using checkpoint: $($Config.CheckpointPath)" -ForegroundColor Green
    } else {
        Write-Host "Training from scratch (no checkpoint specified)" -ForegroundColor Yellow
        Write-Host "  To resume from checkpoint, use: -CheckpointPath <path>" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Session 8i: Distance-Gated Orientation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase: $Phase" -ForegroundColor Yellow
Write-Host "Description: $($Config.Description)" -ForegroundColor Gray
Write-Host "Environments: $($Config.NumEnvs)" -ForegroundColor Gray
Write-Host "Total Steps: $($Config.TotalSteps)" -ForegroundColor Gray
Write-Host "Trajectory Filter: $($Config.TrajectoryFilter)" -ForegroundColor Gray
Write-Host ""

# Session 8i innovations
Write-Host "Session 8i Innovations:" -ForegroundColor Green
Write-Host "  1. Distance-gated orientation rewards:" -ForegroundColor White
Write-Host "     Far (>0.7m): ori_weight=4.0 (focus on reaching)" -ForegroundColor Cyan
Write-Host "     Close (<0.7m): ori_weight=30.0 (focus on alignment)" -ForegroundColor Cyan
Write-Host "  2. Observation enhancements (+3 dims → 73 total):" -ForegroundColor White
Write-Host "     Axis-angle error: Shortest rotation path" -ForegroundColor Cyan
Write-Host "  3. New reward components:" -ForegroundColor White
Write-Host "     Orientation progress bonus: Reward ori improvement" -ForegroundColor Cyan
Write-Host "     Angular velocity penalty: Encourage smoothness" -ForegroundColor Cyan
Write-Host "  4. Early stopping strategy:" -ForegroundColor White
Write-Host "     Stop if ori degrades >15% or position >250cm" -ForegroundColor Cyan
Write-Host ""

# Baseline comparison
Write-Host "Baseline (Session 8h @ 40M):" -ForegroundColor Magenta
Write-Host "  Position: 237.3 cm (BEST EVER)" -ForegroundColor Gray
Write-Host "  Orientation: 135.1° (TARGET: 80-100°)" -ForegroundColor Gray
Write-Host "  Workspace: 0.554m (converged)" -ForegroundColor Gray
Write-Host "  Unreachable: 3.1% (excellent)" -ForegroundColor Gray
Write-Host ""

# Phase-specific info
if ($Phase -eq "smoke") {
    Write-Host "Smoke Test:" -ForegroundColor Yellow
    Write-Host "  Validates: 73 dims, distance-gated logic, no crashes" -ForegroundColor Gray
    Write-Host "  Expected time: ~30 seconds" -ForegroundColor Gray
    Write-Host ""
} elseif ($Phase -eq "short") {
    Write-Host "Phase 2: 40M Validation Run (0→40M)" -ForegroundColor Yellow
    Write-Host "  Starting from: Scratch (clean baseline)" -ForegroundColor White
    Write-Host "  Distance gate: 0.7m threshold" -ForegroundColor Gray
    Write-Host "  Orientation weights:" -ForegroundColor White
    Write-Host "    Far (>0.7m): 4.0 (low priority, avoid interfering with base)" -ForegroundColor Gray
    Write-Host "    Close (<0.7m): 30.0 (high priority, precise alignment)" -ForegroundColor Gray
    Write-Host "  Expected time: ~6 hours" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Success Criteria @ 40M:" -ForegroundColor Cyan
    Write-Host "    Orientation: <110° mean (20% improvement from 135.1°)" -ForegroundColor Gray
    Write-Host "    Position: <250cm mean (maintain ~237cm)" -ForegroundColor Gray
    Write-Host "    Workspace distance: 0.50-0.65m" -ForegroundColor Gray
    Write-Host "    Unreachable %: <10%" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Decision Point:" -ForegroundColor Magenta
    Write-Host "    ✅ If ALL criteria pass → Run: .\scripts\launch_session_8i.ps1 -Phase continuation" -ForegroundColor Green
    Write-Host "    ❌ If ANY criteria fail → Diagnose issue, adjust config, restart from scratch" -ForegroundColor Red
    Write-Host "       Do NOT continue to Phase 3 - bad training state will carry over!" -ForegroundColor Red
    Write-Host ""
} elseif ($Phase -eq "continuation") {
    Write-Host "Phase 3: Continuation Run (40M→120M)" -ForegroundColor Yellow
    Write-Host "  ⚠️  PREREQUISITE: Phase 2 (short) must pass ALL success criteria!" -ForegroundColor Magenta
    Write-Host "  Starting from: Auto-found 40M checkpoint from 'short' phase" -ForegroundColor White
    Write-Host "  Distance gate: 0.7m threshold (fixed throughout)" -ForegroundColor Gray
    Write-Host "  Evaluation milestones: 50M, 60M, 70M, 80M, ..." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  ⚠️  Safety Check:" -ForegroundColor Red
    Write-Host "    If Phase 2 failed any criteria, do NOT run this phase!" -ForegroundColor Red
    Write-Host "    Bad training state will carry over to Phase 3." -ForegroundColor Red
    Write-Host "    Instead: Diagnose issue → Adjust config → Run 'full' phase from scratch" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Monitoring Strategy:" -ForegroundColor White
    Write-Host "    Evaluate every 10M steps" -ForegroundColor Gray
    Write-Host "    Stop if orientation degrades >15% vs previous milestone" -ForegroundColor Gray
    Write-Host "    Stop if position error >250cm mean" -ForegroundColor Gray
    Write-Host "    Checkpoint every 2M steps" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Expected time: ~10 hours (80M additional steps)" -ForegroundColor Gray
    Write-Host "  NOTE: Watch for orientation improvement without position collapse!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Target @ 120M:" -ForegroundColor Cyan
    Write-Host "    Orientation: 80-100° (40% improvement)" -ForegroundColor Green
    Write-Host "    Position: ~237cm (maintain baseline)" -ForegroundColor Green
    Write-Host ""
} elseif ($Phase -eq "full") {
    Write-Host "Full Training Run (0→120M, Independent)" -ForegroundColor Yellow
    Write-Host "  Starting from: Scratch (NOT continuing from 'short')" -ForegroundColor White
    Write-Host "  Use cases:" -ForegroundColor Cyan
    Write-Host "    1. You want a fresh 120M run without validation checkpoint" -ForegroundColor Gray
    Write-Host "    2. Phase 2 (short) failed criteria and you've fixed the config" -ForegroundColor Gray
    Write-Host "    3. Starting clean after config changes to reward weights/gating" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Distance gate: 0.7m threshold (fixed throughout)" -ForegroundColor Gray
    Write-Host "  Evaluation milestones: 10M, 20M, 30M, 40M, 50M, ..." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Monitoring Strategy:" -ForegroundColor White
    Write-Host "    Evaluate every 10M steps" -ForegroundColor Gray
    Write-Host "    Stop if orientation degrades >15% vs previous milestone" -ForegroundColor Gray
    Write-Host "    Stop if position error >250cm mean" -ForegroundColor Gray
    Write-Host "    Checkpoint every 2M steps" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Expected time: ~16 hours" -ForegroundColor Gray
    Write-Host "  NOTE: If you want validation checkpoint, use 'short' → 'continuation' instead!" -ForegroundColor Yellow
    Write-Host "        Only use 'full' if you need clean restart after config fix." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Target @ 120M:" -ForegroundColor Cyan
    Write-Host "    Orientation: 80-100° (40% improvement)" -ForegroundColor Green
    Write-Host "    Position: ~237cm (maintain baseline)" -ForegroundColor Green
    Write-Host ""
}

# Key differences from Session 8h
Write-Host "Key Differences from Session 8h:" -ForegroundColor Yellow
Write-Host "  Observation space: 70 → 73 dims (axis-angle error)" -ForegroundColor Gray
Write-Host "  Orientation reward: Fixed → Distance-gated (4.0/30.0)" -ForegroundColor Gray
Write-Host "  New components: Progress bonus + angular vel penalty" -ForegroundColor Gray
Write-Host "  Strategy: Separate reach/align modes vs unified curriculum" -ForegroundColor Gray
Write-Host ""

# Important notes
Write-Host "IMPORTANT NOTES:" -ForegroundColor Red
Write-Host "  1. Observation space changed (70→73 dims)" -ForegroundColor Yellow
Write-Host "     → VecNormalize stats from 8h are INCOMPATIBLE" -ForegroundColor Yellow
Write-Host "     → Recommended: Train from scratch for clean baseline" -ForegroundColor Yellow
Write-Host "  2. If resuming from checkpoint:" -ForegroundColor Yellow
Write-Host "     → Policy will adapt to new observation space" -ForegroundColor Yellow
Write-Host "     → Expect ~10-20M steps for adaptation period" -ForegroundColor Yellow
Write-Host "  3. Distance gating is SPATIAL, not temporal" -ForegroundColor Yellow
Write-Host "     → Each env switches modes independently based on distance" -ForegroundColor Yellow
Write-Host "     → No curriculum scheduling needed" -ForegroundColor Yellow
Write-Host "  4. Trajectory curriculum NOT YET ENABLED" -ForegroundColor Yellow
Write-Host "     → Currently using chassis-only trajectories" -ForegroundColor Yellow
Write-Host "     → stage0/1/2/3 directories need trajectory JSON files" -ForegroundColor Yellow
Write-Host "  5. Orientation monitoring callback NOT YET IMPLEMENTED" -ForegroundColor Yellow
Write-Host "     → Manual evaluation needed at milestones (10M, 20M, ...)" -ForegroundColor Yellow
Write-Host "     → Consider adding custom callback in train.py for auto-stopping" -ForegroundColor Yellow
Write-Host ""

if ($Test) {
    Write-Host "Running environment sanity test..." -ForegroundColor Yellow
    & $IsaacLabBat -p $TestScript --num_envs 4 --headless
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[SUCCESS] Environment test passed!" -ForegroundColor Green
        Write-Host "  Observation space: 73 dims (70 + 3 axis-angle)" -ForegroundColor Gray
        Write-Host "  Distance gate: 0.7m threshold" -ForegroundColor Gray
        Write-Host "  Orientation weights: far=4.0, close=30.0" -ForegroundColor Gray
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
if ($Config.TrajectoryFilter -eq "chassis") {
    $TrainArgs += @("--use_chassis_only")
    Write-Host "Using chassis-only trajectories" -ForegroundColor Cyan
}

# Resume from checkpoint if specified
if ($Config.CheckpointPath) {
    if (Test-Path $Config.CheckpointPath) {
        $TrainArgs += @("--checkpoint", $Config.CheckpointPath)
        Write-Host "`nResuming from checkpoint:" -ForegroundColor Green
        Write-Host "  Path: $($Config.CheckpointPath)" -ForegroundColor Gray
        Write-Host "  NOTE: Policy will adapt to new 73-dim observation space" -ForegroundColor Yellow
        Write-Host "  Expected adaptation period: 10-20M steps" -ForegroundColor Yellow
    } else {
        Write-Host "`nERROR: Checkpoint not found!" -ForegroundColor Red
        Write-Host "  Path: $($Config.CheckpointPath)" -ForegroundColor Gray
        Write-Host "  Use Find-RecentCheckpoints or train from scratch" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "`nTraining from scratch (no checkpoint)" -ForegroundColor Yellow
    Write-Host "  Recommended for Session 8i due to observation space change" -ForegroundColor Gray
}

# Add session tag for logging
$TrainArgs += @("--wandb_project", "cinebotrl_session_8i")

Write-Host "`nLaunching training with command:" -ForegroundColor Cyan
Write-Host "  $IsaacLabBat $($TrainArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Confirmation for long runs
if ($Phase -eq "full" -and -not $Resume) {
    Write-Host "WARNING: Full training run (~16 hours)" -ForegroundColor Red
    $confirmation = Read-Host "Continue? (y/n)"
    if ($confirmation -ne 'y') {
        Write-Host "Training cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Launch training
Write-Host "`n[TRAINING START] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
& $IsaacLabBat @TrainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[TRAINING COMPLETE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Evaluate at milestones (10M, 20M, 30M, 40M, ...)" -ForegroundColor Gray
    Write-Host "  2. Compare with Session 8h @ 40M baseline:" -ForegroundColor Gray
    Write-Host "     Position: 237.3cm → Target: ~237cm (maintain)" -ForegroundColor Gray
    Write-Host "     Orientation: 135.1° → Target: 80-100° (improve)" -ForegroundColor Gray
    Write-Host "  3. Monitor distance-gated reward components:" -ForegroundColor Gray
    Write-Host "     - orientation_progress_bonus (should increase)" -ForegroundColor Gray
    Write-Host "     - angular_velocity_penalty (should decrease)" -ForegroundColor Gray
    Write-Host "  4. Check spatial behavior:" -ForegroundColor Gray
    Write-Host "     - Far envs: Low ori reward, focus on reaching" -ForegroundColor Gray
    Write-Host "     - Close envs: High ori reward, focus on alignment" -ForegroundColor Gray
} else {
    Write-Host "`n[TRAINING FAILED] Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
