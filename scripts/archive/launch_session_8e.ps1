# Session 8e Training Launcher
# "Comfort Zone" refinement - fixes Session 8d orientation collapse
# 
# Key Changes from Session 8d:
# 1. Bell-shaped reachability bonus: Peaks at 0.5m (not linear from 0m)
# 2. Inner margin penalty: Prevents base from getting <0.35m (cramping arm)
# 3. Optimal working distance: 0.4-0.6m zone maximally rewarded
# 4. Expected outcome: Orientation < 25° (vs 47° in 8d), Position < 200cm
#
# Root Cause Fix (Session 8d):
# - Base rushed too close (<0.3m) to minimize reachability penalty
# - Result: Arm cramped, orientation collapsed (20° → 47°)
# - Solution: Bell curve rewards 0.5m distance, penalizes <0.35m
#
# Training Strategy:
# - Complete run with 16,384 envs, 150-200M steps
# - Monitor workspace_distance_mean (target: 0.45-0.55m, not 0.2m!)
# - If successful: Deploy. If not: Consider curriculum (Session 8f)
#
# Hardware: RTX 4090, ~6-8 hours for 200M steps

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('smoke', 'easy', 'medium', 'full', 'complete')]
    [string]$Phase = 'complete',
    
    [Parameter(Mandatory=$false)]
    [string]$Checkpoint = $null,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoHeadless,
    
    [Parameter(Mandatory=$false)]
    [switch]$Test
)

# Environment setup
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

# Base paths
$IsaacLabPath = "I:\isaaclab"
$ProjectPath = "C:\Users\yanbo\wSpace\cinebotRL"
$TrainScript = "$ProjectPath\scripts\reinforcement_learning\sb3\train.py"

# Common arguments
$CommonArgs = @(
    "--task", "MobileMMTrackEE-v0",
    "--learning_rate", "3e-4",
    "--n_epochs", "4",  # Session 8e: Keep 4 epochs for ~48 minibatches/update
    "--enable_entropy_decay",
    "--final_ent_coef", "1e-4",
    "--enable_kl_schedule",
    "--clip_range_vf", "0.3",  # Stabilize critic with reachability shaping
    "--trajectory_type", "multi_recorded",
    "--trajectory_dir", "trajectoryToLearn/world_json",
    "--use_all_trajectories"
)

if (-not $NoHeadless) {
    $CommonArgs += "--headless"
}

# Phase-specific configurations
switch ($Phase) {
    'smoke' {
        Write-Host "=== Session 8e: SMOKE TEST (10M, 32 envs, 64 steps) ===" -ForegroundColor Cyan
        Write-Host "Testing 'Comfort Zone' reward configuration:" -ForegroundColor Yellow
        Write-Host "  - Bell-shaped bonus: Peaks at 0.5m (not 0.0m)" -ForegroundColor White
        Write-Host "  - Inner margin penalty: Active at <0.35m" -ForegroundColor White
        Write-Host "  - Expected: Base stays 0.4-0.6m (not <0.3m like 8d)" -ForegroundColor White
        
        $PhaseArgs = @(
            "--num_envs", "32",
            "--n_steps", "64",
            "--batch_size", "2048",
            "--total_timesteps", "10000000",
            "--ent_coef", "0.001",
            "--decay_start_timestep", "120000000",  # Won't trigger in smoke test
            "--decay_duration_timesteps", "80000000",
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",
            "--save_freq", "2000000"
        )
    }
    
    'easy' {
        Write-Host "=== Session 8e: EASY Phase (40M timesteps, curriculum start) ===" -ForegroundColor Green
        Write-Host "NOTE: Start with high parallelism. Curriculum optional." -ForegroundColor Yellow
        
        $PhaseArgs = @(
            "--num_envs", "128",
            "--n_steps", "96",
            "--batch_size", "2048",
            "--total_timesteps", "40000000",
            "--ent_coef", "0.001",
            "--decay_start_timestep", "120000000",  # No decay in easy phase
            "--decay_duration_timesteps", "80000000",
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",
            "--save_freq", "2000000"
        )
    }
    
    'medium' {
        Write-Host "=== Session 8e: MEDIUM Phase (60M timesteps, curriculum mid) ===" -ForegroundColor Yellow
        if (-not $Checkpoint) {
            Write-Host "ERROR: -Checkpoint required for medium phase (provide easy phase final model)" -ForegroundColor Red
            exit 1
        }
        $PhaseArgs = @(
            "--num_envs", "160",
            "--n_steps", "96",
            "--batch_size", "2048",
            "--total_timesteps", "60000000",
            "--ent_coef", "0.001",
            "--decay_start_timestep", "120000000",  # No decay yet
            "--decay_duration_timesteps", "80000000",
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",
            "--save_freq", "2000000",
            "--checkpoint", $Checkpoint
        )
    }
    
    'full' {
        Write-Host "=== Session 8e: FULL Phase (100M timesteps, all trajectories) ===" -ForegroundColor Magenta
        if (-not $Checkpoint) {
            Write-Host "ERROR: -Checkpoint required for full phase (provide medium phase final model)" -ForegroundColor Red
            exit 1
        }
        $PhaseArgs = @(
            "--num_envs", "192",
            "--n_steps", "96",
            "--batch_size", "2048",
            "--total_timesteps", "100000000",
            "--ent_coef", "0.001",
            "--decay_start_timestep", "20000000",  # Decay starts at 20M into full phase (120M total)
            "--decay_duration_timesteps", "80000000",
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",
            "--save_freq", "2000000",
            "--checkpoint", $Checkpoint
        )
    }
    
    'complete' {
        Write-Host "=== Session 8e: COMPLETE Run (200M timesteps, COMFORT ZONE) ===" -ForegroundColor White
        Write-Host ""
        Write-Host "Session 8e 'Goldilocks Zone' Refinements:" -ForegroundColor Cyan
        Write-Host "  ✅ Bell-shaped bonus: Peaks at 0.5m working distance" -ForegroundColor Green
        Write-Host "  ✅ Inner margin penalty: Prevents <0.35m cramping" -ForegroundColor Green
        Write-Host "  ✅ No dead zone: Smooth gradients 0.3-0.7m" -ForegroundColor Green
        Write-Host "  ✅ Orientation priority: Maintained at 200 weight" -ForegroundColor Green
        Write-Host ""
        Write-Host "Expected Performance:" -ForegroundColor Yellow
        Write-Host "  Target: Position error < 200cm (vs 311cm in 8d)" -ForegroundColor White
        Write-Host "  Target: Orientation error < 25° (vs 47° in 8d)" -ForegroundColor White
        Write-Host "  Target: Base @ 0.4-0.6m > 60% time (vs 20% in 8d)" -ForegroundColor White
        Write-Host "  If achieved: Deploy to real robot!" -ForegroundColor White
        Write-Host "  If orientation still >30°: Increase orientation weight to 250" -ForegroundColor White
        Write-Host ""
        Write-Host "Session 8d Failure Analysis:" -ForegroundColor Red
        Write-Host "  ❌ Orientation collapsed: 20° → 47° (+131%)" -ForegroundColor Red
        Write-Host "  ❌ Base rushed too close: <0.3m (cramped arm)" -ForegroundColor Red
        Write-Host "  ❌ Old reward: Linear from 0m (encouraged hugging)" -ForegroundColor Red
        Write-Host "  ✅ New reward: Bell curve @ 0.5m (optimal workspace)" -ForegroundColor Green
        Write-Host ""
        
        # HIGH PARALLELISM OPTIONS:
        # 16,384 envs: ~6-8 hours, 18-20 GB VRAM (RECOMMENDED)
        # 8,192 envs:  ~12-15 hours, 12-15 GB VRAM (conservative)
        
        $PhaseArgs = @(
            "--num_envs", "16384",  # Change to 8192 if GPU memory limited
            "--n_steps", "128",     # Larger rollout buffer for better sample efficiency
            "--batch_size", "4096", # Larger batches to match increased sample count
            "--total_timesteps", "200000000",
            "--ent_coef", "0.001",
            "--decay_start_timestep", "120000000",  # Decay after 120M steps
            "--decay_duration_timesteps", "80000000",  # Decay to 1e-4 by 200M
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",
            "--save_freq", "2000000"
        )
        if ($Checkpoint) {
            $PhaseArgs += @("--checkpoint", $Checkpoint)
        }
    }
}

# Build complete command
$AllArgs = @($IsaacLabPath, "isaaclab.bat", "-p", $TrainScript) + $CommonArgs + $PhaseArgs

Write-Host "`nLaunching Isaac Lab with arguments:" -ForegroundColor Cyan
Write-Host ($AllArgs -join " ") -ForegroundColor Gray
Write-Host ""

# Run test script first if requested
if ($Test) {
    Write-Host "=== Running Environment Test First ===" -ForegroundColor Yellow
    Set-Location $IsaacLabPath
    & .\isaaclab.bat -p "$ProjectPath\scripts\test_mobile_mm_env.py" --task MobileMMTrackEE-v0 --num_envs 2 --headless
    $testExitCode = $LASTEXITCODE
    Set-Location $ProjectPath
    
    if ($testExitCode -ne 0) {
        Write-Host "Environment test FAILED! Fix issues before training." -ForegroundColor Red
        exit 1
    }
    Write-Host "Environment test PASSED! Proceeding with training..." -ForegroundColor Green
    Write-Host ""
}

# Change to Isaac Lab directory and run training
Set-Location $IsaacLabPath
& .\isaaclab.bat -p $TrainScript @CommonArgs @PhaseArgs
$trainExitCode = $LASTEXITCODE

# Return to project directory
Set-Location $ProjectPath

if ($trainExitCode -eq 0) {
    Write-Host "`n=== Training Complete ===" -ForegroundColor Green
    Write-Host "Check logs in: logs/sb3/mobilemmtrackee_v0/<timestamp>" -ForegroundColor Yellow
    Write-Host "Checkpoints saved every 2M steps" -ForegroundColor Yellow
    
    if ($Phase -eq 'smoke') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Check TensorBoard for:" -ForegroundColor White
        Write-Host "   - monitoring/workspace_distance_mean (target: 0.45-0.55m, NOT 0.2m!)" -ForegroundColor White
        Write-Host "   - reward_components/reachability_bonus (should be 25-35)" -ForegroundColor White
        Write-Host "   - reward_components/inner_margin_penalty (should be <5)" -ForegroundColor White
        Write-Host "   - reward_components/orientation_tracking (should be 85-95 vs 70 in 8d)" -ForegroundColor White
        Write-Host "2. If smoke test looks good, run: .\scripts\launch_session_8e.ps1 -Phase complete" -ForegroundColor White
    } elseif ($Phase -eq 'easy') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Evaluate at 40M checkpoint" -ForegroundColor White
        Write-Host "2. If orientation < 30° AND position < 250cm, run: .\scripts\launch_session_8e.ps1 -Phase medium -Checkpoint <path>" -ForegroundColor White
        Write-Host "3. If orientation still >40°, increase orientation_tracking weight to 250 and restart" -ForegroundColor White
    } elseif ($Phase -eq 'medium') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Evaluate at 100M checkpoint (40+60)" -ForegroundColor White
        Write-Host "2. If both metrics improving, run: .\scripts\launch_session_8e.ps1 -Phase full -Checkpoint <path>" -ForegroundColor White
    } elseif ($Phase -eq 'full') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Evaluate final 200M checkpoint (40+60+100)" -ForegroundColor White
        Write-Host "2. Compare Session 8b vs 8c-v2 vs 8d vs 8e results" -ForegroundColor White
        Write-Host "3. If orientation < 25° AND position < 200cm: SUCCESS! Deploy!" -ForegroundColor White
        Write-Host "4. If orientation still >30°: Plan Session 8f with higher orientation weight" -ForegroundColor White
    } elseif ($Phase -eq 'complete') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Parse TensorBoard data:" -ForegroundColor White
        Write-Host "   python scripts/parse_tensorboard.py logs/sb3/mobilemmtrackee_v0/<timestamp>" -ForegroundColor Gray
        Write-Host ""
        Write-Host "2. Evaluate at 50M checkpoint (early check):" -ForegroundColor White
        Write-Host "   cd I:\isaaclab" -ForegroundColor Gray
        Write-Host "   .\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate_quantitative.py \" -ForegroundColor Gray
        Write-Host "     --checkpoint <logs/.../ppo_mobile_mm_50000000_steps.zip> \" -ForegroundColor Gray
        Write-Host "     --num_episodes 200 \" -ForegroundColor Gray
        Write-Host "     --num_envs 64 \" -ForegroundColor Gray
        Write-Host "     --headless \" -ForegroundColor Gray
        Write-Host "     --output_dir C:\Users\yanbo\wSpace\cinebotRL\evaluation_plots\session_8e_50M" -ForegroundColor Gray
        Write-Host ""
        Write-Host "3. Key metrics to check (vs Session 8d @ 109M):" -ForegroundColor White
        Write-Host "   Orientation error: Target <30° (vs 47.4° in 8d)" -ForegroundColor Yellow
        Write-Host "   Position error: Target <250cm (vs 311cm in 8d)" -ForegroundColor Yellow
        Write-Host "   Workspace distance: Target 0.45-0.55m (vs 0.3m in 8d)" -ForegroundColor Yellow
        Write-Host "   Reachability bonus: Target 25-35 (vs 7.06 in 8d)" -ForegroundColor Yellow
        Write-Host "   Inner margin penalty: Target <5 (new metric)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "4. Decision tree @ 50M:" -ForegroundColor White
        Write-Host "   Orientation < 30° + Position < 250cm → Continue to 150M" -ForegroundColor Green
        Write-Host "   Orientation 30-40° → Promising, monitor closely" -ForegroundColor Yellow
        Write-Host "   Orientation > 40° → STOP, increase orientation weight to 250" -ForegroundColor Red
        Write-Host ""
        Write-Host "5. If training completes successfully (200M):" -ForegroundColor White
        Write-Host "   - Full evaluation @ 150M and 200M checkpoints" -ForegroundColor White
        Write-Host "   - Generate comparison plots (8b vs 8c-v2 vs 8d vs 8e)" -ForegroundColor White
        Write-Host "   - Create SESSION_8E_RESULTS.md analysis" -ForegroundColor White
        Write-Host "   - If orientation < 25° AND position < 200cm: Deploy to real robot!" -ForegroundColor Green
        Write-Host ""
        Write-Host "6. Success criteria (Session 8e):" -ForegroundColor Cyan
        Write-Host "   PRIMARY: Fix orientation collapse (47° → <25°)" -ForegroundColor White
        Write-Host "   SECONDARY: Maintain or improve position (<250cm)" -ForegroundColor White
        Write-Host "   TERTIARY: Base stays in comfort zone (0.4-0.6m > 60% time)" -ForegroundColor White
    }
} else {
    Write-Host "`n=== Training Failed ===" -ForegroundColor Red
    Write-Host "Check logs for errors" -ForegroundColor Yellow
    exit $trainExitCode
}
