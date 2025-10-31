# Session 8d Training Launcher
# Based on Session 8c with improved reward shaping
# 
# Key Changes from Session 8c-v2:
# 1. Reachability penalty: 60% weaker (80× vs 100×, still quadratic)
# 2. Base mobilization: 3× stronger (450 vs 150, cap 0.35m vs 0.2m)
# 3. Linear position penalty: Active (weight=40) for gradient at large distances
# 4. Base alignment: 3× stronger (30 vs 10)
# 5. Smooth workspace distance: Active with soft/hard margins
# 6. Expected outcome: Position error < 250cm (vs 328cm in 8c-v2)
#
# Training Strategy:
# - Start with complete run (16,384 envs, 200M steps)
# - Test improved penalty-reward balance
# - If fails (>300cm), try curriculum in Session 8e
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
    "--n_epochs", "4",  # Session 8c/8d: Reduced from 10 → 4 for ~48 minibatches/update
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
        Write-Host "=== Session 8d: SMOKE TEST (10M, 32 envs, 64 steps) ===" -ForegroundColor Cyan
        Write-Host "Testing improved reward configuration:" -ForegroundColor Yellow
        Write-Host "  - Reachability penalty: -583 per step @ 3.3m (vs -1,458 in 8c-v2)" -ForegroundColor White
        Write-Host "  - Base mobilization: +158 max (vs +30 in 8c-v2)" -ForegroundColor White
        Write-Host "  - Penalty:reward ratio: 3.7:1 (vs 48.6:1 in 8c-v2)" -ForegroundColor White
        
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
        Write-Host "=== Session 8d: EASY Phase (40M timesteps, curriculum start) ===" -ForegroundColor Green
        Write-Host "NOTE: Curriculum is optional for 8d. Testing high parallelism first." -ForegroundColor Yellow
        
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
        Write-Host "=== Session 8d: MEDIUM Phase (60M timesteps, curriculum mid) ===" -ForegroundColor Yellow
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
        Write-Host "=== Session 8d: FULL Phase (100M timesteps, all trajectories) ===" -ForegroundColor Magenta
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
        Write-Host "=== Session 8d: COMPLETE Run (200M timesteps, HIGH PARALLELISM) ===" -ForegroundColor White
        Write-Host ""
        Write-Host "Session 8d Improvements over 8c-v2:" -ForegroundColor Cyan
        Write-Host "  ✅ Reachability penalty: -583 @ 3.3m (60% weaker)" -ForegroundColor Green
        Write-Host "  ✅ Base mobilization: +158 max (5.2× stronger)" -ForegroundColor Green
        Write-Host "  ✅ Linear position gradient: Active (weight=40)" -ForegroundColor Green
        Write-Host "  ✅ Smooth workspace distance: Active" -ForegroundColor Green
        Write-Host "  ✅ Penalty:reward ratio: 3.7:1 (13× better)" -ForegroundColor Green
        Write-Host ""
        Write-Host "Expected Performance:" -ForegroundColor Yellow
        Write-Host "  Target: Position error < 250cm (vs 328cm in 8c-v2)" -ForegroundColor White
        Write-Host "  Target: Samples < 1m > 15% (vs 7% in 8c-v2)" -ForegroundColor White
        Write-Host "  If achieved: Success! Deploy to real robot" -ForegroundColor White
        Write-Host "  If >300cm: Switch to linear penalty or curriculum (Session 8e)" -ForegroundColor White
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
        Write-Host "   - reachability_distance_penalty (should be ~-583 @ 3.3m)" -ForegroundColor White
        Write-Host "   - base_mobilization_reward (should reach +150 range)" -ForegroundColor White
        Write-Host "   - monitoring/unreachable_fraction (track % time outside workspace)" -ForegroundColor White
        Write-Host "2. If smoke test looks good, run: .\scripts\launch_session_8d.ps1 -Phase complete" -ForegroundColor White
        Write-Host "   OR try curriculum: .\scripts\launch_session_8d.ps1 -Phase easy" -ForegroundColor White
    } elseif ($Phase -eq 'easy') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Evaluate at 40M checkpoint" -ForegroundColor White
        Write-Host "2. If position error < 280cm, run: .\scripts\launch_session_8d.ps1 -Phase medium -Checkpoint <path>" -ForegroundColor White
        Write-Host "3. If position error > 300cm, stop and switch to linear penalty for Session 8e" -ForegroundColor White
    } elseif ($Phase -eq 'medium') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Evaluate at 100M checkpoint (40+60)" -ForegroundColor White
        Write-Host "2. If position error improving, run: .\scripts\launch_session_8d.ps1 -Phase full -Checkpoint <path>" -ForegroundColor White
    } elseif ($Phase -eq 'full') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Evaluate final 200M checkpoint (40+60+100)" -ForegroundColor White
        Write-Host "2. Compare Session 8b vs 8c-v2 vs 8d results" -ForegroundColor White
        Write-Host "3. If position error < 250cm: SUCCESS! Proceed to deployment" -ForegroundColor White
        Write-Host "4. If position error > 280cm: Plan Session 8e with linear penalty" -ForegroundColor White
    } elseif ($Phase -eq 'complete') {
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "1. Parse TensorBoard data:" -ForegroundColor White
        Write-Host "   python scripts/parse_tensorboard.py logs/sb3/mobilemmtrackee_v0/<timestamp>" -ForegroundColor Gray
        Write-Host ""
        Write-Host "2. Evaluate at 40M checkpoint:" -ForegroundColor White
        Write-Host "   cd I:\isaaclab" -ForegroundColor Gray
        Write-Host "   .\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate_quantitative.py \" -ForegroundColor Gray
        Write-Host "     --checkpoint <logs/.../rl_model_40000000_steps.zip> \" -ForegroundColor Gray
        Write-Host "     --task MobileMMTrackEE-v0 \" -ForegroundColor Gray
        Write-Host "     --num_episodes 200 \" -ForegroundColor Gray
        Write-Host "     --num_envs 64 \" -ForegroundColor Gray
        Write-Host "     --headless \" -ForegroundColor Gray
        Write-Host "     --trajectory_type multi_recorded \" -ForegroundColor Gray
        Write-Host "     --trajectory_dir C:\Users\yanbo\wSpace\cinebotRL\trajectoryToLearn\world_json \" -ForegroundColor Gray
        Write-Host "     --use_all_trajectories" -ForegroundColor Gray
        Write-Host ""
        Write-Host "3. Decision tree:" -ForegroundColor White
        Write-Host "   Position error < 250cm → SUCCESS! Continue to 200M" -ForegroundColor Green
        Write-Host "   Position error 250-280cm → Promising, continue but monitor" -ForegroundColor Yellow
        Write-Host "   Position error > 300cm → STOP, switch to linear penalty (Session 8e)" -ForegroundColor Red
        Write-Host ""
        Write-Host "4. If training completes successfully (200M):" -ForegroundColor White
        Write-Host "   - Generate comparison plots (8b vs 8c-v2 vs 8d)" -ForegroundColor White
        Write-Host "   - Create analysis report" -ForegroundColor White
        Write-Host "   - If < 250cm: Deploy to real robot!" -ForegroundColor Green
    }
} else {
    Write-Host "`n=== Training Failed ===" -ForegroundColor Red
    Write-Host "Check logs for errors" -ForegroundColor Yellow
    exit $trainExitCode
}
