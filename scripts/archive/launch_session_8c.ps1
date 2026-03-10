# Session 8c Training Launcher
# Based on user's detailed Session 8c plan
# 
# Key Changes from Session 8b:
# 1. Curriculum learning: 40M easy, 60M medium, 100M full (200M total)
# 2. Lower env count: 128-192 envs (vs 20,480 in 8b)
# 3. Lower n_steps: 96 (vs 128 in 8b) for 12-18k samples per update
# 4. Batch size: 2048 (vs 4096 in 8b)
# 5. n_epochs: 4 (default, ~48 minibatches/update)
# 6. Delayed entropy decay: start 0.001, decay after 120M, end 1e-4 at 200M
# 7. Tighter KL: warmup=0.15, main=0.1, finetune=0.05, target=0.5
# 8. Advantage normalization + value clipping (clip_range_vf=0.3)
# 9. Checkpoint every 2M steps
# 10. Smoke test option: 10M with 32 envs, 64 steps

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('smoke', 'easy', 'medium', 'full', 'complete')]
    [string]$Phase = 'smoke',
    
    [Parameter(Mandatory=$false)]
    [string]$Checkpoint = $null,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoHeadless
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
    "--n_epochs", "4",  # Session 8c: Reduced from 10 → 4 for ~48 minibatches/update with batch_size=2048
    "--enable_entropy_decay",
    "--final_ent_coef", "1e-4",
    "--enable_kl_schedule",
    "--clip_range_vf", "0.3",  # Session 8c: Stabilize critic when reachability term spikes
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
        Write-Host "=== Session 8c: SMOKE TEST (10M, 32 envs, 64 steps) ===" -ForegroundColor Cyan
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
            "--target_kl", "0.5",
            "--save_freq", "2000000"  # Every 2M steps
        )
    }
    
    'easy' {
        Write-Host "=== Session 8c: EASY Phase (40M timesteps, curriculum start) ===" -ForegroundColor Green
        # TODO: Implement trajectory filtering for easy trajectories
        # For now, use all trajectories but with gentle exploration
        $PhaseArgs = @(
            "--num_envs", "128",
            "--n_steps", "96",
            "--batch_size", "2048",
            "--total_timesteps", "40000000",
            "--ent_coef", "0.001",  # High exploration initially
            "--decay_start_timestep", "120000000",  # No decay in easy phase
            "--decay_duration_timesteps", "80000000",
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",  # FIX (8c-v2): Tightened from 0.5 (consistent with complete phase)
            "--save_freq", "2000000"
        )
    }
    
    'medium' {
        Write-Host "=== Session 8c: MEDIUM Phase (60M timesteps, curriculum mid) ===" -ForegroundColor Yellow
        if (-not $Checkpoint) {
            Write-Host "ERROR: -Checkpoint required for medium phase (provide easy phase final model)" -ForegroundColor Red
            exit 1
        }
        $PhaseArgs = @(
            "--num_envs", "160",
            "--n_steps", "96",
            "--batch_size", "2048",
            "--total_timesteps", "60000000",
            "--ent_coef", "0.001",  # Still exploring
            "--decay_start_timestep", "120000000",  # No decay yet
            "--decay_duration_timesteps", "80000000",
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",  # FIX (8c-v2): Tightened from 0.5 (consistent with complete phase)
            "--save_freq", "2000000",
            "--checkpoint", $Checkpoint
        )
    }
    
    'full' {
        Write-Host "=== Session 8c: FULL Phase (100M timesteps, all trajectories) ===" -ForegroundColor Magenta
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
            "--decay_duration_timesteps", "80000000",  # Decay to 1e-4 by end of phase
            "--kl_warmup", "0.15",
            "--kl_main", "0.1",
            "--kl_finetune", "0.05",
            "--target_kl", "0.03",  # FIX (8c-v2): Tightened from 0.5 (consistent with complete phase)
            "--save_freq", "2000000",
            "--checkpoint", $Checkpoint
        )
    }
    
    'complete' {
        Write-Host "=== Session 8c: COMPLETE Run (200M timesteps, single run, HIGH PARALLELISM) ===" -ForegroundColor White
        Write-Host "WARNING: This bypasses curriculum learning. Use only if smoke test validates config." -ForegroundColor Yellow
        
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
            "--target_kl", "0.03",  # FIX (8c-v2): Tightened from 0.5 (was 30× actual KL of 0.015)
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

# Change to Isaac Lab directory and run
Set-Location $IsaacLabPath
& .\isaaclab.bat -p $TrainScript @CommonArgs @PhaseArgs

# Return to project directory
Set-Location $ProjectPath

Write-Host "`n=== Training Complete ===" -ForegroundColor Green
Write-Host "Check logs in: logs/sb3/mobilemmtrackee_v0/<timestamp>" -ForegroundColor Yellow
Write-Host "Checkpoints saved every 2M steps" -ForegroundColor Yellow

if ($Phase -eq 'smoke') {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Check TensorBoard for reachability reward (should go positive by end)" -ForegroundColor White
    Write-Host "2. If smoke test looks good, run: .\scripts\launch_session_8c.ps1 -Phase easy" -ForegroundColor White
} elseif ($Phase -eq 'easy') {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Evaluate at 40M checkpoint" -ForegroundColor White
    Write-Host "2. If P95 error improving, run: .\scripts\launch_session_8c.ps1 -Phase medium -Checkpoint <path_to_final_model>" -ForegroundColor White
} elseif ($Phase -eq 'medium') {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Evaluate at 100M checkpoint (40+60)" -ForegroundColor White
    Write-Host "2. If metrics good, run: .\scripts\launch_session_8c.ps1 -Phase full -Checkpoint <path_to_final_model>" -ForegroundColor White
} elseif ($Phase -eq 'full') {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Evaluate final 200M checkpoint (40+60+100)" -ForegroundColor White
    Write-Host "2. Compare Session 8b vs 8c results" -ForegroundColor White
    Write-Host "3. If metrics meet targets, proceed to deployment preparation" -ForegroundColor White
}
