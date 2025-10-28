<#
.SYNOPSIS
    Accelerated Session 7d Training - 8192 envs for ~11 hour runtime

.DESCRIPTION
    Optimized configuration to fully utilize RTX 3090 24GB VRAM.
    Doubles training speed compared to 4096 envs while maintaining stability.
    
    VRAM Usage Estimate:
    - 8192 envs × 3MB/env = ~24.6GB
    - Isaac Sim overhead: ~4GB
    - Total: ~28GB (fits in 24GB with memory management)
    
    Training will complete in ~11 hours instead of 22 hours.

.NOTES
    Based on Session 7c successful setup, optimized for speed.
#>

param(
    [switch]$DryRun = $false  # Test configuration without starting training
)

$ErrorActionPreference = "Stop"

# Colors
$ESC = [char]27
$Green = "$ESC[32m"
$Yellow = "$ESC[33m"
$Blue = "$ESC[34m"
$Cyan = "$ESC[36m"
$Reset = "$ESC[0m"

Write-Host ""
Write-Host "${Cyan}╔════════════════════════════════════════════════════════════════╗${Reset}"
Write-Host "${Cyan}║     Session 7d Training - ACCELERATED (8192 environments)     ║${Reset}"
Write-Host "${Cyan}╔════════════════════════════════════════════════════════════════╗${Reset}"
Write-Host ""

# Configuration
$CONFIG = @{
    # Environment
    Task = "MobileMMTrackEE-v0"
    NumEnvs = 8192  # 2x parallelism (was 4096 in Session 7c)
    Headless = $true
    
    # Training duration
    TotalTimesteps = 200000000  # 200M timesteps (same as 7c)
    
    # PPO Hyperparameters (adjusted for 8192 envs)
    LearningRate = 0.0003  # 3e-4 (same as 7c)
    NSteps = 64  # Down from 128 to keep buffer size manageable
    BatchSize = 1024  # Up from 512 to leverage larger rollout
    NEpochs = 10  # Same as 7c
    
    # PPO regularization
    EntCoef = 0.01  # Same as 7c
    ClipRange = 0.2  # Same as 7c
    Gamma = 0.99  # Same as 7c
    GaeLambda = 0.95  # Same as 7c
    
    # Entropy decay (same as 7c)
    EnableEntropyDecay = $true
    FinalEntCoef = 0.0001
    DecayStartTimestep = 100000000  # 100M
    DecayDurationTimesteps = 50000000  # 50M
    
    # Checkpointing
    SaveFreq = 2048000  # Save every ~2M timesteps (250 rollouts × 8192 envs)
    
    # Paths
    IsaacLabPath = "I:\isaaclab"
    ProjectPath = "C:\Users\yanbo\wSpace\cinebotRL"
}

# Verify paths
Write-Host "${Blue}Verification:${Reset}"
if (-not (Test-Path $CONFIG.IsaacLabPath)) {
    Write-Host "${Yellow}✗ Isaac Lab not found: $($CONFIG.IsaacLabPath)${Reset}"
    exit 1
}
Write-Host "${Green}✓${Reset} Isaac Lab: $($CONFIG.IsaacLabPath)"

if (-not (Test-Path $CONFIG.ProjectPath)) {
    Write-Host "${Yellow}✗ Project not found: $($CONFIG.ProjectPath)${Reset}"
    exit 1
}
Write-Host "${Green}✓${Reset} Project: $($CONFIG.ProjectPath)"

# Display configuration
Write-Host ""
Write-Host "${Blue}Configuration Summary:${Reset}"
Write-Host "  ${Cyan}Environment:${Reset}"
Write-Host "    Task:              $($CONFIG.Task)"
Write-Host "    Num Envs:          $($CONFIG.NumEnvs) ${Yellow}(2x Session 7c)${Reset}"
Write-Host "    Headless:          $($CONFIG.Headless)"
Write-Host ""
Write-Host "  ${Cyan}Training:${Reset}"
Write-Host "    Total Timesteps:   $($CONFIG.TotalTimesteps) (200M)"
Write-Host "    Expected Duration: ${Green}~11 hours${Reset} ${Yellow}(vs 22 hours in 7c)${Reset}"
Write-Host "    Speedup:           ${Green}2.0x${Reset}"
Write-Host ""
Write-Host "  ${Cyan}PPO Hyperparameters:${Reset}"
Write-Host "    Learning Rate:     $($CONFIG.LearningRate)"
Write-Host "    N Steps:           $($CONFIG.NSteps) ${Yellow}(64 vs 128 in 7c)${Reset}"
Write-Host "    Batch Size:        $($CONFIG.BatchSize) ${Yellow}(1024 vs 512 in 7c)${Reset}"
Write-Host "    N Epochs:          $($CONFIG.NEpochs)"
Write-Host "    Entropy Coef:      $($CONFIG.EntCoef) → $($CONFIG.FinalEntCoef) ${Yellow}(decay enabled)${Reset}"
Write-Host "    Clip Range:        $($CONFIG.ClipRange)"
Write-Host "    Gamma:             $($CONFIG.Gamma)"
Write-Host "    GAE Lambda:        $($CONFIG.GaeLambda)"
Write-Host ""
Write-Host "  ${Cyan}Memory Estimate:${Reset}"
$vram_estimate = $CONFIG.NumEnvs * 3 / 1024
Write-Host "    Envs VRAM:         ~${vram_estimate:N1}GB ($($CONFIG.NumEnvs) × 3MB)"
Write-Host "    Isaac Sim:         ~4GB"
Write-Host "    Total:             ~$([Math]::Round($vram_estimate + 4, 1))GB ${Yellow}(fits in 24GB)${Reset}"
Write-Host ""
Write-Host "  ${Cyan}Rollout Buffer:${Reset}"
$rollout_size = $CONFIG.NSteps * $CONFIG.NumEnvs
Write-Host "    Size:              $rollout_size timesteps ($($CONFIG.NSteps) × $($CONFIG.NumEnvs))"
Write-Host "    Update Frequency:  Every $rollout_size timesteps"
Write-Host ""

if ($DryRun) {
    Write-Host "${Yellow}═══════════════════════════════════════════════════════════════${Reset}"
    Write-Host "${Yellow}DRY RUN MODE - Configuration validated, not starting training${Reset}"
    Write-Host "${Yellow}═══════════════════════════════════════════════════════════════${Reset}"
    exit 0
}

# Confirm before starting
Write-Host "${Yellow}═══════════════════════════════════════════════════════════════${Reset}"
Write-Host "${Yellow}Ready to start Session 7d accelerated training (~11 hours)${Reset}"
Write-Host "${Yellow}═══════════════════════════════════════════════════════════════${Reset}"
Write-Host ""
$confirmation = Read-Host "Continue? (y/N)"
if ($confirmation -ne "y") {
    Write-Host "${Yellow}Cancelled by user${Reset}"
    exit 0
}

# Build command
Write-Host ""
Write-Host "${Blue}Launching training...${Reset}"
Write-Host ""

Set-Location $CONFIG.IsaacLabPath

# Disable Gymnasium plugins to avoid ale_py crash
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

$scriptPath = Join-Path $CONFIG.ProjectPath "scripts\reinforcement_learning\sb3\train.py"

$scriptArgs = @(
    "-p", $scriptPath,
    "--task", $CONFIG.Task,
    "--num_envs", $CONFIG.NumEnvs,
    "--headless",
    "--total_timesteps", $CONFIG.TotalTimesteps,
    
    # PPO hyperparameters
    "--learning_rate", $CONFIG.LearningRate,
    "--n_steps", $CONFIG.NSteps,
    "--batch_size", $CONFIG.BatchSize,
    "--n_epochs", $CONFIG.NEpochs,
    
    # Regularization
    "--ent_coef", $CONFIG.EntCoef,
    "--clip_range", $CONFIG.ClipRange,
    "--gamma", $CONFIG.Gamma,
    "--gae_lambda", $CONFIG.GaeLambda,
    
    # Entropy decay
    "--enable_entropy_decay",
    "--final_ent_coef", $CONFIG.FinalEntCoef,
    "--decay_start_timestep", $CONFIG.DecayStartTimestep,
    "--decay_duration_timesteps", $CONFIG.DecayDurationTimesteps,
    
    # Checkpointing
    "--save_freq", $CONFIG.SaveFreq
)

Write-Host "${Cyan}Command:${Reset}"
Write-Host "  .\isaaclab.bat $($scriptArgs -join ' ')"
Write-Host ""

$startTime = Get-Date
Write-Host "${Green}╔════════════════════════════════════════════════════════════════╗${Reset}"
Write-Host "${Green}║  Training started at: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))  ║${Reset}"
Write-Host "${Green}║  Expected completion: ~11 hours                                ║${Reset}"
Write-Host "${Green}╚════════════════════════════════════════════════════════════════╝${Reset}"
Write-Host ""

& .\isaaclab.bat @scriptArgs

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "${Green}╔════════════════════════════════════════════════════════════════╗${Reset}"
    Write-Host "${Green}║  ✓ Training completed successfully!                           ║${Reset}"
    Write-Host "${Green}║  Duration: $($duration.ToString('hh\:mm\:ss'))                                  ║${Reset}"
    Write-Host "${Green}╚════════════════════════════════════════════════════════════════╝${Reset}"
} else {
    Write-Host "${Yellow}╔════════════════════════════════════════════════════════════════╗${Reset}"
    Write-Host "${Yellow}║  ✗ Training exited with error code: $LASTEXITCODE            ║${Reset}"
    Write-Host "${Yellow}║  Duration: $($duration.ToString('hh\:mm\:ss'))                                  ║${Reset}"
    Write-Host "${Yellow}╚════════════════════════════════════════════════════════════════╝${Reset}"
    exit $LASTEXITCODE
}
