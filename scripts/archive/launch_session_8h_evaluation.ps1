<#
.SYNOPSIS
    Launch Session 8h checkpoint evaluation with proper environment setup.

.DESCRIPTION
    Evaluates Session 8h checkpoints at key milestones (20M, 40M, 60M, 80M, 100M)
    and compares them against Session 8f/8g baselines.

.PARAMETER Checkpoints
    Which checkpoints to evaluate (e.g., "20M", "40M", "100M", or "final")
    Default: 20M 40M 60M 80M 100M

.PARAMETER NumEnvs
    Number of parallel environments (default: 64)

.PARAMETER NumEpisodes
    Number of episodes per checkpoint (recommend 200+ for robust stats)
    Default: 200

.PARAMETER Headless
    Run in headless mode (faster, no rendering)

.PARAMETER Quick
    Quick evaluation mode: fewer episodes (50), fewer checkpoints (20M, 40M, 100M)

.EXAMPLE
    .\scripts\launch_session_8h_evaluation.ps1 -Headless
    # Evaluate all checkpoints (20M, 40M, 60M, 80M, 100M) with 200 episodes each

.EXAMPLE
    .\scripts\launch_session_8h_evaluation.ps1 -Checkpoints "40M","100M" -NumEpisodes 100 -Headless
    # Evaluate only 40M and 100M checkpoints with 100 episodes

.EXAMPLE
    .\scripts\launch_session_8h_evaluation.ps1 -Quick
    # Quick test: 20M, 40M, 100M with 50 episodes (headless)
#>

param(
    [string[]]$Checkpoints = @("20M", "40M", "60M", "80M", "100M"),
    [int]$NumEnvs = 64,
    [int]$NumEpisodes = 200,
    [switch]$Headless,
    [switch]$Quick
)

# Colors
$ColorInfo = "Cyan"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"

Write-Host "========================================" -ForegroundColor $ColorInfo
Write-Host "Session 8h Checkpoint Evaluation" -ForegroundColor $ColorInfo
Write-Host "========================================" -ForegroundColor $ColorInfo

# Quick mode overrides
if ($Quick) {
    Write-Host "Quick evaluation mode enabled" -ForegroundColor $ColorWarning
    $Checkpoints = @("20M", "40M", "100M")
    $NumEpisodes = 50
    $NumEnvs = 16
    $Headless = $true
}

# Session 8h directory (latest training)
$Session8hDir = "logs\sb3\mobilemmtrackee_v0\20251103_235918"

# Check if directory exists
if (-not (Test-Path $Session8hDir)) {
    Write-Host "❌ Session 8h directory not found: $Session8hDir" -ForegroundColor $ColorError
    Write-Host "Please verify the path or update the script." -ForegroundColor $ColorError
    exit 1
}

Write-Host "Session 8h directory: $Session8hDir" -ForegroundColor $ColorInfo
Write-Host "Checkpoints to evaluate: $($Checkpoints -join ', ')" -ForegroundColor $ColorInfo
Write-Host "Episodes per checkpoint: $NumEpisodes" -ForegroundColor $ColorInfo
Write-Host "Parallel environments: $NumEnvs" -ForegroundColor $ColorInfo
Write-Host "Headless mode: $Headless" -ForegroundColor $ColorInfo
Write-Host ""

# Isaac Lab path
$IsaacLabPath = "I:\isaaclab"
$IsaacLabBat = Join-Path $IsaacLabPath "isaaclab.bat"

if (-not (Test-Path $IsaacLabBat)) {
    Write-Host "❌ Isaac Lab not found: $IsaacLabBat" -ForegroundColor $ColorError
    exit 1
}

# Script path
$EvalScript = "scripts\reinforcement_learning\sb3\evaluate_session_8h_simple.py"

if (-not (Test-Path $EvalScript)) {
    Write-Host "❌ Evaluation script not found: $EvalScript" -ForegroundColor $ColorError
    exit 1
}

# Disable Gymnasium plugin entrypoints (Windows compatibility)
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

# Build command - note: This script runs evaluate_quantitative.py multiple times
# so it calls python directly, not through isaaclab.bat (the script handles that internally)
$CheckpointsArg = $Checkpoints -join " "

$Command = "python `"$EvalScript`" " +
    "--session_8h_dir `"$Session8hDir`" " +
    "--checkpoints $CheckpointsArg " +
    "--num_envs $NumEnvs " +
    "--num_episodes $NumEpisodes"

if ($Headless) {
    $Command += " --headless"
}

Write-Host "Launching evaluation..." -ForegroundColor $ColorSuccess
Write-Host "Command: $Command" -ForegroundColor $ColorInfo
Write-Host ""

# Execute
try {
    Invoke-Expression $Command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor $ColorSuccess
        Write-Host "✅ Evaluation completed successfully!" -ForegroundColor $ColorSuccess
        Write-Host "========================================" -ForegroundColor $ColorSuccess
        Write-Host "Results saved to: evaluation_results\session_8h_comparison\" -ForegroundColor $ColorInfo
    } else {
        Write-Host ""
        Write-Host "❌ Evaluation failed with exit code: $LASTEXITCODE" -ForegroundColor $ColorError
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error during evaluation: $_" -ForegroundColor $ColorError
    exit 1
}
