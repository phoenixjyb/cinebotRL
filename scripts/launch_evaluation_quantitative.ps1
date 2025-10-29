<#
.SYNOPSIS
    Launch comprehensive quantitative evaluation for trained MobileMMTrackEE policy.

.DESCRIPTION
    This script runs detailed evaluation with extensive logging:
    - Position and orientation tracking errors
    - Joint angles and velocities
    - Base velocities
    - Reward components
    - Success/failure statistics
    
    Results are saved to evaluation_results/ directory with:
    - eval_summary_*.json (statistics)
    - episodes_*.csv (per-episode data)
    - steps_*.csv (per-step data)
    - arrays_*.npz (numpy arrays)

.PARAMETER Checkpoint
    Path to trained model checkpoint (.zip file)

.PARAMETER NumEnvs
    Number of parallel environments (default: 64 for fast evaluation)

.PARAMETER NumEpisodes
    Number of episodes to evaluate (default: 200 for robust statistics)

.PARAMETER OutputDir
    Directory to save evaluation results (default: evaluation_results)

.PARAMETER Headless
    Run in headless mode (no GUI, faster)

.PARAMETER UseAllTrajectories
    Use all 1,038 trajectories from training dataset

.PARAMETER UseChassisOnly
    Use only chassis-requiring trajectories (519 trajectories)

.PARAMETER SaveEveryNSteps
    Save detailed logs every N steps (default: 10)

.EXAMPLE
    # Quick evaluation (20 episodes, 4 envs, with GUI):
    .\scripts\launch_evaluation_quantitative.ps1 -NumEpisodes 20 -NumEnvs 4

.EXAMPLE
    # Full evaluation (200 episodes, 64 envs, headless):
    .\scripts\launch_evaluation_quantitative.ps1 -NumEpisodes 200 -NumEnvs 64 -Headless

.EXAMPLE
    # Evaluate on all 1,038 trajectories:
    .\scripts\launch_evaluation_quantitative.ps1 -UseAllTrajectories -NumEpisodes 500 -Headless

.EXAMPLE
    # Custom checkpoint and output directory:
    .\scripts\launch_evaluation_quantitative.ps1 `
        -Checkpoint "logs/sb3/session_7d/final_model.zip" `
        -OutputDir "eval_session_7d" `
        -Headless
#>

param(
    [Parameter(HelpMessage="Path to trained model checkpoint (.zip)")]
    [string]$Checkpoint = "logs\sb3\MobileMMTrackEE-v0\Oct29_13-24-52_7d_200Mts_multi\final_model.zip",
    
    [Parameter(HelpMessage="Number of parallel environments")]
    [int]$NumEnvs = 64,
    
    [Parameter(HelpMessage="Number of episodes to evaluate")]
    [int]$NumEpisodes = 200,
    
    [Parameter(HelpMessage="Output directory for results")]
    [string]$OutputDir = "evaluation_results",
    
    [Parameter(HelpMessage="Run in headless mode (no GUI)")]
    [switch]$Headless,
    
    [Parameter(HelpMessage="Use all 1,038 trajectories")]
    [switch]$UseAllTrajectories,
    
    [Parameter(HelpMessage="Use only chassis-requiring trajectories")]
    [switch]$UseChassisOnly,
    
    [Parameter(HelpMessage="Save detailed logs every N steps")]
    [int]$SaveEveryNSteps = 10
)

# Configuration
$IsaacLabPath = "I:\isaaclab"
$IsaacLabBat = "$IsaacLabPath\isaaclab.bat"
$EvalScript = "scripts/reinforcement_learning/sb3/evaluate_quantitative.py"

# Colors for output
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# Header
Write-ColorOutput "`n================================================================================" "Cyan"
Write-ColorOutput "   COMPREHENSIVE QUANTITATIVE EVALUATION" "Cyan"
Write-ColorOutput "================================================================================" "Cyan"
Write-Host ""

# Validate checkpoint
if (-not (Test-Path $Checkpoint)) {
    Write-ColorOutput "ERROR: Checkpoint not found: $Checkpoint" "Red"
    exit 1
}

# Display configuration
Write-ColorOutput "Configuration:" "Yellow"
Write-Host "  Checkpoint:          $Checkpoint"
Write-Host "  Num environments:    $NumEnvs"
Write-Host "  Num episodes:        $NumEpisodes"
Write-Host "  Output directory:    $OutputDir"
Write-Host "  Headless:            $Headless"
Write-Host "  Use all traj:        $UseAllTrajectories"
Write-Host "  Use chassis only:    $UseChassisOnly"
Write-Host "  Save every N steps:  $SaveEveryNSteps"
Write-Host ""

# Estimate runtime
$EstimatedMinutes = [math]::Ceiling(($NumEpisodes * 500) / ($NumEnvs * 3600.0) * 60)
Write-ColorOutput "Estimated time: ~$EstimatedMinutes minutes" "Yellow"
Write-Host ""

# Build arguments array
$CmdArgs = @(
    "--checkpoint", $Checkpoint,
    "--num_envs", $NumEnvs,
    "--num_episodes", $NumEpisodes,
    "--output_dir", $OutputDir,
    "--save_every_n_steps", $SaveEveryNSteps,
    "--deterministic"
)

if ($Headless) {
    $CmdArgs += "--headless"
}

if ($UseAllTrajectories) {
    $CmdArgs += "--use_all_trajectories"
}

if ($UseChassisOnly) {
    $CmdArgs += "--use_chassis_only"
}

# Display command
Write-ColorOutput "Launching Isaac Lab..." "Green"
Write-Host "  Script: $EvalScript"
Write-Host "  Args: $($CmdArgs -join ' ')"
Write-Host ""

# Set environment variables (same as training)
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

Write-ColorOutput "Starting evaluation..." "Green"
Write-Host ""

# Execute with arguments properly passed
& $IsaacLabBat "-p" $EvalScript $CmdArgs

$ExitCode = $LASTEXITCODE

# Summary
Write-Host ""
Write-ColorOutput "================================================================================" "Cyan"
if ($ExitCode -eq 0) {
    Write-ColorOutput "   EVALUATION COMPLETE!" "Green"
    Write-Host ""
    Write-Host "Results saved to: $OutputDir"
    Write-Host ""
    Write-ColorOutput "Next steps:" "Yellow"
    Write-Host "  1. Analyze results:"
    Write-Host "     python scripts/reinforcement_learning/sb3/visualize_eval_results.py --input_dir $OutputDir"
    Write-Host ""
    Write-Host "  2. View plots:"
    Write-Host "     explorer evaluation_plots\"
    Write-Host ""
    Write-Host "  3. Read report:"
    Write-Host "     notepad evaluation_plots\evaluation_report.txt"
} else {
    Write-ColorOutput "   EVALUATION FAILED (Exit code: $ExitCode)" "Red"
}
Write-ColorOutput "================================================================================" "Cyan"
Write-Host ""
