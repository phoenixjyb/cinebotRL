# Visualize Training Environment
# Launches Isaac Sim in GUI mode to visualize the training environment
# Use this script while training is running in headless mode

param(
    [string]$Task = "MobileMMTrackEE-v0",
    [int]$NumEnvs = 16,  # Use fewer envs for visualization
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Visualize Training Environment

Usage:
  .\scripts\visualize_training.ps1 [-Task <TaskName>] [-NumEnvs <Count>]

Parameters:
  -Task       Task to visualize (default: MobileMMTrackEE-v0)
  -NumEnvs    Number of environments to show (default: 16, keep low for performance)
  -Help       Show this help message

Examples:
  # Visualize with default settings (16 envs)
  .\scripts\visualize_training.ps1

  # Visualize with 4 environments (better for detailed observation)
  .\scripts\visualize_training.ps1 -NumEnvs 4

  # Visualize with 1 environment (single robot view)
  .\scripts\visualize_training.ps1 -NumEnvs 1

Notes:
  - This launches a SEPARATE Isaac Sim instance with GUI enabled
  - Training can continue in parallel in headless mode
  - Use fewer environments for better visualization performance
  - Press ESC in Isaac Sim to close the visualization
  - No checkpoint is loaded - this shows random policy for environment inspection
"@
    exit 0
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Isaac Sim Training Visualization" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Validate Isaac Lab installation
$IsaacLabPath = "I:\isaaclab"
$IsaacLabBat = Join-Path $IsaacLabPath "isaaclab.bat"

if (-not (Test-Path $IsaacLabBat)) {
    Write-Host "ERROR: Isaac Lab not found at $IsaacLabPath" -ForegroundColor Red
    Write-Host "Please check your Isaac Lab installation" -ForegroundColor Yellow
    exit 1
}

# Validate project path
$ProjectRoot = "C:\Users\yanbo\wSpace\cinebotRL"
$TrainScript = Join-Path $ProjectRoot "scripts\reinforcement_learning\sb3\train.py"

if (-not (Test-Path $TrainScript)) {
    Write-Host "ERROR: Training script not found at $TrainScript" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Task:        $Task" -ForegroundColor White
Write-Host "  Environments: $NumEnvs" -ForegroundColor White
Write-Host "  Mode:        GUI (visualization)" -ForegroundColor White
Write-Host ""

Write-Host "Launching Isaac Sim with GUI..." -ForegroundColor Yellow
Write-Host "This will open a new window showing the environment" -ForegroundColor Yellow
Write-Host ""
Write-Host "Controls:" -ForegroundColor Cyan
Write-Host "  - ESC: Close visualization" -ForegroundColor White
Write-Host "  - Mouse: Rotate/pan camera" -ForegroundColor White
Write-Host "  - Scroll: Zoom in/out" -ForegroundColor White
Write-Host ""

# Build command
$Command = "I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py --task $Task --num_envs $NumEnvs --headless false --total_timesteps 1000000"

Write-Host "Command: $Command" -ForegroundColor DarkGray
Write-Host ""

# Set environment variable to disable gymnasium plugin entrypoints
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

# Change to project directory
Set-Location $ProjectRoot

# Launch in GUI mode (omit --headless for GUI)
& I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
    --task $Task `
    --num_envs $NumEnvs `
    --total_timesteps 1000000

Write-Host ""
Write-Host "Visualization closed." -ForegroundColor Green
