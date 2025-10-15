# Visualize Trained Policy
# Loads a checkpoint and visualizes the trained policy in action

param(
    [string]$Task = "MobileMMTrackEE-v0",
    [int]$NumEnvs = 1,  # Single env for best observation
    [string]$Checkpoint = "",  # Path to checkpoint file
    [switch]$Latest,  # Use latest checkpoint
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Visualize Trained Policy

Usage:
  .\scripts\visualize_policy.ps1 [-Task <TaskName>] [-Checkpoint <Path>] [-Latest]

Parameters:
  -Task         Task to visualize (default: MobileMMTrackEE-v0)
  -NumEnvs      Number of environments (default: 1)
  -Checkpoint   Path to checkpoint file (.zip)
  -Latest       Automatically use the latest checkpoint
  -Help         Show this help message

Examples:
  # Show latest checkpoint available
  .\scripts\visualize_policy.ps1 -Latest

  # Load specific checkpoint
  .\scripts\visualize_policy.ps1 -Checkpoint "I:\isaaclab\logs\sb3\MobileMMTrackEE-v0\2025-10-15_14-30-45\checkpoints\model_1000000_steps.zip"

  # Visualize with multiple environments
  .\scripts\visualize_policy.ps1 -Latest -NumEnvs 4

Notes:
  - This requires a trained checkpoint file
  - Training checkpoints are saved in I:\isaaclab\logs\sb3\<Task>\<timestamp>\checkpoints\
  - Use -Latest to automatically find the most recent checkpoint
  - Press ESC in Isaac Sim to close
"@
    exit 0
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Policy Visualization" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Find checkpoint
$LogsPath = "I:\isaaclab\logs\sb3\$Task"

if ($Latest) {
    Write-Host "Finding latest checkpoint..." -ForegroundColor Yellow
    
    if (-not (Test-Path $LogsPath)) {
        Write-Host "ERROR: No logs found for task $Task" -ForegroundColor Red
        Write-Host "Path checked: $LogsPath" -ForegroundColor DarkGray
        exit 1
    }
    
    # Find most recent run directory
    $RunDirs = Get-ChildItem -Path $LogsPath -Directory | Sort-Object LastWriteTime -Descending
    
    if ($RunDirs.Count -eq 0) {
        Write-Host "ERROR: No training runs found" -ForegroundColor Red
        exit 1
    }
    
    $LatestRun = $RunDirs[0]
    $CheckpointDir = Join-Path $LatestRun.FullName "checkpoints"
    
    if (-not (Test-Path $CheckpointDir)) {
        Write-Host "ERROR: No checkpoints directory found in latest run" -ForegroundColor Red
        Write-Host "Run directory: $($LatestRun.FullName)" -ForegroundColor DarkGray
        exit 1
    }
    
    # Find latest checkpoint file
    $CheckpointFiles = Get-ChildItem -Path $CheckpointDir -Filter "*.zip" | Sort-Object LastWriteTime -Descending
    
    if ($CheckpointFiles.Count -eq 0) {
        Write-Host "ERROR: No checkpoint files found" -ForegroundColor Red
        Write-Host "Checkpoint directory: $CheckpointDir" -ForegroundColor DarkGray
        Write-Host "Training may still be in progress. Wait for first checkpoint to be saved." -ForegroundColor Yellow
        exit 1
    }
    
    $Checkpoint = $CheckpointFiles[0].FullName
    Write-Host "Found checkpoint: $($CheckpointFiles[0].Name)" -ForegroundColor Green
    Write-Host "From run: $($LatestRun.Name)" -ForegroundColor Green
}

if (-not $Checkpoint -or -not (Test-Path $Checkpoint)) {
    Write-Host "ERROR: No valid checkpoint specified" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available options:" -ForegroundColor Yellow
    Write-Host "  1. Use -Latest to automatically find the most recent checkpoint" -ForegroundColor White
    Write-Host "  2. Specify -Checkpoint with full path to a .zip checkpoint file" -ForegroundColor White
    Write-Host ""
    Write-Host "Example: .\scripts\visualize_policy.ps1 -Latest" -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Task:        $Task" -ForegroundColor White
Write-Host "  Checkpoint:  $Checkpoint" -ForegroundColor White
Write-Host "  Environments: $NumEnvs" -ForegroundColor White
Write-Host ""

Write-Host "NOTE: Full checkpoint loading requires extending train.py with test mode" -ForegroundColor Yellow
Write-Host "For now, this will launch the environment in GUI mode." -ForegroundColor Yellow
Write-Host "To implement policy loading, add --test and --checkpoint flags to train.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to continue with environment visualization..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

# Launch visualization
& C:\Users\yanbo\wSpace\cinebotRL\scripts\visualize_training.ps1 -Task $Task -NumEnvs $NumEnvs
