# Quick Environment Inspector
# Opens Isaac Sim GUI to inspect the robot and environment setup
# No training, just visualization of the scene

param(
    [string]$Task = "MobileMMTrackEE-v0",
    [int]$NumEnvs = 1,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Quick Environment Inspector

Usage:
  .\scripts\inspect_environment.ps1 [-Task <TaskName>] [-NumEnvs <Count>]

Parameters:
  -Task       Task environment to inspect (default: MobileMMTrackEE-v0)
  -NumEnvs    Number of robot instances to show (default: 1)
  -Help       Show this help message

Examples:
  # Inspect single robot
  .\scripts\inspect_environment.ps1

  # Inspect 4 robots in grid layout
  .\scripts\inspect_environment.ps1 -NumEnvs 4

Purpose:
  - Verify robot USD asset loaded correctly
  - Check joint ranges and articulation
  - Inspect collision meshes
  - Validate scene setup
  - No training - just environment inspection

Controls:
  - Mouse: Rotate/pan/zoom camera
  - Click robot: Select and inspect properties in GUI
  - ESC: Close window
"@
    exit 0
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Environment Inspector" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = "C:\Users\yanbo\wSpace\cinebotRL"

Write-Host "Launching Isaac Sim GUI..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Task: $Task" -ForegroundColor Green
Write-Host "Instances: $NumEnvs robot(s)" -ForegroundColor Green
Write-Host ""
Write-Host "What you'll see:" -ForegroundColor Cyan
Write-Host "  - Robot USD asset in the scene" -ForegroundColor White
Write-Host "  - End-effector target trajectory (if available)" -ForegroundColor White
Write-Host "  - Ground plane and lighting" -ForegroundColor White
Write-Host ""
Write-Host "TIP: The robot will perform random actions (no trained policy)" -ForegroundColor Yellow
Write-Host "     This is normal for environment inspection" -ForegroundColor Yellow
Write-Host ""

# Set environment
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"
Set-Location $ProjectRoot

# Launch with minimal timesteps, GUI mode (omit --headless for GUI)
Write-Host "Starting Isaac Sim (this may take 30-60 seconds)..." -ForegroundColor Yellow
Write-Host ""

& I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
    --task $Task `
    --num_envs $NumEnvs `
    --total_timesteps 10000

Write-Host ""
Write-Host "Inspector closed." -ForegroundColor Green
