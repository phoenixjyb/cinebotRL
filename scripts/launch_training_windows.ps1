<#
.SYNOPSIS
    Launch RL training on Windows with Isaac Lab

.DESCRIPTION
    This script activates the Isaac Lab environment and launches training
    for the mobile manipulator task. It handles environment setup and
    provides convenient parameter passing.

.PARAMETER Task
    Task ID to train (default: RecomoProto2TrackEE-v0)

.PARAMETER NumEnvs
    Number of parallel environments (default: 1024)

.PARAMETER Headless
    Run in headless mode without GUI (recommended for training)

.PARAMETER TotalTimesteps
    Total number of training timesteps (default: 1000000)

.PARAMETER Checkpoint
    Path to checkpoint to resume from (optional)

.PARAMETER Test
    Run environment test instead of training

.EXAMPLE
    .\scripts\launch_training_windows.ps1 -Test
    # Quick test with 1 environment

.EXAMPLE
    .\scripts\launch_training_windows.ps1 -Headless -NumEnvs 512
    # Train with 512 parallel environments in headless mode

.EXAMPLE
    .\scripts\launch_training_windows.ps1 -Headless -NumEnvs 1024 -TotalTimesteps 5000000
    # Full training run
#>

param(
    [string]$Task = "RecomoProto2TrackEE-v0",
    [int]$NumEnvs = 1024,
    [switch]$Headless,
    [int]$TotalTimesteps = 1000000,
    [string]$Checkpoint = "",
    [switch]$Test
)

# Colors for output
$ESC = [char]27
$Green = "$ESC[32m"
$Yellow = "$ESC[33m"
$Red = "$ESC[31m"
$Blue = "$ESC[34m"
$Reset = "$ESC[0m"

Write-Host ""
Write-Host "${Blue}========================================${Reset}"
Write-Host "${Blue}  Cinebot RL - Windows Training Launcher${Reset}"
Write-Host "${Blue}========================================${Reset}"
Write-Host ""

# Resolve Isaac Lab path: prefer ISAAC_LAB_ROOT env var, fall back to default
if ($env:ISAAC_LAB_ROOT -and (Test-Path $env:ISAAC_LAB_ROOT)) {
    $IsaacLabPath = $env:ISAAC_LAB_ROOT
} else {
    $IsaacLabPath = "I:\isaaclab"
}

if (-not (Test-Path $IsaacLabPath)) {
    Write-Host "${Red}✗ Isaac Lab not found at: $IsaacLabPath${Reset}"
    Write-Host "  Set the ISAAC_LAB_ROOT environment variable to your Isaac Lab installation path."
    Write-Host "  e.g.: `$env:ISAAC_LAB_ROOT = 'D:\isaaclab'"
    exit 1
}
Write-Host "${Green}✓${Reset} Isaac Lab found: $IsaacLabPath"

# Resolve project path: use script location (works regardless of where you call from)
$ProjectPath = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectPath "pyproject.toml"))) {
    # Fallback: try common locations
    foreach ($candidate in @("C:\Users\yanbo\wSpace\cinebotRL", "I:\wSpace\cinebotRL")) {
        if (Test-Path (Join-Path $candidate "pyproject.toml")) {
            $ProjectPath = $candidate
            break
        }
    }
}
if (-not (Test-Path (Join-Path $ProjectPath "pyproject.toml"))) {
    Write-Host "${Red}✗ Project root not found (no pyproject.toml found near script)${Reset}"
    exit 1
}
Write-Host "${Green}✓${Reset} Project found: $ProjectPath"

# Check GPU
Write-Host ""
Write-Host "${Blue}GPU Configuration:${Reset}"
try {
    $gpuInfo = nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
    $gpuLines = $gpuInfo -split "`n"
    foreach ($line in $gpuLines) {
        if ($line -match "RTX|GeForce|Ada|Blackwell") {
            Write-Host "${Green}  ✓ $line${Reset}"
        } elseif ($line.Trim() -ne "") {
            Write-Host "    $line"
        }
    }
} catch {
    Write-Host "${Yellow}  ⚠️  Could not detect GPU (nvidia-smi not found)${Reset}"
}

Write-Host ""
Write-Host "${Blue}Configuration:${Reset}"
Write-Host "  Task:            $Task"
Write-Host "  Num Envs:        $NumEnvs"
Write-Host "  Headless:        $Headless"
if (-not $Test) {
    Write-Host "  Total Steps:     $TotalTimesteps"
    if ($Checkpoint) {
        Write-Host "  Checkpoint:      $Checkpoint"
    }
}

Write-Host ""
Write-Host "${Blue}Launching...${Reset}"
Write-Host ""

# Change to Isaac Lab directory
Set-Location $IsaacLabPath

# Set environment variable to disable problematic Gymnasium plugins (Atari envs)
# This prevents the ale_py crash on Windows
$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"
Write-Host "${Yellow}Note: Disabled Gymnasium plugin entrypoints to avoid ale_py crash${Reset}"
Write-Host ""

# Build command
if ($Test) {
    # Run test script
    $scriptPath = Join-Path $ProjectPath "scripts\test_mobile_mm_env.py"
    $scriptArgs = @(
        "-p", $scriptPath,
        "--task", $Task,
        "--num_envs", "1",
        "--steps", "5"
    )
    if ($Headless) {
        $scriptArgs += "--headless"
    }
} else {
    # Run training script
    $scriptPath = Join-Path $ProjectPath "scripts\reinforcement_learning\sb3\train.py"
    $scriptArgs = @(
        "-p", $scriptPath,
        "--task", $Task,
        "--num_envs", $NumEnvs,
        "--total_timesteps", $TotalTimesteps
    )
    if ($Headless) {
        $scriptArgs += "--headless"
    }
    if ($Checkpoint) {
        $scriptArgs += @("--checkpoint", $Checkpoint)
    }
}

# Launch with Isaac Lab's batch file
Write-Host "${Yellow}Executing: .\isaaclab.bat $scriptArgs${Reset}"
Write-Host ""

& .\isaaclab.bat @scriptArgs

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "${Green}========================================${Reset}"
    Write-Host "${Green}  ✓ Completed successfully!${Reset}"
    Write-Host "${Green}========================================${Reset}"
} else {
    Write-Host ""
    Write-Host "${Red}========================================${Reset}"
    Write-Host "${Red}  ✗ Exited with error code: $LASTEXITCODE${Reset}"
    Write-Host "${Red}========================================${Reset}"
    exit $LASTEXITCODE
}
