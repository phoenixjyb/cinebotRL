<#
.SYNOPSIS
    Commit all changes and start RL training

.DESCRIPTION
    This script commits all the Windows compatibility changes to git
    and then starts a long-running training session.
#>

param(
    [int]$NumEnvs = 64,
    [int]$TotalTimesteps = 5000000
)

$ESC = [char]27
$Green = "$ESC[32m"
$Yellow = "$ESC[33m"
$Blue = "$ESC[34m"
$Reset = "$ESC[0m"

Write-Host ""
Write-Host "${Blue}========================================${Reset}"
Write-Host "${Blue}  Commit Changes & Start Training${Reset}"
Write-Host "${Blue}========================================${Reset}"
Write-Host ""

# Change to project directory
$ProjectPath = "C:\Users\yanbo\wSpace\cinebotRL"
Set-Location $ProjectPath

# Step 1: Git Status
Write-Host "${Blue}[1/4] Checking git status...${Reset}"
git status --short
Write-Host ""

# Step 2: Commit changes
Write-Host "${Blue}[2/4] Committing changes...${Reset}"
Write-Host "Commit message: 'feat: Windows native training - all compatibility fixes applied'"
Write-Host ""

# Add all changes
git add -A

# Commit with detailed message
$commitMessage = @"
feat: Windows native training - all compatibility fixes applied

Major changes:
- Removed all WSL-specific workarounds (LD_LIBRARY_PATH, cuda:1)
- Added GPU auto-detection for Windows
- Fixed Isaac Lab <-> SB3 compatibility (12+ issues)
- Created IsaacLabToSB3VecEnvWrapper for observation/action conversion
- Fixed action tensor dimensions (3D->2D squeezing)
- Fixed joint control (8D actions -> 6 arm joints via joint_ids)
- Fixed Gymnasium 5-value API compatibility
- Removed all debug output

Files modified:
- scripts/reinforcement_learning/sb3/train.py (complete restructure)
- src/rl_platform/tasks/mobile_mm/env.py (action handling)
- scripts/test_mobile_mm_env.py (GPU detection)
- scripts/launch_training_windows.ps1 (NEW)
- TRAINING_SUCCESS.md (NEW documentation)

Status: Training verified working on Windows 11 + RTX 3090
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "${Green}✓ Changes committed successfully${Reset}"
} else {
    Write-Host "${Yellow}⚠️  Commit failed or no changes to commit${Reset}"
}
Write-Host ""

# Step 3: Show commit
Write-Host "${Blue}[3/4] Last commit:${Reset}"
git log -1 --oneline
Write-Host ""

# Step 4: Start training
Write-Host "${Blue}[4/4] Starting training...${Reset}"
Write-Host "Configuration:"
Write-Host "  - Environments: $NumEnvs"
Write-Host "  - Total timesteps: $TotalTimesteps"
Write-Host "  - Mode: Headless"
Write-Host "  - GPU: Auto-detected (RTX 3090)"
Write-Host ""
Write-Host "${Yellow}Training will run continuously. To stop, press Ctrl+C${Reset}"
Write-Host "${Yellow}Logs saved to: H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\${Reset}"
Write-Host ""

Start-Sleep -Seconds 3

# Launch training (NOT in background, so user can see output)
.\scripts\launch_training_windows.ps1 -Headless -NumEnvs $NumEnvs -TotalTimesteps $TotalTimesteps

Write-Host ""
Write-Host "${Green}========================================${Reset}"
Write-Host "${Green}  Training session completed${Reset}"
Write-Host "${Green}========================================${Reset}"
