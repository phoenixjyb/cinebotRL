#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean up checkpoint files for all training sessions.

.DESCRIPTION
    Iterates through all session directories and cleans up checkpoints,
    keeping only important milestones and regular intervals.

.PARAMETER DryRun
    If specified, show what would be deleted without actually deleting

.EXAMPLE
    .\cleanup_all_checkpoints.ps1 -DryRun
    
.EXAMPLE
    .\cleanup_all_checkpoints.ps1
#>

param(
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cleanupScript = Join-Path $scriptDir "cleanup_checkpoints.ps1"

if (!(Test-Path $cleanupScript)) {
    Write-Error "cleanup_checkpoints.ps1 not found at: $cleanupScript"
    exit 1
}

Write-Host "`n╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║" -ForegroundColor Cyan -NoNewline
Write-Host "          CLEANUP ALL TRAINING SESSION CHECKPOINTS            " -ForegroundColor Green -NoNewline
Write-Host "║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "Mode: DRY RUN (no files will be deleted)`n" -ForegroundColor Magenta
} else {
    Write-Host "Mode: LIVE (files will be deleted)`n" -ForegroundColor Red
}

# Find all session directories
$sessionsDir = "logs/sb3"
if (!(Test-Path $sessionsDir)) {
    Write-Error "Sessions directory not found: $sessionsDir"
    exit 1
}

$taskDirs = Get-ChildItem -Path $sessionsDir -Directory
$sessions = @()

foreach ($taskDir in $taskDirs) {
    $sessionDirs = Get-ChildItem -Path $taskDir.FullName -Directory
    foreach ($sessionDir in $sessionDirs) {
        $checkpointDir = Join-Path $sessionDir.FullName "checkpoints"
        if (Test-Path $checkpointDir) {
            $checkpointCount = (Get-ChildItem -Path $checkpointDir -Filter "ppo_*.zip" -ErrorAction SilentlyContinue).Count
            if ($checkpointCount -gt 0) {
                $sessions += @{
                    Path = $sessionDir.FullName
                    Name = "$($taskDir.Name)/$($sessionDir.Name)"
                    CheckpointCount = $checkpointCount
                }
            }
        }
    }
}

if ($sessions.Count -eq 0) {
    Write-Host "No sessions with checkpoints found." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($sessions.Count) sessions with checkpoints:`n" -ForegroundColor Yellow

$totalBefore = 0
$totalAfter = 0
$totalDeleted = 0

foreach ($session in $sessions) {
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host "Session: $($session.Name)" -ForegroundColor Cyan
    Write-Host "Path: $($session.Path)" -ForegroundColor Gray
    Write-Host ""
    
    # Run cleanup script for this session
    $params = @{
        SessionPath = $session.Path
    }
    
    if ($DryRun) {
        $params.DryRun = $true
    }
    
    & $cleanupScript @params
    
    Write-Host ""
}

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║" -ForegroundColor Cyan -NoNewline
Write-Host "                     ALL SESSIONS COMPLETE                     " -ForegroundColor Green -NoNewline
Write-Host "║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "This was a DRY RUN. No files were deleted." -ForegroundColor Magenta
    Write-Host "Run without -DryRun to actually delete files.`n" -ForegroundColor Yellow
}
