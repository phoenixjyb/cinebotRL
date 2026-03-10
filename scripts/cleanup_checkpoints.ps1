#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean up checkpoint files to save disk space.

.DESCRIPTION
    Removes checkpoint files, keeping only:
    - Every 1M steps (1,000,000 steps)
    - Important milestones: 20M, 40M, 60M, 80M, 100M
    - First and last checkpoints
    - Custom keep list if specified

.PARAMETER SessionPath
    Path to the session directory (e.g., logs/sb3/mobilemmtrackee_v0/20251103_235918)

.PARAMETER KeepIntervalSteps
    Keep checkpoints at this interval (default: 1000000 = 1M steps)

.PARAMETER Milestones
    Important milestones to keep (default: 20M, 40M, 60M, 80M, 100M)

.PARAMETER DryRun
    If specified, show what would be deleted without actually deleting

.PARAMETER KeepCustom
    Additional checkpoint step numbers to keep

.EXAMPLE
    .\cleanup_checkpoints.ps1 -SessionPath "logs/sb3/mobilemmtrackee_v0/20251103_235918" -DryRun
    
.EXAMPLE
    .\cleanup_checkpoints.ps1 -SessionPath "logs/sb3/mobilemmtrackee_v0/20251103_235918" -KeepIntervalSteps 2000000
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$SessionPath,
    
    [Parameter(Mandatory=$false)]
    [int]$KeepIntervalSteps = 1000000,  # 1M steps
    
    [Parameter(Mandatory=$false)]
    [int[]]$Milestones = @(20000000, 40000000, 60000000, 80000000, 100000000),  # 20M, 40M, 60M, 80M, 100M
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [int[]]$KeepCustom = @()
)

$checkpointDir = Join-Path $SessionPath "checkpoints"

if (!(Test-Path $checkpointDir)) {
    Write-Error "Checkpoint directory not found: $checkpointDir"
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CHECKPOINT CLEANUP TOOL" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Session: $SessionPath" -ForegroundColor Yellow
Write-Host "Keep interval: Every $($KeepIntervalSteps.ToString('N0')) steps" -ForegroundColor Yellow
Write-Host "Milestones: $($Milestones -join ', ')" -ForegroundColor Yellow
if ($DryRun) {
    Write-Host "Mode: DRY RUN (no files will be deleted)`n" -ForegroundColor Magenta
} else {
    Write-Host "Mode: LIVE (files will be deleted)`n" -ForegroundColor Red
}

# Get all checkpoint files
$checkpoints = Get-ChildItem -Path $checkpointDir -Filter "ppo_*.zip" | Sort-Object Name

if ($checkpoints.Count -eq 0) {
    Write-Host "No checkpoint files found." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($checkpoints.Count) checkpoint files" -ForegroundColor White

# Extract step numbers from filenames
$checkpointInfo = @()
foreach ($checkpoint in $checkpoints) {
    if ($checkpoint.Name -match 'ppo_mobile_mm_(\d+)_steps\.zip') {
        $steps = [int]$matches[1]
        $checkpointInfo += @{
            File = $checkpoint
            Steps = $steps
            Keep = $false
        }
    }
}

if ($checkpointInfo.Count -eq 0) {
    Write-Host "No valid checkpoint files found." -ForegroundColor Yellow
    exit 0
}

# Sort by steps
$checkpointInfo = $checkpointInfo | Sort-Object { $_.Steps }

# Determine which checkpoints to keep
Write-Host "`nAnalyzing checkpoints..." -ForegroundColor Cyan

# Always keep first and last
$checkpointInfo[0].Keep = $true
$checkpointInfo[-1].Keep = $true
Write-Host "  ✓ Keeping first: $($checkpointInfo[0].Steps.ToString('N0')) steps" -ForegroundColor Green
Write-Host "  ✓ Keeping last: $($checkpointInfo[-1].Steps.ToString('N0')) steps" -ForegroundColor Green

# Keep milestones
$keptMilestones = 0
foreach ($milestone in $Milestones) {
    $closest = $checkpointInfo | Where-Object { 
        [Math]::Abs($_.Steps - $milestone) -lt ($KeepIntervalSteps / 2) 
    } | Sort-Object { [Math]::Abs($_.Steps - $milestone) } | Select-Object -First 1
    
    if ($closest) {
        $closest.Keep = $true
        $keptMilestones++
        Write-Host "  ✓ Keeping milestone: $($closest.Steps.ToString('N0')) steps (target: $($milestone.ToString('N0')))" -ForegroundColor Green
    }
}

# Keep every N steps (find closest checkpoint to each interval)
$keptInterval = 0
$maxSteps = ($checkpointInfo | Measure-Object -Property Steps -Maximum).Maximum
$tolerance = $KeepIntervalSteps / 2  # Accept checkpoints within 500K of target

# Exclude milestones from the interval targets (they're already kept)
$intervalTargets = @()
for ($targetSteps = $KeepIntervalSteps; $targetSteps -le $maxSteps; $targetSteps += $KeepIntervalSteps) {
    # Skip if this is already a milestone
    $isMilestone = $false
    foreach ($milestone in $Milestones) {
        if ([Math]::Abs($targetSteps - $milestone) -lt 100) {
            $isMilestone = $true
            break
        }
    }
    if (-not $isMilestone) {
        $intervalTargets += $targetSteps
    }
}

foreach ($targetSteps in $intervalTargets) {
    # Find closest checkpoint to this target
    $closest = $checkpointInfo | Where-Object { 
        [Math]::Abs($_.Steps - $targetSteps) -lt $tolerance 
    } | Sort-Object { [Math]::Abs($_.Steps - $targetSteps) } | Select-Object -First 1
    
    if ($closest -and !$closest.Keep) {
        $closest.Keep = $true
        $keptInterval++
    }
}
Write-Host "  ✓ Keeping $keptInterval checkpoints at ~$($KeepIntervalSteps.ToString('N0')) step intervals" -ForegroundColor Green

# Keep custom checkpoints
if ($KeepCustom.Count -gt 0) {
    $keptCustom = 0
    foreach ($customSteps in $KeepCustom) {
        $closest = $checkpointInfo | Where-Object { 
            [Math]::Abs($_.Steps - $customSteps) -lt ($KeepIntervalSteps / 2) 
        } | Sort-Object { [Math]::Abs($_.Steps - $customSteps) } | Select-Object -First 1
        
        if ($closest -and !$closest.Keep) {
            $closest.Keep = $true
            $keptCustom++
            Write-Host "  ✓ Keeping custom: $($closest.Steps.ToString('N0')) steps" -ForegroundColor Green
        }
    }
    Write-Host "  ✓ Kept $keptCustom custom checkpoints" -ForegroundColor Green
}

# Calculate statistics
$toKeep = $checkpointInfo | Where-Object { $_.Keep }
$toDelete = $checkpointInfo | Where-Object { !$_.Keep }

$totalSize = ($checkpointInfo | ForEach-Object { $_.File.Length } | Measure-Object -Sum).Sum
$keepSize = ($toKeep | ForEach-Object { $_.File.Length } | Measure-Object -Sum).Sum
$deleteSize = ($toDelete | ForEach-Object { $_.File.Length } | Measure-Object -Sum).Sum

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total checkpoints: $($checkpointInfo.Count)" -ForegroundColor White
Write-Host "  To keep: $($toKeep.Count) ($([math]::Round($keepSize/1GB, 2)) GB)" -ForegroundColor Green
Write-Host "  To delete: $($toDelete.Count) ($([math]::Round($deleteSize/1GB, 2)) GB)" -ForegroundColor Red
Write-Host "Space saved: $([math]::Round($deleteSize/1GB, 2)) GB ($([math]::Round(($deleteSize/$totalSize)*100, 1))%)" -ForegroundColor Cyan

if ($toDelete.Count -eq 0) {
    Write-Host "`nNo checkpoints to delete." -ForegroundColor Yellow
    exit 0
}

if ($DryRun) {
    Write-Host "`nDRY RUN - Files that would be deleted:" -ForegroundColor Magenta
    $toDelete | Select-Object -First 10 | ForEach-Object {
        Write-Host "  - $($_.File.Name) ($($_.Steps.ToString('N0')) steps)" -ForegroundColor Gray
    }
    if ($toDelete.Count -gt 10) {
        Write-Host "  ... and $($toDelete.Count - 10) more" -ForegroundColor Gray
    }
    Write-Host "`nRun without -DryRun to actually delete files." -ForegroundColor Yellow
} else {
    # Confirm deletion
    Write-Host "`n⚠️  WARNING: This will permanently delete $($toDelete.Count) checkpoint files!" -ForegroundColor Red
    $confirm = Read-Host "Type 'DELETE' to confirm"
    
    if ($confirm -ne 'DELETE') {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # Delete files
    Write-Host "`nDeleting checkpoints..." -ForegroundColor Cyan
    $deleted = 0
    foreach ($info in $toDelete) {
        try {
            Remove-Item -Path $info.File.FullName -Force
            $deleted++
            if ($deleted % 100 -eq 0) {
                Write-Host "  Deleted $deleted / $($toDelete.Count)..." -ForegroundColor Gray
            }
        } catch {
            Write-Host "  ✗ Failed to delete: $($info.File.Name)" -ForegroundColor Red
        }
    }
    
    Write-Host "`n✅ Cleanup complete!" -ForegroundColor Green
    Write-Host "Deleted $deleted checkpoint files" -ForegroundColor White
    Write-Host "Freed $([math]::Round($deleteSize/1GB, 2)) GB of disk space" -ForegroundColor Cyan
}

Write-Host "`n========================================`n" -ForegroundColor Cyan
