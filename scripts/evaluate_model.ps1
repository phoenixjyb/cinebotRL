# Evaluate Trained Model - Convenient Launcher
# This script provides easy access to common evaluation scenarios

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('visualize', 'test-all', 'test-chassis', 'benchmark-quick', 'benchmark-full')]
    [string]$Mode = 'visualize',
    
    [Parameter(Mandatory=$false)]
    [string]$Checkpoint = "C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip",
    
    [Parameter(Mandatory=$false)]
    [int]$NumEpisodes = 10,
    
    [Parameter(Mandatory=$false)]
    [switch]$Stochastic
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Header {
    param([string]$Text)
    Write-Host "`n$('='*80)" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "$('='*80)`n" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Text)
    Write-Host "ℹ️  $Text" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

# Check if checkpoint exists
if (-not (Test-Path $Checkpoint)) {
    Write-Host "❌ Checkpoint not found: $Checkpoint" -ForegroundColor Red
    Write-Host "`nAvailable checkpoints:" -ForegroundColor Yellow
    
    $logDir = "C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0"
    if (Test-Path $logDir) {
        Get-ChildItem $logDir -Directory | Sort-Object Name -Descending | ForEach-Object {
            $finalModel = Join-Path $_.FullName "final_model.zip"
            if (Test-Path $finalModel) {
                Write-Host "  - $finalModel" -ForegroundColor Yellow
            }
        }
    }
    
    exit 1
}

Write-Success "Found checkpoint: $Checkpoint"

# Determine deterministic flag
$deterministicFlag = if ($Stochastic) { "" } else { "--deterministic" }

# Execute based on mode
switch ($Mode) {
    'visualize' {
        Write-Header "Visual Evaluation Mode"
        Write-Info "4 parallel environments with GUI"
        Write-Info "Watch the robot follow trajectory targets"
        Write-Info "🔴 Red = Target, 🟢 Green = End-Effector"
        Write-Host ""
        
        $cmd = @"
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py ``
    --checkpoint "$Checkpoint" ``
    --num_envs 4 ``
    --num_episodes $NumEpisodes ``
    $deterministicFlag ``
    --trajectory_type multi_recorded ``
    --use_all_trajectories
"@
        
        Write-Host $cmd -ForegroundColor Gray
        Write-Host ""
        Invoke-Expression $cmd
    }
    
    'test-all' {
        Write-Header "Test on ALL Training Trajectories (1,038)"
        Write-Info "Visual mode with full trajectory dataset"
        Write-Host ""
        
        $cmd = @"
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py ``
    --checkpoint "$Checkpoint" ``
    --num_envs 4 ``
    --num_episodes $NumEpisodes ``
    $deterministicFlag ``
    --trajectory_type multi_recorded ``
    --use_all_trajectories
"@
        
        Write-Host $cmd -ForegroundColor Gray
        Write-Host ""
        Invoke-Expression $cmd
    }
    
    'test-chassis' {
        Write-Header "Test on Chassis-Required Trajectories (519)"
        Write-Info "Visual mode - validates base movement capability"
        Write-Host ""
        
        $cmd = @"
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py ``
    --checkpoint "$Checkpoint" ``
    --num_envs 4 ``
    --num_episodes $NumEpisodes ``
    $deterministicFlag ``
    --trajectory_type multi_recorded ``
    --use_chassis_only
"@
        
        Write-Host $cmd -ForegroundColor Gray
        Write-Host ""
        Invoke-Expression $cmd
    }
    
    'benchmark-quick' {
        Write-Header "Quick Benchmark (5 minutes)"
        Write-Info "Headless mode with 16 parallel environments"
        Write-Info "50 episodes for quick performance metrics"
        Write-Host ""
        
        $cmd = @"
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py ``
    --checkpoint "$Checkpoint" ``
    --num_envs 16 ``
    --num_episodes 50 ``
    $deterministicFlag ``
    --trajectory_type multi_recorded ``
    --use_all_trajectories ``
    --headless
"@
        
        Write-Host $cmd -ForegroundColor Gray
        Write-Host ""
        Invoke-Expression $cmd
    }
    
    'benchmark-full' {
        Write-Header "Full Benchmark (30 minutes)"
        Write-Info "Headless mode with 64 parallel environments"
        Write-Info "500 episodes for comprehensive statistics"
        Write-Host ""
        
        $cmd = @"
& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\evaluate.py ``
    --checkpoint "$Checkpoint" ``
    --num_envs 64 ``
    --num_episodes 500 ``
    $deterministicFlag ``
    --trajectory_type multi_recorded ``
    --use_all_trajectories ``
    --headless
"@
        
        Write-Host $cmd -ForegroundColor Gray
        Write-Host ""
        Invoke-Expression $cmd
    }
}

# Print next steps
Write-Host "`n$('='*80)" -ForegroundColor Green
Write-Host "Evaluation Complete!" -ForegroundColor Green
Write-Host "$('='*80)" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Check the summary statistics above" -ForegroundColor White
Write-Host "  2. Try different modes with: -Mode <mode>" -ForegroundColor White
Write-Host "     Available modes: visualize, test-all, test-chassis, benchmark-quick, benchmark-full" -ForegroundColor White
Write-Host "  3. Compare with intermediate checkpoints" -ForegroundColor White
Write-Host "  4. See docs\EVALUATION_GUIDE.md for detailed analysis" -ForegroundColor White
