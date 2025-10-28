# Update Trajectory Filter Paths
# Updates all references from root to data/trajectory_filters/

$ErrorActionPreference = "Stop"

Write-Host "`n=== Updating Trajectory Filter Paths ===" -ForegroundColor Cyan

$files = @(
    "src\rl_platform\tasks\mobile_mm\env.py",
    "scripts\reinforcement_learning\sb3\train.py",
    "scripts\verify_trajectories.py",
    "scripts\test_trajectory_loading.py",
    "scripts\test_recorded_trajectories_visual.py"
)

$replacements = @{
    'chassis_required_indices\.txt' = 'data/trajectory_filters/chassis_required_indices.txt'
    'chassis_required_trajectories\.txt' = 'data/trajectory_filters/chassis_required_trajectories.txt'
}

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "`n📝 Updating: $file" -ForegroundColor Yellow
        $content = Get-Content $file -Raw
        $updated = $false
        
        foreach ($pattern in $replacements.Keys) {
            $replacement = $replacements[$pattern]
            if ($content -match $pattern) {
                $content = $content -replace $pattern, $replacement
                $updated = $true
                Write-Host "  ✅ $pattern → $replacement" -ForegroundColor Gray
            }
        }
        
        if ($updated) {
            $content | Set-Content $file -NoNewline
        } else {
            Write-Host "  ℹ️  No changes needed" -ForegroundColor DarkGray
        }
    }
}

Write-Host "`n✅ Path updates complete!" -ForegroundColor Green
Write-Host "   All trajectory filter references now point to data/trajectory_filters/`n" -ForegroundColor Yellow
