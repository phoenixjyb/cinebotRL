# Quick Diagnostic - Run this first to understand the reward breakdown

Write-Host "`n================================================================================" -ForegroundColor Red
Write-Host "CRITICAL: Contact Force API Returning Zeros!" -ForegroundColor Red
Write-Host "================================================================================" -ForegroundColor Red
Write-Host "`nYour training completed successfully, BUT self-collision detection was NOT working." -ForegroundColor Yellow
Write-Host "The contact force API returned zeros throughout training." -ForegroundColor Yellow
Write-Host "`nThis means the robot learned it could collide with itself without penalty." -ForegroundColor Yellow
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Diagnose Which Penalty is Exploding" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$checkpoint = "C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251018_001233\final_model.zip"

Write-Host "`nRunning reward component diagnostic (100 steps)..." -ForegroundColor White
Write-Host "This will show which reward term is causing -200k rewards.`n" -ForegroundColor Gray

& "I:\isaaclab\isaaclab.bat" -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\diagnose_rewards.py `
    --checkpoint $checkpoint `
    --num_envs 4 `
    --num_steps 100

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "1. Review the reward component analysis above" -ForegroundColor White
Write-Host "2. Identify which penalty term is exploding (marked with ⚠️)" -ForegroundColor White
Write-Host "3. Read docs\CRITICAL_CONTACT_FORCE_ISSUE.md for solutions" -ForegroundColor White
Write-Host "4. Implement alternative self-collision detection" -ForegroundColor White
Write-Host "5. Retrain with working collision detection" -ForegroundColor White
Write-Host "`nDon't panic! The training infrastructure is solid - we just need to fix collision detection." -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
