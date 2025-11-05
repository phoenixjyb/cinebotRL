# Quick verification script to prove auto-pause implementation exists
# Run this BEFORE starting training to verify code paths

Write-Host "=" * 80
Write-Host "AUTO-PAUSE IMPLEMENTATION VERIFICATION" -ForegroundColor Cyan
Write-Host "=" * 80

$projectRoot = Split-Path -Parent $PSScriptRoot
$allGood = $true

# Test 1: Check config.py has auto-pause parameters
Write-Host "`n[1/5] Checking config.py for auto-pause parameters..."
$configFile = Join-Path $projectRoot "src\rl_platform\tasks\mobile_mm\config.py"
$configContent = Get-Content $configFile -Raw

$params = @('enable_auto_pause', 'kl_threshold', 'variance_threshold', 'checkpoint_frequency_steps')
foreach ($param in $params) {
    if ($configContent -match $param) {
        $line = (Select-String -Path $configFile -Pattern $param | Select-Object -First 1).LineNumber
        Write-Host "  ✅ $param found at line $line" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $param NOT FOUND" -ForegroundColor Red
        $allGood = $false
    }
}

# Test 2: Check train.py has AutoPauseCallback class
Write-Host "`n[2/5] Checking train.py for AutoPauseCallback class..."
$trainFile = Join-Path $projectRoot "scripts\reinforcement_learning\sb3\train.py"
$classMatch = Select-String -Path $trainFile -Pattern "class AutoPauseCallback" -Quiet
if ($classMatch) {
    $line = (Select-String -Path $trainFile -Pattern "class AutoPauseCallback").LineNumber
    Write-Host "  ✅ AutoPauseCallback class found at line $line" -ForegroundColor Green
} else {
    Write-Host "  ❌ AutoPauseCallback class NOT FOUND" -ForegroundColor Red
    $allGood = $false
}

# Test 3: Check train.py has callback registration
Write-Host "`n[3/5] Checking train.py for callback registration..."
$registrationMatch = Select-String -Path $trainFile -Pattern "auto_pause_callback = AutoPauseCallback" -Quiet
if ($registrationMatch) {
    $line = (Select-String -Path $trainFile -Pattern "auto_pause_callback = AutoPauseCallback").LineNumber
    Write-Host "  ✅ Callback registration found at line $line" -ForegroundColor Green
} else {
    Write-Host "  ❌ Callback registration NOT FOUND" -ForegroundColor Red
    $allGood = $false
}

# Test 4: Check train.py reads enable_auto_pause from config
Write-Host "`n[4/5] Checking train.py reads config parameters..."
$readMatch = Select-String -Path $trainFile -Pattern "enable_auto_pause = getattr" -Quiet
if ($readMatch) {
    $line = (Select-String -Path $trainFile -Pattern "enable_auto_pause = getattr").LineNumber
    Write-Host "  ✅ Config reading logic found at line $line" -ForegroundColor Green
} else {
    Write-Host "  ❌ Config reading logic NOT FOUND" -ForegroundColor Red
    $allGood = $false
}

# Test 5: Check train.py appends callback to list
Write-Host "`n[5/5] Checking train.py appends callback to callbacks list..."
$appendMatch = Select-String -Path $trainFile -Pattern "callbacks.append\(auto_pause_callback\)" -Quiet
if ($appendMatch) {
    $line = (Select-String -Path $trainFile -Pattern "callbacks.append\(auto_pause_callback\)").LineNumber
    Write-Host "  ✅ Callback append logic found at line $line" -ForegroundColor Green
} else {
    Write-Host "  ❌ Callback append logic NOT FOUND" -ForegroundColor Red
    $allGood = $false
}

# Final verdict
Write-Host "`n" + ("=" * 80)
if ($allGood) {
    Write-Host "✅ VERIFICATION PASSED - All auto-pause components present!" -ForegroundColor Green
    Write-Host ("=" * 80)
    Write-Host "`nWhat you'll see when you launch training:"
    Write-Host "  1. '[OK] Auto-pause enabled: KL>0.1, variance<0.0'" -ForegroundColor Yellow
    Write-Host "  2. 'Session 8g lesson: No auto-pause → collapse @ 100M'" -ForegroundColor Yellow
    Write-Host "  3. 'Session 8h fix: Proactive monitoring prevents catastrophic failure'" -ForegroundColor Yellow
    Write-Host "`nNow you can run: .\scripts\launch_session_8h.ps1 -Phase smoke"
} else {
    Write-Host "❌ VERIFICATION FAILED - Some components are missing!" -ForegroundColor Red
    Write-Host ("=" * 80)
}
Write-Host ("=" * 80)
