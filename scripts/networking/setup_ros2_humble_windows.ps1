# Source ROS 2 Humble (binary) installation on Windows
param(
    [string]$RosInstall = 'I:\\ros2humble\\ros2-windows'
)

if (-not (Test-Path $RosInstall)) {
    Write-Error "ROS 2 install not found at $RosInstall."
    exit 1
}

$env:ROS_DOMAIN_ID = 55
$env:RMW_IMPLEMENTATION = 'rmw_fastrtps_cpp'
if (Test-Path "$env:USERPROFILE\fastdds_windows.xml") {
    $env:FASTDDS_DEFAULT_PROFILES_FILE = "$env:USERPROFILE\fastdds_windows.xml"
}

$dotSourced = $false
if (Test-Path "$RosInstall\local_setup.ps1") {
    try {
        . "$RosInstall\local_setup.ps1"
        $dotSourced = $true
    } catch {
        Write-Warning "Falling back to batch setup: $($_.Exception.Message)"
    }
}

if (-not $dotSourced) {
    $batPath = $RosInstall -replace '"',''
    $envDump = cmd.exe /c "call `"$batPath\local_setup.bat`" && set"
    foreach ($line in $envDump) {
        if ($line -match '^(.*?)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

# ensure ros2 launcher directory is on PATH
if (Test-Path "$RosInstall\Scripts") {
    $env:PATH = "$RosInstall\Scripts;$env:PATH"
}
