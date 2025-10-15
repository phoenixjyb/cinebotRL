<#
.SYNOPSIS
    Monitor training progress in real-time

.DESCRIPTION
    This script provides multiple ways to monitor your training:
    1. Check latest log output
    2. Monitor GPU usage
    3. Watch training metrics
    4. Launch TensorBoard
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("logs", "gpu", "tensorboard", "all")]
    [string]$Mode = "logs"
)

$ESC = [char]27
$Green = "$ESC[32m"
$Yellow = "$ESC[33m"
$Blue = "$ESC[34m"
$Reset = "$ESC[0m"

Write-Host ""
Write-Host "${Blue}========================================${Reset}"
Write-Host "${Blue}  Training Monitor${Reset}"
Write-Host "${Blue}========================================${Reset}"
Write-Host ""

$ProjectPath = "C:\Users\yanbo\wSpace\cinebotRL"
$LogsPath = "H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0"

function Show-Logs {
    Write-Host "${Blue}Latest Training Logs:${Reset}"
    Write-Host "${Yellow}(Press Ctrl+C to stop monitoring)${Reset}"
    Write-Host ""
    
    # Find the latest log directory
    $latestDir = Get-ChildItem -Path $LogsPath -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($latestDir) {
        Write-Host "Log directory: $($latestDir.FullName)"
        Write-Host ""
        
        # Try to tail the TensorBoard events file or just show directory contents
        $eventFiles = Get-ChildItem -Path $latestDir.FullName -Filter "events.out.tfevents.*" -Recurse
        if ($eventFiles) {
            Write-Host "${Green}TensorBoard logs found. Use 'tensorboard' mode to visualize.${Reset}"
        }
        
        # Show recent files
        Write-Host ""
        Write-Host "Recent files:"
        Get-ChildItem -Path $latestDir.FullName -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | Format-Table Name, Length, LastWriteTime -AutoSize
        
    } else {
        Write-Host "${Yellow}No training logs found yet.${Reset}"
        Write-Host "Expected location: $LogsPath"
    }
}

function Show-GPU {
    Write-Host "${Blue}GPU Usage:${Reset}"
    Write-Host "${Yellow}(Press Ctrl+C to stop monitoring)${Reset}"
    Write-Host ""
    
    while ($true) {
        Clear-Host
        Write-Host "${Blue}GPU Status - $(Get-Date -Format 'HH:mm:ss')${Reset}"
        Write-Host ""
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
        Write-Host ""
        Write-Host "${Yellow}Refreshing every 2 seconds... Press Ctrl+C to stop${Reset}"
        Start-Sleep -Seconds 2
    }
}

function Start-TensorBoard {
    Write-Host "${Blue}Starting TensorBoard...${Reset}"
    Write-Host ""
    Write-Host "Log directory: $LogsPath"
    Write-Host ""
    Write-Host "${Green}TensorBoard will open at: http://localhost:6006${Reset}"
    Write-Host "${Yellow}Press Ctrl+C to stop TensorBoard${Reset}"
    Write-Host ""
    
    # Change to Isaac Lab to use its Python environment
    Set-Location "I:\isaaclab"
    
    # Start TensorBoard using Isaac Lab's Python
    & .\isaaclab.bat -p -c "import tensorboard.main; tensorboard.main.run_main(['--logdir', '$LogsPath', '--host', 'localhost', '--port', '6006'])"
}

function Show-All {
    Write-Host "${Blue}Training Overview:${Reset}"
    Write-Host ""
    
    # GPU Status
    Write-Host "${Blue}GPU Status:${Reset}"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=table
    Write-Host ""
    
    # Latest logs
    Show-Logs
    Write-Host ""
    
    # Instructions
    Write-Host "${Blue}Monitoring Options:${Reset}"
    Write-Host "  ${Yellow}.\scripts\monitor_training.ps1 -Mode logs${Reset}       - Watch log files"
    Write-Host "  ${Yellow}.\scripts\monitor_training.ps1 -Mode gpu${Reset}        - Monitor GPU usage"
    Write-Host "  ${Yellow}.\scripts\monitor_training.ps1 -Mode tensorboard${Reset} - Launch TensorBoard"
    Write-Host ""
}

# Execute based on mode
switch ($Mode) {
    "logs" { Show-Logs }
    "gpu" { Show-GPU }
    "tensorboard" { Start-TensorBoard }
    "all" { Show-All }
}

Write-Host ""
