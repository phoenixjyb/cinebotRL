<#
.SYNOPSIS
  Opens Fast DDS ports in Windows Defender Firewall for Isaac Sim / ROS 2 and
  configures Clash proxy exclusions.

.NOTES
  Run in an elevated PowerShell session.
#>
param(
    [int[]]$UdpPorts = 7400..7410 + 7420 + 8800,
    [string]$RulePrefix = "IsaacSim FastDDS"
)

Write-Host "Creating inbound/outbound firewall rules for UDP ports: $($UdpPorts -join ',')"
foreach ($port in $UdpPorts | Select-Object -Unique) {
    New-NetFirewallRule -DisplayName "$RulePrefix Inbound $port" -Direction Inbound -Action Allow -Protocol UDP -LocalPort $port -Profile Any -ErrorAction SilentlyContinue | Out-Null
    New-NetFirewallRule -DisplayName "$RulePrefix Outbound $port" -Direction Outbound -Action Allow -Protocol UDP -LocalPort $port -Profile Any -ErrorAction SilentlyContinue | Out-Null
}

Write-Host "Ensuring Clash bypass list contains WSL / ROS subnet..."
$clashConfig = "$env:USERPROFILE\.config\clash\config.yaml"
if (Test-Path $clashConfig) {
    $config = Get-Content $clashConfig
    if ($config -notmatch "DIRECT#ROS") {
        Add-Content -Path $clashConfig -Value "  - 'DOMAIN-SUFFIX,local,DIRECT #ROS'"
        Add-Content -Path $clashConfig -Value "  - 'IP-CIDR,172.16.0.0/12,DIRECT #ROS'"
    }
    Write-Host "Updated Clash config ($clashConfig). Restart Clash service to apply." -ForegroundColor Yellow
} else {
    Write-Warning "Clash config not found at $clashConfig. Update bypass rules manually."
}

Write-Host "Firewall configuration complete." -ForegroundColor Green
