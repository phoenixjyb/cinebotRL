<#
.SYNOPSIS
    Install cinebotRL project into Isaac Lab's Python environment

.DESCRIPTION
    This script activates Isaac Lab's environment and installs the
    cinebotRL project in editable mode, along with required dependencies.
#>

param(
    [switch]$SkipDependencies
)

# Colors
$ESC = [char]27
$Green = "$ESC[32m"
$Yellow = "$ESC[33m"
$Red = "$ESC[31m"
$Blue = "$ESC[34m"
$Reset = "$ESC[0m"

Write-Host ""
Write-Host "${Blue}========================================${Reset}"
Write-Host "${Blue}  Setup CinebotRL in Isaac Lab${Reset}"
Write-Host "${Blue}========================================${Reset}"
Write-Host ""

# Check paths
$IsaacLabPath = "I:\isaaclab"
$ProjectPath = "C:\Users\yanbo\wSpace\cinebotRL"

if (-not (Test-Path $IsaacLabPath)) {
    Write-Host "${Red}✗ Isaac Lab not found at: $IsaacLabPath${Reset}"
    exit 1
}
Write-Host "${Green}✓${Reset} Isaac Lab found: $IsaacLabPath"

if (-not (Test-Path $ProjectPath)) {
    Write-Host "${Red}✗ Project not found at: $ProjectPath${Reset}"
    exit 1
}
Write-Host "${Green}✓${Reset} Project found: $ProjectPath"

# Create a temporary Python script to do the installation
$tempScript = Join-Path $env:TEMP "setup_cinebotrl.py"
$pythonCode = @"
import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

# Change to project directory
project_path = r"$ProjectPath"
os.chdir(project_path)

# Check if already installed
try:
    import cinebotrl
    print("✓ cinebotrl already installed")
    print(f"  Location: {cinebotrl.__file__ if hasattr(cinebotrl, '__file__') else 'package level'}")
except ImportError:
    print("○ cinebotrl not yet installed")

# Install project in editable mode
print("\n" + "="*50)
print("Installing cinebotrl project...")
print("="*50)
if not run_command("pip install -e .", "Installing project in editable mode"):
    sys.exit(1)

# Install Stable Baselines3 if not skipping
skip_deps = "$($SkipDependencies.ToString().ToLower())"
if skip_deps != "true":
    print("\n" + "="*50)
    print("Installing Stable Baselines3...")
    print("="*50)
    if not run_command("pip install stable-baselines3[extra]", "Installing SB3"):
        print("⚠️  Warning: SB3 installation failed, but continuing...")

# Verify installation
print("\n" + "="*50)
print("Verifying installation...")
print("="*50)

try:
    import cinebotrl
    print("✓ cinebotrl can be imported")
except ImportError as e:
    print(f"✗ Failed to import cinebotrl: {e}")
    sys.exit(1)

try:
    from src.task_spec import register_isaac_lab_tasks
    print("✓ Can import task_spec")
except ImportError as e:
    print(f"✗ Failed to import task_spec: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"⚠️  PyTorch not available: {e}")

try:
    from stable_baselines3 import PPO
    print("✓ Stable Baselines3 available")
except ImportError:
    print("⚠️  Stable Baselines3 not available (install with: pip install stable-baselines3[extra])")

try:
    import gymnasium
    print(f"✓ Gymnasium: {gymnasium.__version__}")
except ImportError:
    print("⚠️  Gymnasium not available")

print("\n" + "="*50)
print("✓ Setup completed successfully!")
print("="*50)
print("\nYou can now run:")
print("  .\\scripts\\launch_training_windows.ps1 -Test -Headless")
"@

Set-Content -Path $tempScript -Value $pythonCode

Write-Host ""
Write-Host "${Blue}Running installation...${Reset}"
Write-Host ""

# Change to Isaac Lab and run
Set-Location $IsaacLabPath

# Run the setup script using Isaac Lab's Python
& .\isaaclab.bat -p $tempScript

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "${Green}========================================${Reset}"
    Write-Host "${Green}  ✓ Setup completed successfully!${Reset}"
    Write-Host "${Green}========================================${Reset}"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Test the environment:"
    Write-Host "     ${Yellow}.\scripts\launch_training_windows.ps1 -Test -Headless${Reset}"
    Write-Host ""
    Write-Host "  2. Start training:"
    Write-Host "     ${Yellow}.\scripts\launch_training_windows.ps1 -Headless -NumEnvs 512${Reset}"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "${Red}========================================${Reset}"
    Write-Host "${Red}  ✗ Setup failed${Reset}"
    Write-Host "${Red}========================================${Reset}"
    exit $LASTEXITCODE
}

# Cleanup
Remove-Item $tempScript -ErrorAction SilentlyContinue
