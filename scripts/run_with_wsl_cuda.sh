#!/bin/bash
# Wrapper script to set LD_LIBRARY_PATH before launching Isaac Sim in WSL2
#
# CRITICAL: This must be run BEFORE any Python/Isaac Sim imports
# so that Warp can find the CUDA driver in /usr/lib/wsl/lib

# Activate virtual environment if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    if [[ -f "$PROJECT_ROOT/.venv_rl311/bin/activate" ]]; then
        source "$PROJECT_ROOT/.venv_rl311/bin/activate"
        echo "✓ Activated virtual environment: .venv_rl311"
    fi
fi

# Add WSL CUDA library path
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"

# Set device ordering for consistent GPU numbering
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# DO NOT set CUDA_VISIBLE_DEVICES - Omniverse warns it causes issues
# Let Isaac Sim auto-select GPUs, we specify device="cuda:1" in AppLauncher

# Accept EULA
export OMNI_KIT_ACCEPT_EULA="yes"
export ACCEPT_EULA="YES"

# Run the command passed as arguments
exec "$@"
