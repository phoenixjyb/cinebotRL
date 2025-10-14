#!/usr/bin/env bash
# Install Isaac Lab in existing WSL venv for headless RL training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv_rl311"

echo "==========================================="
echo "Isaac Lab Installation for WSL"
echo "==========================================="
echo ""

# Check if venv exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[ERROR] Virtual environment not found at: $VENV_DIR"
    echo "Create it first with: ./scripts/setup_rl_venv.sh --python python3.11"
    exit 1
fi

# Activate venv
echo "[INFO] Activating venv: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "[INFO] Python version: $PYTHON_VERSION"

if ! python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10), (3,11)] else 1)"; then
    echo "[WARN] Isaac Lab officially supports Python 3.10 or 3.11"
    echo "       Current version: $PYTHON_VERSION"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA availability
echo ""
echo "[INFO] Checking CUDA availability..."
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not available. Ensure NVIDIA drivers are installed."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1
echo ""

# Check if PyTorch with CUDA is already installed
echo "[INFO] Checking PyTorch..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "       ✓ PyTorch $TORCH_VERSION with CUDA ($GPU_NAME)"
else
    echo "[ERROR] PyTorch with CUDA is not available"
    echo "       Run setup_rl_venv.sh to install PyTorch first"
    exit 1
fi

# Install Isaac Lab
echo ""
echo "[INFO] Installing Isaac Lab packages..."
echo ""

# Option to choose installation method
echo "Select installation method:"
echo "  1) Lightweight (pip install isaaclab) - Recommended for WSL training"
echo "  2) Full stack (isaacsim-* packages) - Includes full Isaac Sim runtime"
echo "  3) From source (git clone) - For development"
echo ""
read -p "Choice [1-3]: " -n 1 -r INSTALL_CHOICE
echo ""

case $INSTALL_CHOICE in
    1)
        echo "[INFO] Installing lightweight Isaac Lab..."
        # Update pip first
        pip install --upgrade pip
        
        # Install Isaac Lab and core dependencies
        pip install isaaclab || {
            echo ""
            echo "[WARN] 'isaaclab' package not found. Trying alternative installation..."
            echo "[INFO] Installing from NVIDIA PyPI index..."
            pip install --extra-index-url https://pypi.nvidia.com omni-isaac-lab || {
                echo ""
                echo "[ERROR] Could not install Isaac Lab via pip."
                echo "[INFO] As of Oct 2024, Isaac Lab may require source installation."
                echo "       Falling back to source install (Option 3)..."
                INSTALL_CHOICE=3
            }
        }
        ;;
    
    2)
        echo "[INFO] Installing full Isaac Sim stack (this may take 10-20 minutes)..."
        pip install --upgrade pip
        
        # Install Isaac Sim packages
        pip install --extra-index-url https://pypi.nvidia.com \
            isaacsim-rl \
            isaacsim-replicator \
            isaacsim-extscache-physics \
            isaacsim-extscache-kit-sdk \
            isaacsim-extscache-kit \
            isaacsim-app || {
            echo "[ERROR] Failed to install Isaac Sim packages"
            echo "Check package availability at: https://pypi.nvidia.com"
            exit 1
        }
        
        # Install Isaac Lab
        pip install isaaclab || pip install omni-isaac-lab || {
            echo "[WARN] Could not install Isaac Lab package, will try source install"
            INSTALL_CHOICE=3
        }
        ;;
    
    3)
        echo "[INFO] Installing from source..."
        ISAAC_LAB_DIR="$HOME/IsaacLab"
        
        if [[ -d "$ISAAC_LAB_DIR" ]]; then
            echo "[INFO] IsaacLab directory exists at: $ISAAC_LAB_DIR"
            read -p "       Pull latest changes? (Y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                cd "$ISAAC_LAB_DIR"
                git pull origin main
            fi
        else
            echo "[INFO] Cloning Isaac Lab to: $ISAAC_LAB_DIR"
            git clone https://github.com/isaac-sim/IsaacLab.git "$ISAAC_LAB_DIR"
            cd "$ISAAC_LAB_DIR"
        fi
        
        echo "[INFO] Installing Isaac Lab in editable mode..."
        cd "$ISAAC_LAB_DIR"
        pip install -e .
        
        # Set environment variable
        echo "[INFO] Adding ISAAC_LAB_PATH to environment"
        if ! grep -q "ISAAC_LAB_PATH" ~/.bashrc; then
            echo "export ISAAC_LAB_PATH=\"$ISAAC_LAB_DIR\"" >> ~/.bashrc
        fi
        export ISAAC_LAB_PATH="$ISAAC_LAB_DIR"
        ;;
    
    *)
        echo "[ERROR] Invalid choice"
        exit 1
        ;;
esac

# If still using option 3 (from fallback), do source install
if [[ $INSTALL_CHOICE == "3" ]]; then
    ISAAC_LAB_DIR="$HOME/IsaacLab"
    
    if [[ ! -d "$ISAAC_LAB_DIR" ]]; then
        echo "[INFO] Cloning Isaac Lab to: $ISAAC_LAB_DIR"
        git clone https://github.com/isaac-sim/IsaacLab.git "$ISAAC_LAB_DIR"
    fi
    
    cd "$ISAAC_LAB_DIR"
    echo "[INFO] Installing Isaac Lab from source..."
    pip install -e .
fi

# Install additional dependencies
echo ""
echo "[INFO] Installing additional training dependencies..."
pip install wandb tensorboard hydra-core

# Set up environment variables
echo ""
echo "[INFO] Setting up environment variables..."

ENV_FILE="$HOME/.isaaclab_env"
cat > "$ENV_FILE" <<'ENV'
# Isaac Lab environment for WSL headless training
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/lib/wsl/lib"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

# PhysX GPU settings for high-performance simulation
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=524288
export PHYSX_GPU_MAX_RIGID_PATCH_COUNT=163840
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=2097152
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=2097152

# Headless rendering
export DISPLAY=""

# Isaac Lab path (if installed from source)
if [[ -d "$HOME/IsaacLab" ]]; then
    export ISAAC_LAB_PATH="$HOME/IsaacLab"
fi
ENV

echo "[INFO] Created environment file: $ENV_FILE"
echo "[INFO] Adding to venv activation script..."

# Add to venv activation
ACTIVATE_FILE="$VENV_DIR/bin/activate"
if ! grep -q "isaaclab_env" "$ACTIVATE_FILE"; then
    cat >> "$ACTIVATE_FILE" <<'ACT'

# Isaac Lab environment
if [ -f ~/.isaaclab_env ]; then
    source ~/.isaaclab_env
fi
ACT
    echo "       ✓ Updated venv activation script"
fi

# Add to bashrc if not already there
if ! grep -q "isaaclab_env" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Isaac Lab environment" >> ~/.bashrc
    echo "if [ -f ~/.isaaclab_env ]; then" >> ~/.bashrc
    echo "    source ~/.isaaclab_env" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
    echo "       ✓ Added to ~/.bashrc"
fi

# Source the environment
source "$ENV_FILE"

echo ""
echo "==========================================="
echo "Installation Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Deactivate and reactivate your venv:"
echo "     deactivate && source $VENV_DIR/bin/activate"
echo ""
echo "  2. Run verification script:"
echo "     python scripts/verify_isaaclab_wsl.py"
echo ""
echo "  3. Test with minimal environment:"
echo "     python -c 'import omni.isaac.lab; print(\"✓ Isaac Lab ready\")'"
echo ""
echo "  4. Register and test our custom task:"
echo "     python scripts/verify_isaaclab_wsl.py"
echo ""
