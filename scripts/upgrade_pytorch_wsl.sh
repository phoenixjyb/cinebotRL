#!/usr/bin/env bash
# Upgrade PyTorch to NVIDIA's recommended version for Isaac Lab
set -euo pipefail

VENV_DIR="/mnt/c/Users/yanbo/wSpace/cinebotRL/.venv_rl311"

echo "=========================================="
echo "Upgrade PyTorch to NVIDIA Recommended"
echo "=========================================="
echo ""

# Check if venv exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[ERROR] Virtual environment not found at: $VENV_DIR"
    exit 1
fi

# Activate venv
echo "[INFO] Activating venv..."
source "$VENV_DIR/bin/activate"

# Check current PyTorch version
if python -c "import torch" 2>/dev/null; then
    CURRENT_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo "[INFO] Current PyTorch version: $CURRENT_VERSION"
    echo "[INFO] CUDA available: $CUDA_AVAILABLE"
else
    echo "[WARN] PyTorch not currently installed"
fi

echo ""
echo "NVIDIA recommends for Isaac Lab (Oct 2024):"
echo "  - PyTorch 2.7.0"
echo "  - CUDA 12.8"
echo "  - torchvision 0.22.0"
echo ""

read -p "Proceed with upgrade? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "[INFO] Upgrade cancelled"
    exit 0
fi

echo ""
echo "[INFO] Upgrading PyTorch..."
pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[INFO] Verifying installation..."

# Verify
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    NEW_VERSION=$(python -c "import torch; print(torch.__version__)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    
    echo ""
    echo "✓ Upgrade successful!"
    echo "  - PyTorch version: $NEW_VERSION"
    echo "  - CUDA version: $CUDA_VERSION"
    echo "  - GPU: $GPU_NAME"
    echo ""
else
    echo ""
    echo "[ERROR] PyTorch upgrade failed or CUDA not available"
    echo "Check with: python -c 'import torch; print(torch.cuda.is_available())'"
    exit 1
fi

echo "[INFO] PyTorch is ready for Isaac Lab installation"
echo ""
echo "Next step: Run Isaac Lab installer"
echo "  ./scripts/install_isaaclab_wsl.sh"
