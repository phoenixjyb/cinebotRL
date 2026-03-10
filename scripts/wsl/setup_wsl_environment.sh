#!/usr/bin/env bash
# Unified WSL environment setup for CinebotRL project
# This script activates Python venv, ROS 2 Humble, and configures Fast DDS networking
# Usage: source scripts/wsl/setup_wsl_environment.sh

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CinebotRL WSL Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Determine project root
if [ -n "${BASH_SOURCE[0]}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
else
    PROJECT_ROOT="$(pwd)"
fi

VENV_NAME=".venv_rl311"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
FASTDDS_PROFILE="$HOME/fastdds_windows.xml"

# ============================================================================
# 1. Activate Python Virtual Environment
# ============================================================================
echo -e "${BLUE}[1/4]${NC} Activating Python virtual environment..."

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${YELLOW}[WARN]${NC} Virtual environment not found at $VENV_PATH"
    echo "       Run: bash scripts/setup_rl_venv.sh --python python3.11"
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}[✓]${NC} Python $PYTHON_VERSION activated"

# ============================================================================
# 2. Configure CUDA Paths
# ============================================================================
echo -e "${BLUE}[2/4]${NC} Configuring CUDA paths..."

CUDA_PREFIX=${CUDA_PREFIX:-$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)}
CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda-12.6}
if [ -d "$CUDA_PREFIX/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CUDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
    export PATH="${CUDA_PREFIX}/bin:${PATH}"
    echo -e "${GREEN}[✓]${NC} CUDA paths configured ($(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo 'version unknown'))"
else
    echo -e "${YELLOW}[WARN]${NC} CUDA prefix $CUDA_PREFIX/lib64 not found"
fi

# Quick PyTorch CUDA check
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}[✓]${NC} PyTorch CUDA available ($CUDA_DEVICE_COUNT device(s))"
else
    echo -e "${YELLOW}[WARN]${NC} PyTorch CUDA not available"
fi

# ============================================================================
# 3. Source ROS 2 Humble
# ============================================================================
echo -e "${BLUE}[3/4]${NC} Sourcing ROS 2 Humble..."

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}[✓]${NC} ROS 2 Humble sourced (${ROS_DISTRO})"
else
    echo -e "${YELLOW}[WARN]${NC} ROS 2 Humble not found at /opt/ros/humble"
    return 1 2>/dev/null || exit 1
fi

# ============================================================================
# 4. Configure Fast DDS Networking for Windows Communication
# ============================================================================
echo -e "${BLUE}[4/4]${NC} Configuring Fast DDS networking..."

# Set ROS domain ID (must match Windows side)
export ROS_DOMAIN_ID=55
echo -e "${GREEN}[✓]${NC} ROS_DOMAIN_ID set to $ROS_DOMAIN_ID"

# Set RMW implementation to Fast DDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
echo -e "${GREEN}[✓]${NC} RMW_IMPLEMENTATION set to $RMW_IMPLEMENTATION"

# Configure Fast DDS profile
if [ -f "$FASTDDS_PROFILE" ]; then
    export FASTDDS_DEFAULT_PROFILES_FILE="$FASTDDS_PROFILE"
    WIN_IP=$(grep -oP '<address>\K[^<]+' "$FASTDDS_PROFILE" | head -n1 2>/dev/null || echo "unknown")
    echo -e "${GREEN}[✓]${NC} Fast DDS profile loaded (Windows IP: $WIN_IP)"
else
    echo -e "${YELLOW}[WARN]${NC} Fast DDS profile not found at $FASTDDS_PROFILE"
    echo "       Run: bash scripts/networking/configure_fastdds_wsl.sh"
fi

# Display network information
WSL_IP=$(hostname -I | awk '{print $1}')
WIN_IP_RESOLV=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -n1 2>/dev/null || echo "unknown")
echo ""
echo -e "${BLUE}Network Info:${NC}"
echo "  WSL IP:     $WSL_IP"
echo "  Windows IP: $WIN_IP_RESOLV"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Environment Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Active Components:${NC}"
echo "  • Python $PYTHON_VERSION (${VENV_NAME})"
echo "  • ROS 2 ${ROS_DISTRO}"
echo "  • Fast DDS (Domain ID: ${ROS_DOMAIN_ID})"
echo ""
echo -e "${BLUE}Quick Test Commands:${NC}"
echo "  ros2 topic list                      # List available topics"
echo "  ros2 run demo_nodes_cpp talker       # Publish test messages"
echo "  ros2 run demo_nodes_cpp listener     # Subscribe to test messages"
echo ""
echo -e "${BLUE}Useful Aliases (optional):${NC}"
echo "  alias wsl-env='source $PROJECT_ROOT/scripts/wsl/setup_wsl_environment.sh'"
echo "  alias check-wsl='bash $PROJECT_ROOT/scripts/wsl/check_wsl_setup.sh'"
echo ""
