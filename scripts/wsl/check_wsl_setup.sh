#!/usr/bin/env bash
# Comprehensive WSL environment status check for the CinebotRL project
# Verifies all components needed for RL training coordination with Windows Isaac Sim/Lab
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_NAME=".venv_rl311"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

# Track overall status
ERRORS=0
WARNINGS=0

header "WSL Environment Status Check"
info "Project root: $PROJECT_ROOT"
info "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================================
header "1. System Information"
# ============================================================================

info "OS Information:"
if [ -f /etc/os-release ]; then
    source /etc/os-release
    echo "  Distribution: $NAME $VERSION"
    success "Running on Ubuntu 22.04 (expected)"
else
    error "Cannot read /etc/os-release"
    ((ERRORS++))
fi

info "Kernel:"
echo "  $(uname -r)"

# ============================================================================
header "2. GPU and CUDA Status"
# ============================================================================

info "Checking NVIDIA GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while IFS=',' read -r gpu driver memory; do
        echo "  GPU: $gpu"
        echo "  Driver: $driver"
        echo "  Memory: $memory"
    done
    success "NVIDIA driver accessible from WSL"
else
    error "nvidia-smi not found - GPU passthrough may not be working"
    ((ERRORS++))
fi

info "Checking CUDA toolkit..."
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
    echo "  NVCC version: $CUDA_VERSION"
    success "CUDA toolkit installed (expected 12.6)"
else
    warn "nvcc not found in PATH - CUDA toolkit may not be installed"
    ((WARNINGS++))
fi

# ============================================================================
header "3. Python Virtual Environment"
# ============================================================================

info "Checking for $VENV_NAME..."
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    success "Virtual environment exists at $VENV_PATH"
    
    # Activate and check packages
    info "Activating venv and checking key packages..."
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "  Python version: $PYTHON_VERSION"
    
    # Check critical packages
    PACKAGES=("torch" "gymnasium" "stable_baselines3" "rclpy" "numpy" "pandas")
    for pkg in "${PACKAGES[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            success "  $pkg: $VERSION"
        else
            error "  $pkg: NOT INSTALLED"
            ((ERRORS++))
        fi
    done
    
    # Check PyTorch CUDA
    info "Checking PyTorch CUDA availability..."
    TORCH_CUDA=$(python -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null || echo "ERROR")
    if [ "$TORCH_CUDA" = "YES" ]; then
        CUDA_DEVICES=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        success "  PyTorch CUDA: Available ($CUDA_DEVICES device(s))"
        python -c "import torch; [print(f'    Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null
    else
        error "  PyTorch CUDA: NOT AVAILABLE"
        ((ERRORS++))
    fi
    
else
    error "Virtual environment not found at $VENV_PATH"
    echo "  Run: bash scripts/setup_rl_venv.sh --python python3.11"
    ((ERRORS++))
fi

# ============================================================================
header "4. ROS 2 Humble Installation"
# ============================================================================

info "Checking for ROS 2 Humble..."
if command -v ros2 >/dev/null 2>&1; then
    ROS2_PATH=$(which ros2)
    echo "  ROS 2 found at: $ROS2_PATH"
    
    # Try to source ROS 2 environment
    if [ -f /opt/ros/humble/setup.bash ]; then
        # shellcheck disable=SC1091
        source /opt/ros/humble/setup.bash >/dev/null 2>&1
        success "ROS 2 Humble installed and sourced"
        
        # Check ROS environment variables
        if [ -n "${ROS_DISTRO:-}" ]; then
            echo "  ROS_DISTRO: $ROS_DISTRO"
        fi
        if [ -n "${ROS_DOMAIN_ID:-}" ]; then
            echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
        else
            warn "  ROS_DOMAIN_ID not set (should be 55 for Windows communication)"
            ((WARNINGS++))
        fi
        if [ -n "${RMW_IMPLEMENTATION:-}" ]; then
            echo "  RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
        else
            warn "  RMW_IMPLEMENTATION not set (should be rmw_fastrtps_cpp)"
            ((WARNINGS++))
        fi
        
        # Check demo nodes
        if ros2 pkg list | grep -q demo_nodes_cpp; then
            success "  demo_nodes_cpp available for testing"
        else
            warn "  demo_nodes_cpp not found"
            ((WARNINGS++))
        fi
        
    else
        warn "ROS 2 setup script not found at /opt/ros/humble/setup.bash"
        ((WARNINGS++))
    fi
else
    error "ROS 2 not found - install with: sudo apt install ros-humble-desktop"
    ((ERRORS++))
fi

# ============================================================================
header "5. Fast DDS Network Configuration"
# ============================================================================

info "Checking Fast DDS configuration..."
FASTDDS_PROFILE="$HOME/fastdds_windows.xml"
if [ -f "$FASTDDS_PROFILE" ]; then
    success "Fast DDS profile exists at $FASTDDS_PROFILE"
    
    # Extract configured IPs
    WIN_IP=$(grep -oP '<address>\K[^<]+' "$FASTDDS_PROFILE" | head -n1 || echo "unknown")
    echo "  Windows IP: $WIN_IP"
    
    if [ -n "${FASTDDS_DEFAULT_PROFILES_FILE:-}" ]; then
        if [ "$FASTDDS_DEFAULT_PROFILES_FILE" = "$FASTDDS_PROFILE" ]; then
            success "  FASTDDS_DEFAULT_PROFILES_FILE is set correctly"
        else
            warn "  FASTDDS_DEFAULT_PROFILES_FILE points to: $FASTDDS_DEFAULT_PROFILES_FILE"
            ((WARNINGS++))
        fi
    else
        warn "  FASTDDS_DEFAULT_PROFILES_FILE not set in environment"
        ((WARNINGS++))
    fi
else
    warn "Fast DDS profile not found - run: bash scripts/networking/configure_fastdds_wsl.sh"
    ((WARNINGS++))
fi

info "Network information:"
WSL_IP=$(hostname -I | awk '{print $1}')
WIN_IP_RESOLV=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -n1)
echo "  WSL IP: $WSL_IP"
echo "  Windows IP (from resolv.conf): $WIN_IP_RESOLV"

# ============================================================================
header "6. Required Scripts Check"
# ============================================================================

info "Checking helper scripts..."
SCRIPTS=(
    "scripts/wsl/activate_rl_env_wsl.sh"
    "scripts/networking/configure_fastdds_wsl.sh"
    "scripts/wsl/check_phase0_prereqs.sh"
)

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$PROJECT_ROOT/$script"
    if [ -f "$SCRIPT_PATH" ]; then
        success "  $script"
    else
        error "  $script NOT FOUND"
        ((ERRORS++))
    fi
done

# ============================================================================
header "7. Windows Connectivity Test"
# ============================================================================

info "Testing network connectivity to Windows host..."
if ping -c 2 -W 1 "$WIN_IP_RESOLV" >/dev/null 2>&1; then
    success "Windows host is reachable at $WIN_IP_RESOLV"
else
    warn "Cannot ping Windows host at $WIN_IP_RESOLV"
    ((WARNINGS++))
fi

# Check if any ROS 2 topics are visible (requires Windows to be publishing)
if command -v ros2 >/dev/null 2>&1; then
    info "Checking for ROS 2 topics from Windows (5 second timeout)..."
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
    TOPICS=$(timeout 5 ros2 topic list 2>/dev/null || echo "")
    if [ -n "$TOPICS" ] && [ "$TOPICS" != "" ]; then
        success "ROS 2 topics detected:"
        echo "$TOPICS" | head -n 5 | sed 's/^/    /'
        if [ $(echo "$TOPICS" | wc -l) -gt 5 ]; then
            echo "    ... and $(( $(echo "$TOPICS" | wc -l) - 5 )) more"
        fi
    else
        warn "No ROS 2 topics detected (Windows may not be publishing)"
        echo "  This is normal if Windows Isaac Sim/ROS 2 is not running"
        ((WARNINGS++))
    fi
fi

# ============================================================================
header "8. Project Structure"
# ============================================================================

info "Checking project directories..."
DIRS=("src" "scripts" "docs" "assets_own" "experiments")
for dir in "${DIRS[@]}"; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        success "  $dir/"
    else
        warn "  $dir/ NOT FOUND"
        ((WARNINGS++))
    fi
done

# ============================================================================
header "Summary"
# ============================================================================

echo ""
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    success "All checks passed! WSL environment is ready for RL coordination."
elif [ $ERRORS -eq 0 ]; then
    warn "Setup complete with $WARNINGS warning(s). Review messages above."
else
    error "Found $ERRORS error(s) and $WARNINGS warning(s). Please fix issues above."
    exit 1
fi

echo ""
info "Next steps:"
echo "  1. Activate environment: source scripts/wsl/setup_wsl_environment.sh"
echo "  2. Test ROS 2 communication: ros2 run demo_nodes_cpp talker"
echo "  3. Start Isaac Sim on Windows and verify topic exchange"
echo "  4. Begin RL training workflow"
echo ""
