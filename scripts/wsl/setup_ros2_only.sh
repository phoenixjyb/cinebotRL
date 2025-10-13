#!/usr/bin/env bash
# Setup ROS 2 Humble environment for WSL-Windows communication
# This script ONLY sets up ROS 2 (does not activate Python venv)
# Usage: source scripts/wsl/setup_ros2_only.sh

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ROS 2 Humble Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# ============================================================================
# 1. Source ROS 2 Humble
# ============================================================================
echo -e "${BLUE}[1/3]${NC} Sourcing ROS 2 Humble..."

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}[✓]${NC} ROS 2 Humble sourced (${ROS_DISTRO})"
    echo "      Python: $(which python3) ($(python3 --version 2>&1 | awk '{print $2}'))"
else
    echo -e "${YELLOW}[ERROR]${NC} ROS 2 Humble not found at /opt/ros/humble"
    echo "      Install with: sudo apt install ros-humble-desktop"
    return 1 2>/dev/null || exit 1
fi

# ============================================================================
# 2. Configure Fast DDS Networking
# ============================================================================
echo -e "${BLUE}[2/3]${NC} Configuring Fast DDS networking..."

# Set ROS domain ID (must match Windows side)
export ROS_DOMAIN_ID=55
echo -e "${GREEN}[✓]${NC} ROS_DOMAIN_ID set to $ROS_DOMAIN_ID"

# Set RMW implementation to Fast DDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
echo -e "${GREEN}[✓]${NC} RMW_IMPLEMENTATION set to $RMW_IMPLEMENTATION"

# Configure Fast DDS profile
FASTDDS_PROFILE="$HOME/fastdds_windows.xml"
if [ -f "$FASTDDS_PROFILE" ]; then
    export FASTDDS_DEFAULT_PROFILES_FILE="$FASTDDS_PROFILE"
    WIN_IP=$(grep -oP '<address>\K[^<]+' "$FASTDDS_PROFILE" | head -n1 2>/dev/null || echo "unknown")
    echo -e "${GREEN}[✓]${NC} Fast DDS profile loaded (Windows IP: $WIN_IP)"
else
    echo -e "${YELLOW}[WARN]${NC} Fast DDS profile not found at $FASTDDS_PROFILE"
    echo "       Run: bash scripts/networking/configure_fastdds_wsl.sh"
fi

# ============================================================================
# 3. Display Network Information
# ============================================================================
echo -e "${BLUE}[3/3]${NC} Network information..."

WSL_IP=$(hostname -I | awk '{print $1}')
WIN_IP_RESOLV=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -n1 2>/dev/null || echo "unknown")
echo "      WSL IP:     $WSL_IP"
echo "      Windows IP: $WIN_IP_RESOLV"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ROS 2 Environment Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}ROS Environment:${NC}"
echo "  • ROS_DISTRO: ${ROS_DISTRO}"
echo "  • ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
echo "  • RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION}"
echo "  • Python: $(python3 --version 2>&1 | awk '{print $2}')"
echo ""
echo -e "${BLUE}Test Commands:${NC}"
echo "  ros2 topic list                      # List available topics"
echo "  ros2 run demo_nodes_cpp talker       # Publish test messages to Windows"
echo "  ros2 run demo_nodes_cpp listener     # Subscribe to messages from Windows"
echo "  ros2 topic echo /topic_name          # Monitor a specific topic"
echo ""
