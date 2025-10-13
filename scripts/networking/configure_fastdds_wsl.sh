#!/usr/bin/env bash
# Configure Fast DDS networking inside WSL when Mihomo proxy is active.
# Usage: ./configure_fastdds_wsl.sh [windows_host_ip]
set -euo pipefail

WIN_IP=${1:-$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -n1)}
WSL_IP=$(hostname -I | awk '{print $1}')
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-55}
PROFILE_FILE="$HOME/fastdds_windows.xml"

cat <<XML >"$PROFILE_FILE"
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
  <profiles>
    <participant profile_name="wsl_fastdds" is_default_profile="true">
      <rtps>
        <builtin>
          <domainId>$ROS_DOMAIN_ID</domainId>
          <initialPeersList>
            <locator>
              <kind>UDPv4</kind>
              <address>$WIN_IP</address>
              <port>7410</port>
            </locator>
          </initialPeersList>
        </builtin>
        <userTransports>
          <transport_id>udp_transport</transport_id>
        </userTransports>
      </rtps>
      <transport_descriptors>
        <transport_descriptor>
          <transport_id>udp_transport</transport_id>
          <type>UDPv4</type>
          <sendBufferSize>65535</sendBufferSize>
          <receiveBufferSize>65535</receiveBufferSize>
        </transport_descriptor>
      </transport_descriptors>
    </participant>
  </profiles>
</dds>
XML

echo "Fast DDS profile written to $PROFILE_FILE"

# Append proxy bypass entries for Mihomo if files exist
MIHOMO_BYPASS="/etc/mihomo/no_proxy.list"
if [ -f "$MIHOMO_BYPASS" ] && ! grep -q "$WIN_IP" "$MIHOMO_BYPASS"; then
  echo "$WIN_IP/32" | sudo tee -a "$MIHOMO_BYPASS"
  echo "172.16.0.0/12" | sudo tee -a "$MIHOMO_BYPASS"
  echo "Appended bypass entries to $MIHOMO_BYPASS. Restart mihomo for changes."
fi

echo "Export the following in your ROS 2 env script:"
echo "  export ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
echo "  export FASTDDS_DEFAULT_PROFILES_FILE=$PROFILE_FILE"
echo "WSL host IP: $WSL_IP, Windows host IP: $WIN_IP"
