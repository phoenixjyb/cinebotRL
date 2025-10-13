# Cinebot RL Environment Notes

## Toolchain split
- **Isaac Sim / Isaac Lab (Windows)**: installed at `I:\isaacsim` using the bundled Python 3.11.13 (`python.bat`).
- **ROS 2 Humble (Windows)**: extracted to `I:\ros2humble\ros2-windows`; run ROS tooling with Python 3.10 via `py -3.10` to align with Humble.
- **WSL2 (Ubuntu 22.04)**: hosts ROS 2 nodes and scripting; Fast DDS bridge connects back to the Windows simulator.

## Helper scripts
- `scripts/networking/configure_fastdds_firewall.ps1`: opens UDP 7400-7410/7420/8800 on Windows Defender and adds Clash bypass entries.
- `scripts/networking/configure_fastdds_wsl.sh`: generates `~/fastdds_windows.xml`, appends Mihomo bypass entries, and prints the required environment exports.
- `scripts/networking/setup_ros2_humble_windows.ps1`: sources `I:\ros2humble\ros2-windows\local_setup.bat`, sets `ROS_DOMAIN_ID=55`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and reuses the Fast DDS profile when present.

## Daily workflow
1. **Windows shell**
   ```powershell
   Set-Location C:\Users\yanbo\wSpace\cinebotRL
   .\scripts\networking\setup_ros2_humble_windows.ps1
   cmd /c "call I:\ros2humble\ros2-windows\local_setup.bat && ros2 run demo_nodes_cpp listener"
   ```
2. **WSL shell**
   ```bash
   cd /mnt/c/Users/yanbo/wSpace/cinebotRL/scripts/networking
   ./configure_fastdds_wsl.sh   # only needed when IPs change
   export ROS_DOMAIN_ID=55
   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
   ros2 run demo_nodes_cpp talker
   ```
3. Once messages flow, launch Isaac Sim with `--/exts/ros2_bridge/useDomainID=55` and the simulator will publish into the same ROS graph consumed by WSL nodes.

## Python version policy
- ROS 2 Humble commands (`ros2`, `pip`, `fastdds`) should use Python 3.10 (`py -3.10 ...`).
- Isaac Sim / Isaac Lab continues on its bundled Python 3.11.

## WSL RL training environment
- Source the helper whenever you work on RL tooling:
  ```bash
  cd /mnt/h/wSpace/cinebotRL
  source scripts/wsl/activate_rl_env_wsl.sh
  ```
  The helper targets `.venv_rl311` by default (derived from the interpreter version), prepends CUDA 12.6 libraries (`/usr/lib/wsl/lib`, `/usr/local/cuda-12.6/lib64`) and prints a quick torch CUDA check.
  Export `RL_VENV_NAME` beforehand if you need to point at a different virtualenv.
- If you prefer manual setup, replicate the exports inside your shell profile before running training jobs.

Detailed notes and verification steps live in `docs/tracking/phase0_environment.md` (sections 0.8–0.9).

