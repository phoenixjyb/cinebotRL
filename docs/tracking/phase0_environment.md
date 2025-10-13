# Phase 0 - Environment Bring-Up Log

## 0.1 Host and GPU Verification
- lsb_release -a -> Ubuntu 22.04.5 LTS (Jammy) under WSL2.
- nvidia-smi -> Driver 580.97, CUDA 13.0, GPUs detected: Quadro P2000 (display) and RTX 3090 (training).
- nvcc --version -> Cuda compilation tools, release 12.6, V12.6.85 (revalidated 2025-10-13).

## 0.2 Python and Libraries
- python3 available at /usr/bin/python3.
- import torch -> ModuleNotFoundError (PyTorch not yet installed).

## 0.3 Outstanding Actions
1. Create Python environment (conda, mamba, or venv) with GPU-enabled PyTorch and tooling.
2. Prepare Isaac Sim and Isaac Lab install once CUDA and dependencies are in place.
3. Re-run PyTorch GPU smoke test after installation.

## 0.4 Suggested Manual Commands
- Configure NVIDIA CUDA repository, then install cuda-toolkit-12-4.
- Create mamba env 'cinebot' with python 3.10, install pytorch/torchvision/torchaudio with CUDA 12.1 builds.
- Run a torch.cuda.is_available() smoke test after installation.

## 0.5 Automation TODOs
- Author scripts/setup_wsl_env.sh to encapsulate prerequisite checks and optional installs.
- Capture environment variables and Omniverse launch commands for Isaac Sim headless mode.
- Document proxy considerations hinted by WSL NAT warning and align with network configuration.

Last updated (UTC): 2025-09-22 16:25:16

## 0.6 Isaac Sim Installation Audit
- 2025-09-22: nvcc now reports CUDA 12.6.85 after installing NVIDIA repo toolkit.
- 2025-10-13: Reconfirmed nvcc visibility from WSL shell (`nvcc --version` -> CUDA 12.6.85).
- Windows installation detected at /mnt/i/isaacsim (Isaac Sim 5.0.0-rc.45) with .bat launchers.
- Asset libraries located under /mnt/i/isaacsim_assets and /mnt/i/OmniAssets (shared Omniverse cache).
- No Linux shell launchers present; a separate WSL/Linux installation will be required for native WSL execution.
- Windows CUDA toolkit available (nvcc 12.6 via cmd.exe). WSL currently lacks cuda-toolkit packages.

- Venv bootstrap script available at scripts/setup_rl_venv.sh (derives `.venv_rl<pymajor><pyminor>` from the selected interpreter and installs CUDA-enabled PyTorch).

### Proposed Environment Variables
- Add ISAACSIM_WIN_ROOT=/mnt/i/isaacsim for cross-referencing the Windows install from WSL scripts.
- When Linux build installed, plan to set ISAACSIM_ROOT inside WSL (e.g., /home/yanbo/isaacsim) and map assets via OV_ASSETS_ROOT=/mnt/i/isaacsim_assets/Assets.
## 0.7 Isaac Lab Windows bring-up (2025-09-23)
- Using Isaac Sim kit at `I:\isaacsim` (5.0.0-rc.45) as runtime; `I:\isaaclab\_isaac_sim` junction added for helper scripts.
- Editable installs completed: `isaaclab`, `isaaclab_tasks`, `isaaclab_assets`, `isaaclab_rl[rl-games]` against bundled Python 3.11.13 with torch 2.7.0+cu128; extra deps pulled (ray 2.49.2, rl-games python3.11 branch, gym 0.23.1, opencv-python 4.11.0.86).
- Persistent env vars set in Windows session: `OV_ASSETS_ROOT=I:\isaacsim_assets`, `ISAACSIM_PATH=I:\isaacsim`.
- Smoke test command: `isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Cartpole-Direct-v0 --max_iterations 10 --headless`.
  - Run completed in ~40 s, exercised RTX 3090 (CUDA device 0), and wrote checkpoints to `I:\isaaclab\logs\rl_games\cartpole_direct`.
  - Warnings observed: Warp reported missing `cuDeviceGetUuid` API (driver limitation) but training proceeded; gym deprecation notices expected; no fatal errors.
- Next steps: capture networking plan for Windows↔WSL ROS2 bridge and continue with robot spec + task schema scaffolding.

-## 0.8 Windows↔WSL networking plan for ROS 2 (2025-09-23)
- Goal: keep Isaac Sim/Lab on Windows while allowing ROS 2 nodes in WSL2 (Ubuntu 22.04) to exchange topics/services with the Windows stack and, later, the Jetson.
- DDS vendor: stick with Fast DDS (default for Isaac Sim/ROS 2 Humble). Set consistent `ROS_DOMAIN_ID=55` (free ID) on both Windows and WSL to avoid collisions with other robots on the LAN.
- Windows side actions:
  - Create inbound/outbound firewall rules allowing UDP 7400-7500, 7410-7420, and 8800 (Fast DDS discovery/data) for both `kit.exe` and `ros2.exe` executables.
  - If Windows Defender is active, add `I:\isaacsim` and the WSL distro IP range (default 172.16.0.0/12) to the allowed networks list.
  - Optional: use `netsh interface portproxy add v4tov4 listenport=7600 listenaddress=127.0.0.1 connectport=7600 connectaddress=<WSL_IP>` if Rosbridge/other TCP ports are required.
  - VPN/Proxy note: Clash is running on Windows. Add bypass rules for UDP 7400-7500/8800 and the WSL subnet so Fast DDS traffic does not get captured. Confirm with `netsh winhttp show proxy` and Clash UI that `*.local`, `172.16.0.0/12`, and the Jetson IP range are in the DIRECT list.
- Helper script: run `scripts\networking\configure_fastdds_firewall.ps1` (elevated PowerShell) to create firewall rules and append Clash bypass entries.
- 2025-09-23: Script executed successfully; rules added for UDP 7400-7410/7420/8800 and Clash config updated. Restart Clash to apply changes.
- WSL side actions:
  - Expose the Windows host IP as `WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')` and export `FASTDDS_DEFAULT_PROFILES_FILE=~/fastdds_windows.xml` with `<udp_transport>shared-memory off</udp_transport>` to avoid shared-memory attempts.
  - Add `/etc/hosts` entries mapping `win-host` to the Windows IP and `wsl-host` to the WSL IP (from `hostname -I`).
  - Set `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and mirror the `ROS_DOMAIN_ID=55` export in the WSL ROS 2 environment setup script.
  - VPN/Proxy note: Mihomo is active in WSL. Ensure the transparent proxy excludes ROS 2 UDP ports and the Windows host IP (`ipset`/ACL entry). Validate with `ss -ulpn` that packets are flowing directly rather than via tun interface.
- Helper script: `scripts/networking/configure_fastdds_wsl.sh [windows_ip]` generates the Fast DDS profile and appends Mihomo bypass entries.
- 2025-09-23: Script run inside WSL. Produced `/home/yanbo/fastdds_windows.xml`, detected `WIN_IP=10.255.255.254`, `WSL_IP=172.22.61.27`, and printed export instructions.
- 2025-10-13: Windows listener (`ros2 run demo_nodes_cpp listener`) and WSL talker (`ros2 run demo_nodes_cpp talker`) exchanged `/chatter` for >10 s; listener log captured `Hello World: 427` through `Hello World: 441`.
- Verification checklist:
  1. From Windows: run `ros2 topic list --spin-time 5` while WSL publishes a dummy topic using `ros2 topic pub /wsldemo std_msgs/String 'data: WSL'`.
  2. From WSL: confirm Isaac Sim `/isaac_sim/camera_info` (or any omni topic) is discoverable after launching the simulator with `--/exts/ros2_bridge/useDomainID=55`.
  3. Capture IP addresses and any firewall adjustments in this log after first successful round-trip test.
  4. If ROS 2 binaries are not yet installed, perform a `fastdds discovery --nodes` test from both sides to ensure traffic passes before installing ROS.
- 2025-09-24: ROS 2 talker/listener smoke test successful (WSL `ros2 run demo_nodes_cpp talker` ↔ Windows `py -3.8 "%ROS2%\Scripts\ros2-script.py" run demo_nodes_cpp listener`); `/chatter` flowing across domain 55.

## 0.9 ROS 2 Humble on Windows (2025-09-23)
- Installed binary release `ros2-release-humble-20250721-windows-release-amd64.msi` to `C:\dev\ros2_humble` (download via GitHub releases).
- Environment helper: `scripts/networking/setup_ros2_humble_windows.ps1` sets `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, optional `FASTDDS_DEFAULT_PROFILES_FILE`, then calls `local_setup.bat`.
- Next: open a new PowerShell, run the helper script, and launch demo nodes to validate Windows↔WSL communication with the same domain ID.
- ROS build deps (Chocolatey): installed `asio`, `bullet`, `cunit`, `tinyxml-usestl`, `tinyxml2` from local cache (`C:\Users\yanbo\Downloads`); `eigen` already present (3.4.0.20240224).

