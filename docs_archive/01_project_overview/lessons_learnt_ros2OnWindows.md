# Lessons Learned: ROS 2 Humble on Windows

## Timeline Highlights
- **Initial ZIP** lacked an embedded Python runtime. `local_setup.bat` worked, but `ros2`/`rclpy` failed because the system Python layout didn’t match what the ZIP expected.
- **Python Path Fix**: created a `C:\Python38` junction pointing to `C:\Program Files\Python38` so tools that assume the classic location could find the interpreter.
- **Missing `_rclpy_pybind11`**: Older bundles omitted the compiled extension. Switching to a bundle that shipped `_rclpy_pybind11.cp310-win_amd64.pyd` exposed new DLL dependencies instead.
- **DDS Dependencies**: Added Fast-DDS/Fast-CDR/foonathan DLLs to `PATH`, installed `console_bridge`/`yaml-cpp` via vcpkg, and used `RCL_LOGGING_IMPLEMENTATION=rcl_logging_noop` to avoid `spdlog.dll`.
- **Working Bundle (2024-08-07)**: Located in `I:\ros2\ros2-windows`; includes `_rclpy_pybind11.cp38-win_amd64.pyd`. Running with Python 3.8 finally allowed the talker/listener demos to succeed.
- **2025-10-13**: Re-ran the cross-host demo; Windows listener logged `Hello World: 427` through `Hello World: 441` while WSL talker published the sequence over Fast DDS domain 55.

## Root Causes
| Issue | Cause | Fix |
| ----- | ----- | --- |
| `ModuleNotFoundError: _rclpy_pybind11` | Bundle missing `.pyd` or wrong Python ABI | Use bundle whose suffix (`cp38`/`cp310`) matches the interpreter |
| `ImportError: DLL load failed` | Dependent DDS DLL not on `PATH` | Ensure `fastrtps-*.dll`, `fastcdr-*.dll`, `foonathan_memory-*.dll`, `console_bridge.dll`, `yaml-cpp.dll`, `tinyxml2.dll` are present |
| Logging failure (`spdlog.dll`) | Default logging backend requires spdlog | Switch to `rcl_logging_noop` or install spdlog |
| RTI warning | Connext DDS not installed | Ignore when using Fast-DDS |

## Final Working Setup
- **Bundle**: `I:\ros2\ros2-windows` (cp38 build, no embedded Python).
- **Python**: System 3.8 (`C:\Program Files\Python38`), reachable via optional `C:\Python38` link.
- **DDS**: Fast-DDS with supporting DLLs on `PATH`.
- **Logging**: Neutral backend via `RCL_LOGGING_IMPLEMENTATION=rcl_logging_noop`.

## Daily Workflow
### Windows
```
set ROS2=I:\ros2\ros2-windows
call "%ROS2%\local_setup.bat"
set PATH=%ROS2%\bin;%PATH%
set PYTHONPATH=%ROS2%\Lib\site-packages;%PYTHONPATH%
set ROS_DOMAIN_ID=55
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set RCL_LOGGING_IMPLEMENTATION=rcl_logging_noop
py -3.8 -c "import sys, rclpy; print('rclpy OK from', sys.executable)"
start "ROS2 listener" cmd /k "call "%ROS2%\local_setup.bat" && set PATH=%ROS2%\bin;%%PATH%% && set PYTHONPATH=%ROS2%\Lib\site-packages;%%PYTHONPATH%% && set ROS_DOMAIN_ID=55 && set RMW_IMPLEMENTATION=rmw_fastrtps_cpp && set RCL_LOGGING_IMPLEMENTATION=rcl_logging_noop && py -3.8 "%ROS2%\Scripts\ros2-script.py" run demo_nodes_py listener"
py -3.8 "%ROS2%\Scripts\ros2-script.py" run demo_nodes_cpp talker
```

### WSL
```
cd /mnt/c/Users/yanbo/wSpace/cinebotRL/scripts/networking
./configure_fastdds_wsl.sh
export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/fastdds_windows.xml
ros2 run demo_nodes_cpp talker
```

## Quick CLI Checks
```
ros2 topic list
ros2 node list
ros2 topic echo /chatter
```

## Picking the Right Python
- Inspect `Lib/site-packages/rclpy/_rclpy_pybind11.*.pyd`:
  - `…cp38…` → use Python 3.8
  - `…cp310…` → use Python 3.10
- Prefer the bundle’s own `python.exe` if present; otherwise use your system interpreter matching the suffix.

## Troubleshooting
| Symptom | Action |
| ------- | ------ |
| `ModuleNotFoundError: rclpy._rclpy_pybind11` | Ensure the `.pyd` exists and the Python version matches |
| `ImportError/DLL load failed` | Verify required DLLs with `where` commands listed above |
| `spdlog.dll` missing | Use `rcl_logging_noop` or install spdlog |
| RTI warning | Safe to ignore when you rely on Fast-DDS |
| Bitness mismatch | `py -3.8 -c "import struct; print(struct.calcsize('P')*8)"` should print `64` |
| Virtual env interference | Avoid activating venv/conda in this shell |

## Summary
- Always match the Python ABI indicated by `_rclpy_pybind11`.
- Keep DDS dependencies on `PATH`; vcpkg is handy for missing DLLs.
- Fast-DDS (`rmw_fastrtps_cpp`) plus `rcl_logging_noop` yields the leanest setup.
- Windows listener + WSL talker exchange `/chatter` on domain 55 once the environment is aligned.
