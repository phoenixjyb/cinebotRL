:: ============================================================================
:: ref_to_run_ros2_humble_windows.bat
:: Reference script showing how to activate Windows ROS 2 Humble (Python 3.8).
::
:: TWO installations exist on this machine:
::   I:\ros2\ros2-windows         -- Python 3.8  (VERIFIED WORKING ✅) ← this script uses it
::   I:\ros2humble\ros2-windows   -- Python 3.10 (not yet tested)
::
:: For a full PowerShell launcher that supports both, use:
::   .\scripts\networking\setup_ros2_humble_windows.ps1
::   .\scripts\networking\setup_ros2_humble_windows.ps1 -RosInstall 'I:\ros2humble\ros2-windows'
:: ============================================================================

:: set bundle path
set "ROS2=I:\ros2\ros2-windows"

:: (optional, once) if you haven’t already created this link, run in Admin cmd:
:: mklink /D C:\Python38 "C:\Program Files\Python38"

:: source environment
call "%ROS2%\local_setup.bat"

:: make sure ROS 2 DLLs are found
set "PATH=%ROS2%\bin;%PATH%"

:: add Python packages path
set "PYTHONPATH=%ROS2%\Lib\site-packages;%PYTHONPATH%"

:: prefer Fast-DDS; keep logging simple (no spdlog dependency)
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set RCL_LOGGING_IMPLEMENTATION=rcl_logging_noop

:: sanity
py -3.8 -c "import sys, rclpy; print('rclpy OK'); print('Python:',sys.executable)"

:: start listener in a new window
start "ROS2 listener" cmd /k "call ""%ROS2%\local_setup.bat"" && set PATH=%ROS2%\bin;%%PATH%% && set PYTHONPATH=%ROS2%\Lib\site-packages;%%PYTHONPATH%% && set RMW_IMPLEMENTATION=rmw_fastrtps_cpp && set RCL_LOGGING_IMPLEMENTATION=rcl_logging_noop && py -3.8 ""%ROS2%\Scripts\ros2-script.py"" run demo_nodes_py listener"

:: run talker here
py -3.8 "%ROS2%\Scripts\ros2-script.py" run demo_nodes_cpp talker
