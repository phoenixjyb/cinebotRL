@echo off
REM Launcher for cinebotRL training on G:\
REM Usage: run_training.bat [num_envs] [total_timesteps]

set ISAAC_LAB_ROOT=G:\isaaclab
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=0

set NUM_ENVS=%1
if "%NUM_ENVS%"=="" set NUM_ENVS=2048
set TIMESTEPS=%2
if "%TIMESTEPS%"=="" set TIMESTEPS=5000000

echo === cinebotRL Training Launcher ===
echo NUM_ENVS=%NUM_ENVS%, TIMESTEPS=%TIMESTEPS%
echo.

call G:\isaaclab_venv\Scripts\activate.bat
cd /d G:\wSpace\cinebotRL
G:\isaaclab_venv\Scripts\python.exe G:\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py ^
  --task MobileMMTrackEE-v0 ^
  --num_envs %NUM_ENVS% ^
  --headless ^
  --total_timesteps %TIMESTEPS%
