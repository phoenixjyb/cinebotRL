#!/usr/bin/env bash
set -euo pipefail

info() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }

info 'Checking GPU visibility'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  warn 'nvidia-smi not found. Install NVIDIA drivers with WSL2 support.'
fi

info 'Checking for CUDA toolkit (nvcc)'
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | head -n 1
elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
  warn 'nvcc not on PATH, but detected at /usr/local/cuda-12.8/bin/nvcc'
  /usr/local/cuda-12.8/bin/nvcc --version | head -n 1
elif [ -x /usr/local/cuda-12.6/bin/nvcc ]; then
  warn 'nvcc not on PATH, but detected at /usr/local/cuda-12.6/bin/nvcc'
  /usr/local/cuda-12.6/bin/nvcc --version | head -n 1
else
  warn 'nvcc not found. Install cuda-toolkit-12.8 (or 12.6): bash scripts/wsl/install_cuda_wsl.sh'
fi

info 'Checking for python3'
if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  warn 'python3 not found. Install Python 3.10 or newer.'
fi

info 'Reminder: install PyTorch with CUDA support once toolkit is ready.'
