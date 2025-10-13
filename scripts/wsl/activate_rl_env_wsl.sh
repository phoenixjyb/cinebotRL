#!/usr/bin/env bash
# Helper to activate the WSL RL virtual environment with CUDA paths wired.
# Usage: source scripts/wsl/activate_rl_env_wsl.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_VENV_NAME=".venv_rl311"
VENV_NAME=${RL_VENV_NAME:-$DEFAULT_VENV_NAME}
VENV_PATH="$ROOT_DIR/$VENV_NAME"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  if [[ -z ${RL_VENV_NAME:-} ]] && [[ -f "$ROOT_DIR/.venv_rl/bin/activate" ]]; then
    cat <<EOF >&2
[ERROR] Expected virtualenv at $VENV_PATH
[HINT] Legacy env detected at $ROOT_DIR/.venv_rl. Recreate the 3.11 env with scripts/setup_rl_venv.sh or export RL_VENV_NAME=.venv_rl before sourcing this helper.
EOF
  else
    echo "[ERROR] Expected virtualenv at $VENV_PATH" >&2
  fi
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda-12.6}
if [[ -d "$CUDA_PREFIX/lib64" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CUDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
  export PATH="${CUDA_PREFIX}/bin:${PATH}"
else
  echo "[WARN] CUDA prefix $CUDA_PREFIX missing lib64; skipping PATH/LD_LIBRARY_PATH exports" >&2
fi

echo "[INFO] Activated $VENV_NAME (Python $(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))" >&2
python - <<'PY'
import torch
try:
    cuda_ok = torch.cuda.is_available()
    msg = f"torch {torch.__version__} cuda={cuda_ok}"
    if cuda_ok:
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        msg += f" devices={devices}"
    print(msg)
except Exception as exc:
    print(f"torch check failed: {exc}")
PY
