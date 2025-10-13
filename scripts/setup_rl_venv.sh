#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: setup_rl_venv.sh [--python PYTHON] [--name VENV_NAME]

Options:
  --python PYTHON   Path to python interpreter (default: python3)
  --name NAME       Virtualenv directory name (default: .venv_rl<python_major><python_minor>)
USAGE
}

PYTHON=python3
VENV_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --name)
      VENV_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[ERROR] Python interpreter not found: $PYTHON" >&2
  exit 1
fi

if [[ -z "$VENV_NAME" ]]; then
  if ! VENV_NAME=$("$PYTHON" - <<'PYCODE'
import sys
print(f".venv_rl{sys.version_info.major}{sys.version_info.minor}")
PYCODE
  ); then
    echo "[ERROR] Failed to derive default venv name from $PYTHON" >&2
    exit 1
  fi
fi

VENV_PATH="/mnt/c/Users/yanbo/wSpace/cinebotRL/$VENV_NAME"
echo "[INFO] Creating venv at $VENV_PATH"
"$PYTHON" -m venv "$VENV_PATH"

ACTIVATE="$VENV_PATH/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
  echo "[ERROR] Virtualenv activation script not found: $ACTIVATE" >&2
  exit 1
fi

echo "[INFO] Activating virtualenv"
# shellcheck disable=SC1090
source "$ACTIVATE"

echo "[INFO] Upgrading pip"
python -m pip install --upgrade pip

CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu121"

echo "[INFO] Installing PyTorch (CUDA 12.1 wheels)"
python -m pip install --extra-index-url "$CUDA_WHL_INDEX" torch torchvision torchaudio

echo "[INFO] Installing auxiliary packages"
python -m pip install gymnasium[all] numpy pandas jupyter

echo "[INFO] Writing environment activation helper"
cat <<ACT > "$VENV_PATH/activate_rl.sh"
#!/usr/bin/env bash
source "$VENV_PATH/bin/activate"
ACT
chmod +x "$VENV_PATH/activate_rl.sh"

echo "[INFO] Virtual environment ready. Activate with: source $VENV_PATH/bin/activate"



