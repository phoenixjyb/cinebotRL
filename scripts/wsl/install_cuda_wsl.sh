#!/usr/bin/env bash
# Automates CUDA toolkit installation inside WSL2 (Ubuntu 22.04).
# Requires sudo privileges for apt operations. Set CUDA_VERSION env var to override default (12-6).

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-12-6}"
CUDA_VERSION_DOT="${CUDA_VERSION/-/.}"
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
CUDA_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub"
PIN_DEST="/etc/apt/preferences.d/cuda-repository-pin-600"
REPO_LIST="/etc/apt/sources.list.d/cuda-${CUDA_VERSION}.list"

if ! grep -qi microsoft /proc/version; then
  echo "[ERROR] This script is intended for WSL2 on Windows." >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "[ERROR] sudo not available. Install sudo before running." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found. Ensure NVIDIA drivers with WSL support are installed on Windows host." >&2
fi

ensure_commands() {
  local missing_pkgs=()
  local cmd pkg
  for cmd in "$@"; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      case "${cmd}" in
        gpg)
          pkg="gnupg"
          ;;
        *)
          pkg="${cmd}"
          ;;
      esac
      missing_pkgs+=("${pkg}")
    fi
  done

  if (( ${#missing_pkgs[@]} > 0 )); then
    echo "[INFO] Installing prerequisite packages: ${missing_pkgs[*]}" >&2
    sudo apt-get update
    sudo apt-get install -y "${missing_pkgs[@]}"
  fi
}

ensure_commands curl gpg

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

PIN_FILE="${TMP_DIR}/cuda-ubuntu2204.pin"
curl -fsSL "${CUDA_REPO_URL}/cuda-ubuntu2204.pin" -o "${PIN_FILE}"
sudo install -o root -g root -m 644 "${PIN_FILE}" "${PIN_DEST}"

KEY_RING="/usr/share/keyrings/cuda-archive-keyring.gpg"
curl -fsSL "${CUDA_KEY_URL}" | gpg --dearmor | sudo tee "${KEY_RING}" >/dev/null
echo "deb [signed-by=${KEY_RING}] ${CUDA_REPO_URL} /" | sudo tee "${REPO_LIST}" >/dev/null

sudo apt-get update
sudo apt-get install -y "cuda-toolkit-${CUDA_VERSION}"

CUDA_PATH="/usr/local/cuda-${CUDA_VERSION_DOT}"
BASHRC="${HOME}/.bashrc"
PATH_LINE="export PATH=${CUDA_PATH}/bin:\$PATH"
LD_LINE="export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH"

touch "${BASHRC}"

append_if_missing() {
  local line="$1"
  local file="$2"
  if ! grep -F "${line}" "${file}" >/dev/null 2>&1; then
    echo "${line}" >> "${file}"
  fi
}

if ! grep -F "CUDA ${CUDA_VERSION}" "${BASHRC}" >/dev/null 2>&1; then
  {
    echo ""
    echo "# CUDA ${CUDA_VERSION} (added by install_cuda_wsl.sh on $(date +%Y-%m-%d))"
  } >> "${BASHRC}"
fi
append_if_missing "${PATH_LINE}" "${BASHRC}"
append_if_missing "${LD_LINE}" "${BASHRC}"

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | head -n 1
else
  echo "[WARN] nvcc still not visible. Restart shell or source ~/.bashrc." >&2
fi

cat <<'MSG'
[INFO] CUDA toolkit installation finished.
- Restart this shell or run: source ~/.bashrc
- Validate with: nvcc --version && nvidia-smi
- Install GPU-enabled PyTorch next (e.g., via conda/mamba or pip wheels).
MSG
