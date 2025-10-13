#!/usr/bin/env bash
# Installs the Linux headless build of Isaac Sim inside WSL2.
# Download the official Isaac Sim .sh installer ahead of time (e.g. via Windows Omniverse)
# and make it accessible from WSL (default search: /mnt/i/isaacsim/installers).

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: install_isaacsim_headless_wsl.sh [OPTIONS]

Options:
  --version VERSION         Isaac Sim version (default: 2023.1.1)
  --installer PATH          Path to Isaac Sim *.sh installer (overrides search)
  --install-root PATH       Destination directory (default: $HOME/.local/share/ov/pkg)
  --assets-root PATH        Shared asset mount (default: /mnt/i/isaacsim_assets)
  --force                   Reinstall even if target already exists
  --dry-run                 Print commands without executing installer
  -h, --help                Show this help message

Environment overrides:
  ISAACSIM_VERSION, ISAACSIM_INSTALLER, ISAACSIM_INSTALL_ROOT, ISAACSIM_ASSETS

The installer must run inside WSL2 with GPU passthrough enabled and CUDA already configured.
USAGE
}

require_wsl() {
  if ! grep -qi microsoft /proc/version; then
    echo "[ERROR] Not running inside WSL2." >&2
    exit 1
  fi
}

check_gpu_access() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Install Windows NVIDIA drivers with WSL support first." >&2
    exit 1
  fi
}

ensure_prereqs() {
  local packages=(
    build-essential
    python3
    python3-venv
    python3-pip
    libvulkan1
    libegl1
    libgl1
    libxkbcommon0
    libxrandr2
    libxinerama1
    libxcursor1
    libxi6
    libnss3
    libatk1.0-0
    libatk-bridge2.0-0
    libgdk-pixbuf-2.0-0
    libgtk-3-0
    libcanberra-gtk3-module
    libffi-dev
    libssl-dev
    zlib1g-dev
    unzip
    rsync
  )
  echo "[INFO] Checking prerequisite packages" >&2
  sudo apt-get update
  sudo apt-get install -y "${packages[@]}"
}

resolve_installer() {
  if [[ -n "$ISAACSIM_INSTALLER" ]]; then
    INSTALLER="$ISAACSIM_INSTALLER"
  elif [[ -n "$INSTALLER_ARG" ]]; then
    INSTALLER="$INSTALLER_ARG"
  else
    local candidates=(\
      "/mnt/i/isaacsim/installers/isaac-sim-${ISAACSIM_VERSION}-linux-x86_64.sh"\
      "/mnt/i/isaacsim/installers/isaac-sim-${ISAACSIM_VERSION}.sh"\
      "/mnt/i/isaacsim/isaac-sim-${ISAACSIM_VERSION}-linux-x86_64.sh"\
      "$HOME/Downloads/isaac-sim-${ISAACSIM_VERSION}-linux-x86_64.sh")
    for c in "${candidates[@]}"; do
      if [[ -f "$c" ]]; then
        INSTALLER="$c"
        break
      fi
    done
  fi

  if [[ -z "${INSTALLER:-}" ]]; then
    echo "[ERROR] Isaac Sim installer not found." >&2
    echo "  Provide --installer PATH or set ISAACSIM_INSTALLER." >&2
    exit 1
  fi
}

create_install_root() {
  mkdir -p "$INSTALL_ROOT"
}

run_installer() {
  local target_dir="$INSTALL_ROOT/isaac-sim-${ISAACSIM_VERSION}"
  if [[ -d "$target_dir" && "$FORCE" == "0" ]]; then
    echo "[INFO] Isaac Sim ${ISAACSIM_VERSION} already appears installed at $target_dir" >&2
    ISAACSIM_TARGET="$target_dir"
    return
  fi

  echo "[INFO] Running installer $INSTALLER" >&2
  chmod +x "$INSTALLER"
  rm -rf "$target_dir"
  mkdir -p "$target_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] "$INSTALLER" --skip-license --target "$target_dir"" >&2
    ISAACSIM_TARGET="$target_dir"
    return
  fi

  "$INSTALLER" --skip-license --target "$target_dir"

  echo "[INFO] Isaac Sim installed under $target_dir" >&2
  ISAACSIM_TARGET="$target_dir"
}

write_env_file() {
  local target_dir="${ISAACSIM_TARGET:-$INSTALL_ROOT/isaac-sim-${ISAACSIM_VERSION}}"
  local env_file="$HOME/.config/isaac-sim-wsl.env"
  mkdir -p "$(dirname "$env_file")"
  local python_sh="$target_dir/python.sh"
  cat <<ENV >"$env_file"
export ISAACSIM_VERSION=${ISAACSIM_VERSION}
export ISAACSIM_ROOT=${target_dir}
export ISAACSIM_PYTHON=${python_sh}
export ISAACSIM_ASSETS=${ISAACSIM_ASSETS}
export LD_LIBRARY_PATH=\$ISAACSIM_ROOT/exts/omni.isaac.sim/bin:\$ISAACSIM_ROOT/exts/omni.isaac.sim.python/bin:\$LD_LIBRARY_PATH
ENV
  echo "[INFO] Wrote helper environment file $env_file" >&2
}

link_assets() {
  if [[ -d "$ISAACSIM_ASSETS" ]]; then
    local target="${ISAACSIM_TARGET:-$INSTALL_ROOT/isaac-sim-${ISAACSIM_VERSION}}/shared_assets"
    ln -sfn "$ISAACSIM_ASSETS" "$target"
    echo "[INFO] Linked shared assets: $target -> $ISAACSIM_ASSETS" >&2
  else
    echo "[WARN] Assets root $ISAACSIM_ASSETS not found; skipping link." >&2
  fi
}

main() {
  require_wsl
  check_gpu_access

  ISAACSIM_VERSION="${ISAACSIM_VERSION:-2023.1.1}"
  INSTALL_ROOT="${ISAACSIM_INSTALL_ROOT:-$HOME/.local/share/ov/pkg}"
  ISAACSIM_ASSETS="${ISAACSIM_ASSETS:-/mnt/i/isaacsim_assets}"
  FORCE=0
  DRY_RUN=0
  INSTALLER_ARG=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --version)
        ISAACSIM_VERSION="$2"
        shift 2
        ;;
      --installer)
        INSTALLER_ARG="$2"
        shift 2
        ;;
      --install-root)
        INSTALL_ROOT="$2"
        shift 2
        ;;
      --assets-root)
        ISAACSIM_ASSETS="$2"
        shift 2
        ;;
      --force)
        FORCE=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
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

  ensure_prereqs
  resolve_installer
  create_install_root
  run_installer
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Skipping environment file and asset linking." >&2
    exit 0
  fi
  write_env_file
  link_assets

  local python_hint="${ISAACSIM_TARGET:-$INSTALL_ROOT/isaac-sim-${ISAACSIM_VERSION}}/python.sh"
  cat <<NEXT
[INFO] Isaac Sim headless installation complete.
Next steps:
  - source ~/.config/isaac-sim-wsl.env to load environment variables.
  - Run scripts/check_isaacsim_headless.sh for a basic health check.
  - Use Isaac Sim python executable at: ${python_hint}
NEXT
}

main "$@"
