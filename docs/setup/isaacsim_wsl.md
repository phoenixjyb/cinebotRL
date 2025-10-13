# Isaac Sim Headless on WSL2

This workflow keeps the Windows GUI build of Isaac Sim for authoring while driving training and automation from the Linux headless build under WSL2.

## Prerequisites
- Windows 11 with the latest NVIDIA driver that exposes WSL GPU support.
- Isaac Sim GUI (Windows) already installed via Omniverse Launcher (e.g. version 5.0.0-rc.45) with tooling downloads stored on drive `I:`.
- WSL2 Ubuntu 22.04 instance with CUDA verified (`nvidia-smi`, `nvcc --version`).
- Shared Omniverse/asset drive mounted in WSL as `/mnt/i`.

## Install Steps
1. Ensure the official Isaac Sim Linux installer (`isaac-sim-<VERSION>-linux-x86_64.sh`) is downloaded on the Windows side. The script searches `I:\isaacsim\installers\` by default.
2. In WSL, run:

   ```bash
   cd /mnt/c/Users/yanbo/wSpace/cinebotRL
   ./scripts/install_isaacsim_headless_wsl.sh \
     --version 2023.1.1 \
     --assets-root /mnt/i/isaacsim_assets
   ```

   Optional flags:
   - `--installer /path/to/installer.sh` if the default search paths do not apply.
   - `--install-root /data/ov/pkg` to change the installation directory.
   - `--force` to reinstall over an existing copy.
   - `--dry-run` to preview actions without unpacking.

3. Source the generated environment helper:

   ```bash
   source ~/.config/isaac-sim-wsl.env
   ```

   This exposes `ISAACSIM_ROOT`, `ISAACSIM_PYTHON`, `ISAACSIM_ASSETS`, and augments `LD_LIBRARY_PATH` for local sessions.

4. Validate the install using the health check script:

   ```bash
   ./scripts/check_isaacsim_headless.sh
   ```

   The check spins a minimal headless world for a few simulation steps and prints GPU telemetry via `nvidia-smi`.

## Shared Assets
The installer links `${ISAACSIM_ROOT}/shared_assets` to the configured asset root (default `/mnt/i/isaacsim_assets`). Keep large USDs and textures on the shared Windows drive so both GUI and headless builds reference identical data.

Set additional Omniverse variables as needed in your shell profile:

```bash
export OV_ASSETS_ROOT=${ISAACSIM_ASSETS}
export ISAAC_SIM_ASSETS=${ISAACSIM_ASSETS}
```

## Troubleshooting
- **Installer not found**: Pass `--installer` or copy the `.sh` bundle into an accessible path. The script does not download packages automatically.
- **Missing libraries**: The script installs common dependencies (`libvulkan1`, `libegl1`, GTK, etc.). If specific modules are still missing, rerun the script with `--force` after confirming internet connectivity to `apt`.
- **GPU unavailable**: Reboot Windows, verify the NVIDIA driver exposes WSL GPU, and confirm `nvidia-smi` succeeds from WSL prior to running Isaac Sim.
- **Headless launch fails**: Check `/tmp/isaac-sim-headless.log` (set `ISAAC_SIM_LOG_DIR`) and ensure the shared assets mount is available.

## Next Actions
- Install Isaac Lab or other Omniverse extensions under `${ISAACSIM_ROOT}/exts` as required for RL training.
- Integrate the environment helper into your RL virtual environment activation (e.g., source the env file inside `setup_rl_venv.sh`).
- Capture health check outputs in CI (WSL GitHub Actions runner) when the environment is containerized.
