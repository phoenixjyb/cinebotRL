# Isaac Sim + Isaac Lab on Windows 11

This guide establishes a reproducible Windows-first environment for Isaac Sim 5.0.0 and Isaac Lab. It assumes an RTX 30/40-series GPU, Windows 11 Pro, and admin access.

## 1. Prepare the system
- Update to the latest NVIDIA **production** driver (Game Ready or Studio). Verify with `nvidia-smi`.
- Install [Python 3.11.x](https://www.python.org/downloads/windows/) (64-bit, Add to PATH). Verify `py -3.11 --version`.
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++" workload for MSVC.
- Optionally install [Git for Windows](https://git-scm.com/download/win) if not already available.

## 2. Create a dedicated virtual environment
```powershell
py -3.11 -m venv C:\\venvs\\isaaclab
C:\\venvs\\isaaclab\\Scripts\\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

> Tip: keep the venv path short to avoid long-path issues.

## 3. Install Isaac Sim 5.0.0 (pip distribution)
```powershell
pip install --extra-index-url https://pypi.nvidia.com \
    isaacsim==5.0.0.0 \
    isaacsim-extscache==2.1.0 \
    isaacsim-runtime==5.0.0.0
```

NVIDIA may update package names; cross-check with the [latest pip instructions](https://docs.omniverse.nvidia.com/isaacsim/latest/install_windows_pip.html).

Set required environment variables for content cache (adapt paths as needed):
```powershell
$env:ISAACSIM_PATH="C:\\Users\\Public\\Documents\\NVIDIA\\IsaacSim-5.0.0"
$env:ISAACSIM_PYTHON="${env:VIRTUAL_ENV}\\Scripts\\python.exe"
$env:OV_ASSETS_ROOT="I:\\isaacsim_assets"
```

Persist these via System Properties → Environment Variables once validated.

## 4. Install Isaac Lab dependencies
```powershell
pip install --extra-index-url https://pypi.nvidia.com isaacsim-assets
pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install hydra-core==1.3.2 wandb tensorboard pandas numpy==1.26.4
```

Clone Isaac Lab into a workspace folder (e.g., `C:\\dev\\IsaacLab`):
```powershell
cd C:\\dev
git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git
cd IsaacLab
pip install -e .
```

If the repo is already mirrored in this project, adjust the path accordingly.

## 5. Smoke tests
1. Launch the simulator headless to compile extensions:
   ```powershell
   python -m isaacsim.headless.native.app --/app/quitAfterExecute=true
   ```
2. Run an Isaac Lab training sanity check:
   ```powershell
   cd C:\\dev\\IsaacLab
   python scripts/rl_games/train.py task=Isaac-VelocityYAML headless=true max_iterations=10
   ```
   Confirm `nvidia-smi` shows GPU load and the script exits cleanly.

## 6. Optional GUI validation
- Activate the same venv and run `python -m isaacsim.app` for GUI testing.
- Load robot assets from the shared `I:\\isaacsim_assets` path (set `OV_ASSETS_ROOT`).

## 7. Coordinate with WSL2 tooling
- Ensure Windows firewall allows UDP traffic for DDS (fastdds default ports 7400-7500).
- In WSL, export matching DDS/domain env vars (e.g., `FASTRTPS_DEFAULT_PROFILES_FILE`).
- Document IP mappings or use `localhost` with `netsh interface portproxy` if needed.

## 8. Maintenance checklist
- Track driver updates; rerun smoke tests after upgrades.
- Keep the venv requirements file (`docs/setup/requirements_windows.txt`, TBD) to lock versions.
- Schedule periodic cleanup of `%LOCALAPPDATA%\\ov\\pkgcache` to reclaim disk space.

Document any deviations or additional steps in `docs/tracking/phase0_environment.md`.
