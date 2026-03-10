# Isaac Sim + Isaac Lab on Windows 11

This guide establishes a reproducible Windows-first environment for Isaac Sim 5.0.0 and Isaac Lab 2.2.0. It assumes an RTX 30/40-series GPU, Windows 11 Pro, and admin access.

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
pip install --extra-index-url https://pypi.nvidia.com `
    isaacsim==5.0.0.0 `
    isaacsim-extscache==2.1.0 `
    isaacsim-runtime==5.0.0.0
```

> **Note on Isaac Sim 5.1.0:** A newer stable release (5.1.0) is available on PyPI. However,
> Isaac Lab 2.2.0 was validated against 5.0.0. Full 5.1.x support is targeted in Isaac Lab 3.0
> (develop branch). Stay on 5.0.0 until Isaac Lab 3.0 releases.
>
> To upgrade Isaac Sim after confirming Isaac Lab compatibility:
> ```powershell
> pip install --extra-index-url https://pypi.nvidia.com isaacsim==5.1.0.0
> ```

NVIDIA may update package names; cross-check with the [latest pip instructions](https://docs.omniverse.nvidia.com/isaacsim/latest/install_windows_pip.html).

Set required environment variables for content cache (adapt paths as needed):
```powershell
$env:ISAACSIM_PATH="C:\\Users\\Public\\Documents\\NVIDIA\\IsaacSim-5.0.0"
$env:ISAACSIM_PYTHON="${env:VIRTUAL_ENV}\\Scripts\\python.exe"
$env:OV_ASSETS_ROOT="I:\\isaacsim_assets"
```

Persist these via System Properties → Environment Variables once validated.

## 4. Install Isaac Lab and dependencies
```powershell
# PyTorch with CUDA 12.8 (matches torch 2.7.0+cu128 — required for Isaac Lab 2.2.0)
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab 2.2.0 (pip package — no git clone needed)
pip install --extra-index-url https://pypi.nvidia.com isaaclab==2.2.0

# Additional dependencies
pip install stable-baselines3>=2.0.0 gymnasium>=1.0.0 numpy>=1.20.0
pip install hydra-core wandb tensorboard pandas
```

> **Note:** Isaac Lab 2.2.0 is distributed as a pip package alongside Isaac Sim 5.0.0.
> It uses the `isaaclab` namespace (not the old `omni.isaac.lab`).

## 5. Smoke tests
1. Launch the simulator headless to compile extensions:
   ```powershell
   python -m isaacsim.headless.native.app --/app/quitAfterExecute=true
   ```
2. Verify Isaac Lab import:
   ```powershell
   python -c "import isaaclab; print('Isaac Lab OK')"
   ```
3. Run an Isaac Lab training sanity check:
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

## 8. Upgrading an existing installation

To upgrade from 5.0.0-rc.x to the stable 5.0.0 release:
```powershell
# Activate the Isaac Lab venv first
C:\\venvs\\isaaclab\\Scripts\\Activate.ps1

pip install --extra-index-url https://pypi.nvidia.com `
    isaacsim==5.0.0.0 `
    isaacsim-extscache==2.1.0 `
    isaacsim-runtime==5.0.0.0
```

After upgrading, run the smoke tests in section 5 to verify.

## 9. Maintenance checklist
- Track driver updates; rerun smoke tests after upgrades.
- Keep a `requirements-lock.txt` at the project root with exact pinned versions.
- Schedule periodic cleanup of `%LOCALAPPDATA%\\ov\\pkgcache` to reclaim disk space.

Document any deviations or additional steps in `docs/tracking/phase0_environment.md`.

