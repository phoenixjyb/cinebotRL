# Copilot instructions for CinebotRL

This file gives short, actionable context for AI coding agents to be productive in this repository.

Key facts (read before editing code)
- Primary training runs on Windows with Isaac Lab / Isaac Sim (path: `I:\isaaclab`). WSL2 is optional and used only for ROS2 monitoring and offline analysis.
- Do NOT import `task_spec` or IsaacLab-specific modules at top-level of long-running scripts; Isaac Sim must be initialized first. See `scripts/reinforcement_learning/sb3/train.py` where `AppLauncher` is created before `register_isaac_lab_tasks()` is imported and called.
- Training is launched via `scripts\launch_training_windows.ps1` (preferred) which runs `I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py ...`.

Important files & where to look for patterns
- `README.md` — high-level architecture, quick-start commands, Windows-first guidance.
- `scripts/reinforcement_learning/sb3/train.py` — canonical CLI flags, TF32 + cuDNN enabling, device selection, logging, callbacks (entropy decay, AdaptiveKL), and the `IsaacLabToSB3VecEnvWrapper` (converts dict/torch observations → numpy arrays for SB3).
- `scripts\launch_training_windows.ps1` — exact PowerShell flags used by humans; reproducing these flags keeps CLI parity.
- `src/task_spec.py` — centralised robot task spec (joint order, limits, observation/action schema) and `register_isaac_lab_tasks()` example.
- `trajectoryToLearn/`, `chassis_required_indices.txt`, and `chassis_required_trajectories.txt` — trajectory loading/filtering conventions; `--use_chassis_only` reads `chassis_required_indices.txt` to filter trajectories.
- `pyproject.toml` — formatting (Black, isort) and typing/dev deps (pytest, mypy). Use these for style/CI adjustments.

Conventions and patterns to follow (concrete)
- Windows-first: prefer PowerShell launchers under `scripts/` for training or reproduce the exact `isaaclab.bat -p ...` call when building automation.
- Isaac Lab init order: always initialize AppLauncher (or run via `isaaclab.bat`) before importing task modules. If adding new scripts, include the pattern from `train.py` (AppLauncher then import/register tasks).
- Observation/action conversions: Isaac Lab returns dicts and torch tensors. Use or extend `IsaacLabToSB3VecEnvWrapper` semantics: convert tensors to numpy via `.cpu().numpy()` and update observation_space after first reset.
- Logging & checkpoints: default log path is `logs/sb3/{task}/{timestamp}`. Checkpoints saved under `checkpoints/` with `save_freq // num_envs` in `train.py`. Preserve this path convention.
- GPU/device handling: training code auto-selects CUDA device based on capability and enables TF32 and cuDNN benchmark. Follow this pattern when adding device-aware code.

Common quick examples to follow (copy-paste-safe)
- Launch training (preferred):
  .\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 64 -Headless
- Direct Isaac Lab call (matches `launch_training_windows.ps1`):
  I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py --task MobileMMTrackEE-v0 --num_envs 64 --headless

Gotchas and explicit guards
- Gymnasium plugin crash on Windows: launcher disables plugin entrypoints via env var `GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS=1` — keep this for Windows runs to avoid `ale_py` issues.
- Do not assume observations are numpy arrays. Handle dict obs and both old/new Gymnasium reset/step signatures (see wrappers in `train.py`).
- Trajectory selection: `--trajectory_dir` is often relative; `train.py` resolves it against project root. Use absolute paths when adding tests to avoid accidental path resolution bugs.

Where to add tests and how to run them
- Unit tests use pytest (dev extras in `pyproject.toml`). Put new unit tests under `tests/` following project layout. For quick env sanity, `scripts\test_mobile_mm_env.py` is invoked by the PowerShell launcher when `-Test` is passed.

If you change runtime behaviour, update docs
- If you add new CLI flags, training callbacks, or change checkpoint locations, update `README.md` and `docs/README.md` (and any relevant `docs/` pages) so humans and automation stay in sync.

If unsure, inspect these files first: `README.md`, `scripts/reinforcement_learning/sb3/train.py`, `scripts/launch_training_windows.ps1`, `src/task_spec.py`, `trajectoryToLearn/`.

Questions or missing context? Ask the repo owner which Isaac Lab/Sim versions and absolute paths should be assumed for CI or test runners before changing launchers.
