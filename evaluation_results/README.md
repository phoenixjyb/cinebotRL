# Evaluation Results Directory

This directory contains comprehensive evaluation results for trained policies, organized by model training session.

## Directory Structure

```
evaluation_results/
├── <model_folder_name>/        # e.g., 20251028_200923
│   ├── eval_summary_<timestamp>.json  # Complete statistics
│   ├── episodes_<timestamp>.csv        # Per-episode data
│   ├── steps_<timestamp>.csv           # Time-series step data
│   ├── arrays_<timestamp>.npz          # Raw numpy arrays
│   ├── ANALYSIS_REPORT.md              # Comprehensive analysis
│   └── README.md                        # Model-specific summary
└── README.md                            # This file
```

## Model Folders

Each model folder is named after its training session (e.g., `20251028_200923` corresponds to `logs/sb3/mobilemmtrackee_v0/20251028_200923/`).

### Available Models

- **`20251028_200923/`** - Session 7d (200M timesteps)
  - Training duration: 15.3 hours
  - Final model: `logs/sb3/mobilemmtrackee_v0/20251028_200923/final_model.zip`
  - Evaluation: 200 episodes, 1,038 trajectories
  - Status: ⚠️ **NOT READY FOR DEPLOYMENT**
  - Key findings: Position error 3.64m (18× worse than target), orientation error 140.7° (14× worse)
  - Next steps: Retrain with fixed reward weights

## File Types

### JSON Summary (`eval_summary_<timestamp>.json`)
Complete evaluation statistics including:
- Tracking errors (position & orientation)
- Joint statistics (angles, velocities)
- Base velocity statistics
- Reward component breakdown
- Episode statistics

### CSV Files
- **`episodes_<timestamp>.csv`**: One row per episode (episode reward, length, success)
- **`steps_<timestamp>.csv`**: Time-series data sampled every N steps

### NPZ Arrays (`arrays_<timestamp>.npz`)
Raw numpy arrays for custom analysis:
- `tracking_errors_pos`: Position errors [N, 3] in meters
- `tracking_errors_ori`: Orientation errors [N] in radians
- `joint_positions`: Joint angles [N, 6] in radians
- `joint_velocities`: Joint velocities [N, 6] in rad/s
- `base_velocities`: Base velocities [N, 3] (vx, vy, ωz)
- `reward_components`: Dict of reward component arrays

### Analysis Report (`ANALYSIS_REPORT.md`)
Comprehensive 2000+ word analysis including:
- Executive summary with overall assessment
- Detailed metrics breakdown
- Root cause analysis
- Recommendations for improvement
- Success criteria for next iteration

## Usage

### Running Evaluation

```powershell
# Evaluate a trained model
.\scripts\launch_evaluation_quantitative.ps1 `
    -Checkpoint "logs\sb3\mobilemmtrackee_v0\20251028_200923\final_model.zip" `
    -NumEpisodes 200 `
    -NumEnvs 64 `
    -Headless

# Results will be saved to: evaluation_results/<model_folder>/
```

### Generating Plots

```powershell
# Generate visualization plots
python scripts/reinforcement_learning/sb3/visualize_eval_results.py `
    --input evaluation_results/20251028_200923/eval_summary_<timestamp>.json `
    --output_dir evaluation_plots

# Plots will be saved to: evaluation_plots/<model_folder>/
```

### Loading Data for Custom Analysis

```python
import json
import numpy as np
import pandas as pd

# Load summary statistics
with open('evaluation_results/20251028_200923/eval_summary_<timestamp>.json') as f:
    summary = json.load(f)

# Load episode data
episodes_df = pd.read_csv('evaluation_results/20251028_200923/episodes_<timestamp>.csv')

# Load raw arrays
arrays = np.load('evaluation_results/20251028_200923/arrays_<timestamp>.npz')
pos_errors = arrays['tracking_errors_pos']  # [N, 3]
joint_vels = arrays['joint_velocities']     # [N, 6]
```

## Success Criteria

Policies are evaluated against these criteria:

### Minimum for Deployment
- ✅ Mean position error: **< 20 cm**
- ✅ Median position error: **< 10 cm**
- ✅ P95 position error: **< 50 cm**
- ✅ Mean orientation error: **< 10°**
- ✅ Velocity penalty: **< 2.0**
- ✅ Jerk penalty: **< 1.0**
- ✅ Success rate: **> 75%**

### Good Performance
- ⭐ Mean position error: **< 10 cm**
- ⭐ Mean orientation error: **< 5°**
- ⭐ P95 position error: **< 20 cm**
- ⭐ Success rate: **> 90%**

## Related Files

- **Evaluation script**: `scripts/reinforcement_learning/sb3/evaluate_quantitative.py`
- **Visualization script**: `scripts/reinforcement_learning/sb3/visualize_eval_results.py`
- **PowerShell launcher**: `scripts/launch_evaluation_quantitative.ps1`
- **Documentation**: `scripts/reinforcement_learning/sb3/EVALUATION_README.md`
- **Plots directory**: `evaluation_plots/`
