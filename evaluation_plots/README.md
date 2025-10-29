# Evaluation Plots Directory

This directory contains visualization plots for evaluation results, organized by model training session.

## Directory Structure

```
evaluation_plots/
├── <model_folder_name>/        # e.g., 20251028_200923
│   ├── tracking_errors.png      # Position & orientation error distributions
│   ├── joint_angles.png         # Joint angle utilization (6 DOF)
│   ├── joint_velocities.png     # Joint velocity profiles
│   ├── reward_components.png    # Reward breakdown bar chart
│   ├── episode_statistics.png   # Episode rewards & success rate
│   └── evaluation_report.txt    # Text summary with assessment
└── README.md                     # This file
```

## Plot Types

### 1. Tracking Errors (`tracking_errors.png`)
**What it shows**: Distribution of position and orientation tracking errors
- Position error histogram (cm) with CDF overlay
- Orientation error histogram (degrees) with CDF overlay
- Mean, median, P95 markers
- Success criteria reference lines

**Use for**: Assessing tracking accuracy, identifying failure modes

### 2. Joint Angles (`joint_angles.png`)
**What it shows**: Joint angle distribution for all 6 DOF
- 6 histograms (one per joint)
- Joint limits marked as red dashed lines
- Mean value markers
- Utilization percentage

**Use for**: Understanding workspace utilization, checking joint limit violations

### 3. Joint Velocities (`joint_velocities.png`)
**What it shows**: Joint velocity profiles for all 6 DOF
- 6 histograms (one per joint)
- Velocity limits marked
- Mean, max velocity annotations

**Use for**: Identifying jerky motion, velocity limit violations

### 4. Reward Components (`reward_components.png`)
**What it shows**: Breakdown of all reward components
- Bar chart with positive (green) and negative (red) components
- Sorted by absolute magnitude
- Mean values over all episodes

**Use for**: Understanding reward balance, identifying dominant penalties

### 5. Episode Statistics (`episode_statistics.png`)
**What it shows**: Episode-level performance metrics
- Episode reward distribution histogram
- Episode length distribution
- Success rate (if available)

**Use for**: Overall performance assessment, identifying outliers

### 6. Evaluation Report (`evaluation_report.txt`)
**What it shows**: Text summary with quantitative assessment
- Tracking accuracy statistics
- Episode statistics (mean/median/range)
- Assessment against success criteria
- Pass/fail indicators

**Use for**: Quick numerical summary, sharing results

## Usage

### Generate All Plots

```powershell
# For a specific evaluation
python scripts/reinforcement_learning/sb3/visualize_eval_results.py `
    --input evaluation_results/20251028_200923/eval_summary_<timestamp>.json `
    --output_dir evaluation_plots

# Plots will be saved to: evaluation_plots/20251028_200923/
```

### View Plots

```powershell
# Open all plots
start evaluation_plots/20251028_200923/*.png

# Or view specific plot
start evaluation_plots/20251028_200923/tracking_errors.png
```

## Model Folders

### Available Models

- **`20251028_200923/`** - Session 7d (200M timesteps)
  - Status: ⚠️ **NOT READY FOR DEPLOYMENT**
  - Key issues: Poor tracking accuracy, reward imbalance
  - See `evaluation_results/20251028_200923/ANALYSIS_REPORT.md` for details

## Interpreting Plots

### Good vs. Bad Signs

**✅ Good signs:**
- Position errors clustered below 20 cm
- Orientation errors below 10°
- Joint angles well-distributed (not hitting limits)
- Joint velocities smooth (no spikes)
- Positive rewards > negative penalties
- Episode rewards concentrated near positive values

**❌ Bad signs:**
- Position errors spread to meters
- Orientation errors > 90° (pointing wrong way)
- Joint angles hitting limits frequently
- Joint velocities spiking (jerky motion)
- Penalties overwhelming rewards
- Episode rewards highly negative
- High variance in episode performance

### Example Analysis

For **Session 7d (20251028_200923)**:

**Tracking Errors Plot**:
- ❌ Position errors: Mean 3.64m, heavily right-skewed
- ❌ Orientation errors: Mean 140.7°, robot pointing backwards
- **Diagnosis**: Policy not tracking trajectories effectively

**Joint Velocities Plot**:
- ❌ Joint 5 showing spikes to 4.0 rad/s
- **Diagnosis**: Velocity limit violations causing penalties

**Reward Components Plot**:
- ❌ Velocity penalty: -15.5 (massive)
- ❌ Jerk penalty: -14.0 (huge)
- ✅ Position tracking: +27.7 (good, but overwhelmed)
- **Diagnosis**: Reward imbalance - penalties dominate

**Conclusion**: Retrain with adjusted reward weights

## Related Files

- **Evaluation results**: `evaluation_results/<model_folder>/`
- **Analysis report**: `evaluation_results/<model_folder>/ANALYSIS_REPORT.md`
- **Visualization script**: `scripts/reinforcement_learning/sb3/visualize_eval_results.py`
- **Evaluation documentation**: `scripts/reinforcement_learning/sb3/EVALUATION_README.md`
