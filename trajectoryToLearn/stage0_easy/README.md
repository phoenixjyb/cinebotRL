# Stage 0: Easy Trajectories (0-20M steps)

**Purpose**: Learn basic arm tracking + workspace positioning without discouragement

## Selection Criteria

- **Length**: 30-60 waypoints (short)
- **Reach**: 0.4-0.6m (comfortable zone only)
- **Speed**: <0.1 m/s (quasi-static, minimal velocity tracking challenge)
- **Sectors**: Front and sides only (no behind-base positions)
- **Orientation**: Gentle changes (<30° between waypoints)

## Goals

1. Policy learns to track arm motion at comfortable reach distance
2. Base learns to maintain 0.55-0.65m working distance
3. No "unreachable from start" failures (8g had 78% @ step 100!)
4. Build confidence before harder challenges

## Trajectory Sources

**TODO**: Populate with trajectories meeting criteria
- Option 1: Filter from existing `world_json/` trajectories
- Option 2: Generate synthetic easy trajectories (MATLAB)
- Option 3: Use chassis-only subset with limited reach

## Expected Metrics @ 20M

- Position error: <350cm mean
- Orientation error: <80° mean (baseline for Stage 1)
- Workspace distance: 0.50-0.65m converged
- Unreachable %: <10% (vs 8g's 78%!)
