# Stage 3: Full Trajectories (70M-100M steps)

**Purpose**: Final policy refinement on hardest cases, full curriculum weights active

## Selection Criteria

- **Length**: ALL (30-300+ waypoints, full range)
- **Reach**: Full workspace (0.35-1.0m, including extremes)
- **Speed**: Up to 0.2 m/s (maximum challenge)
- **Sectors**: Full 360° coverage (all behind-base scenarios)
- **Orientation**: All difficulties (including >90° rotations)

## Goals

1. Polish policy on hardest trajectories
2. Ensure no regression on easy cases (continued training on all difficulties)
3. Maximize final performance with full weights (10.0, 30.0)
4. Prevent overfitting to any trajectory subset

## Stage 3 Characteristics

**Weight Status**: Full curriculum weights active
- Position: 10.0 (100% of final)
- Orientation: 30.0 (100% of final)
- Ratio: 1:3 maintained throughout

**Trajectory Mix**: All difficulties included
- 30% easy (stage0 level)
- 30% recovery (stage1 level)
- 20% moderate (stage2 level)
- 20% hard (long, fast, complex orientations)

## Expected Metrics @ 100M (Success Criteria)

**Minimum (Must Achieve)**:
- Position error: <300cm mean (match/beat 8g @ 40M and 8f baseline)
- Orientation error: <80° mean (50% improvement vs 8g @ 40M's 130°)
- Workspace distance: 0.55-0.60m sustained
- Training stability: No collapse, std<1.0 throughout

**Stretch (Ideal)**:
- Position error: <280cm mean (better than 8f's 308cm)
- Orientation error: <60° mean (approaching 8f's 46.5°)
- Reward: >-120k (better than 8f's -126k)
- Curriculum smooth: No instability during entire run

## If Failure Occurs

- Evaluate last 10 checkpoints (every 2M steps) to find stable model
- Use best pre-collapse checkpoint for deployment
- Analyze collapse timing vs trajectory stage switches
- Design Session 8i with adjusted curriculum or no curriculum
