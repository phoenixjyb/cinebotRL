# Stage 2: Moderate Trajectories (40-70M steps)

**Purpose**: Generalize to longer trajectories, moderate challenges, weight transition 45-55M

## Selection Criteria

- **Length**: 60-120 waypoints (medium duration)
- **Reach**: 0.5-0.8m (expanded range, some stretching)
- **Speed**: <0.15 m/s (moderate velocity tracking)
- **Sectors**: 270° coverage (includes some behind-base positions)
- **Orientation**: Moderate changes (30-60° between waypoints)

## Goals

1. Extend tracking to longer episodes without performance degradation
2. Practice behind-base positions (challenging for mobile manipulator)
3. **Navigate curriculum weight transition** (45-55M linear ramp)
4. Refine precision as weights increase to full (10.0, 30.0)

## Critical Period: Weight Transition (45-55M)

**Monitoring during 45-55M**:
- Watch for KL divergence spikes (threshold: 0.1)
- Check explained variance stays positive
- Position/orientation errors should stay stable or improve
- If instability detected → auto-pause triggers

**Session 8g lesson**: Instant switch @ 50M caused collapse. Session 8h uses gradual 10M ramp.

## Expected Metrics @ 70M

- Position error: <300cm mean (approaching 8f baseline 308cm)
- Orientation error: <60° mean (substantial improvement)
- Workspace distance: 0.55-0.60m locked in
- Training stability: KL<0.05, variance>0.5 throughout transition
