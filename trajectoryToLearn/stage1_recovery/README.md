# Stage 1: Recovery Trajectories (20-40M steps)

**Purpose**: Learn to approach from far distances → comfortable working zone

## Selection Criteria

- **Length**: 60-90 waypoints
- **Initial distance**: 1.5-2.0m (far start, requires base mobilization)
- **Target motion**: Static or very slow (<0.05 m/s)
- **Task**: Drive from far → 0.55-0.65m comfortable zone
- **Sectors**: All directions (360° coverage for approach scenarios)

## Goals

1. Policy learns "recovery strategy" when starting far from target
2. Base learns aggressive but controlled approach motion
3. Prevents reachability failures when episodes reset far out
4. Practices transition from mobilization → precision tracking

## Trajectory Design

**Recovery Drill Format**:
```
Waypoints 0-30:   Target static @ distance 1.5-2.0m
                  Base must drive closer
Waypoints 30-60:  Target still static, now at 0.55-0.65m
                  Base maintains position, arm tracks
Waypoints 60+:    Target moves slowly (<0.05 m/s)
                  Combined base-arm tracking
```

## Expected Metrics @ 40M

- Position error: <320cm mean (improvement from stage0)
- Orientation error: <70° mean (continued improvement)
- Workspace distance: 0.55-0.60m stable
- Approach success: >90% (from far starts)
