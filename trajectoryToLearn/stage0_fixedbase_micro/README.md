# stage0_fixedbase_micro

Purpose: first Stage A curriculum for arm/gimbal tracking with base actions frozen.

- trajectories: 25
- waypoints per trajectory: 60
- waypoint dt in loader: 0.1 s
- nominal duration: 6.0 s
- center: [1.050, 0.080, 0.860]
- max local amplitudes: x=0.095 m, y=0.050 m, z=0.050 m
- orientation: constant xyzw [0.5, -0.5, 0.5, -0.5]

This stage is deliberately not cinematic. It tests whether the policy can learn
reachable fixed-base EE tracking before base-motion curricula are reintroduced.
