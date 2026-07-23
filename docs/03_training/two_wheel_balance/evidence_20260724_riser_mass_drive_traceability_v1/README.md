# Riser mass and drive traceability v1

CPU-only evidence generated from the current URDF, provisional plant prior,
hardware-envelope calculation, pinned 750 W vendor snapshot, active 400 W
simulation profile, and unmeasured bench template.

The audit passes calculation traceability and supplier/bench design-review
readiness. It does not authorize procurement, hardware transfer, runtime,
capture, BC, PPO, or training.

Key values:

- whole robot: `28.000 kg`;
- modeled riser moving subtree: `4.342 kg`;
- conservative drive sizing mass: `8.000 kg`;
- calculated 15%-margin moving-mass ceiling: `14.803715 kg`;
- 750 W candidate emergency force margin at 8 kg: `1.986781`;
- camera working range: `0.60--1.80 m`;
- target speed: `1.0 m/s`;
- recommended mechanical stroke: `1.50 m`, without extending camera height.

`summary.json` SHA-256:
`52ddff232c339d5cd3057cf680a98ce19939150943d9c45e21f14836d28c507a`.
