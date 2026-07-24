# Riser Vendor-Source Reconciliation

This CPU-only evidence reconciles the current 750 W riser recommendation with
official motor, drive-family, and fixed-axis source snapshots retrieved on
2026-07-24.

The audit confirms the selected `ELVM8075V48EH-M17-HD` motor ratings and the
`ELD2-CAN7020B` drive pairing. It also records the model-specific safety
boundary: the dedicated CN6 STO interface documented for
`ELD2-CAN7040B/7060B` must not be attributed to the selected `7020B`.
Independent safety-rated power removal remains required.

`igus drylin ZLW-1080 Standard` is retained only as a fixed-axis concept
reference. Its catalog speed, stroke, lead, and radial-load figures do not
constitute supplier approval for this vertical mobile-axis duty.

The audit does not select an exact gearbox, production axis, regeneration
path, anti-fall device, or safety-power architecture. It does not authorize
procurement, hardware transfer, simulation-plant switching, capture, BC, PPO,
or training.
