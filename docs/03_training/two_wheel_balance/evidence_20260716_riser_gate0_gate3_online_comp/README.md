# Riser Gate 0-3 online-compensation evidence

This directory is the accepted deterministic Isaac milestone for the arm-free
two-wheel riser robot. The source teacher remains semantic DFR attitude, the
reward/observation frame remains physical `cam_link`, and PPO remains blocked.

The key correction is online full-root attitude compensation. Offline proxy
angles assume an upright chassis, but the balancing base rolls and pitches.
The simulated adapter therefore solves the proxy coordinates from the current
root quaternion and desired world DFR attitude at 200 Hz. The semantic command
sequence remains rate-audited at 24 deg/s; internal stabilization is separately
bounded by the RS4 hard envelope of 360 deg/s.

Accepted evidence files:

- `gate0_asset.json`: regenerated articulation, mass, DOF, frame, and USD-limit gate.
- `gate1_static_heights.json`: balance at camera heights 0.6, 0.9, and 1.8 m.
- `gate2_riser_dynamic.json`: complete up/down moves at 0.1, 0.25, 0.5, and 1.0 m/s.
- `gate3_representative_playback.json`: non-rendered cases 1, 31, and 73.
- `gate3_render_case_*.json`: D3D12 offscreen render-time metric gates.
- `summary.json`: compact accepted metrics and video hashes.

The next gate is corrected regeneration and deterministic evaluation of the
remaining 17 trajectories. This evidence does not authorize PPO or residual
training by itself.
